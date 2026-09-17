"""Shared CUDA FP32 search inference and per-model cache ownership."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import time

import torch

from . import cozy_bridge, search
from .batch_scheduler import WorkRequest
from .gumbel_search import gumbel_stepwise, GumbelConfig
from .merged_executors import _make_root_eval_executor, _make_decode_wave_executor
from .position_evaluator import CachedPositionEvaluator, _project_legal_logits
from .search import PositionEval, HalvingConfig


@dataclass(frozen=True)
class HalvingResult:
    move_uci: str
    candidates: list[dict]


def load_runtime(
    *,
    repo_config,
    checkpoint,
    device="cuda",
    algorithm="gumbel",
    root_batch_tokens=1024,
    stats=None,
):
    from imba_chess.data.board_state import BoardStateEncoder
    from imba_chess.data.move_vocab import MoveVocab
    from .position_evaluator import load_hstu_checkpoint

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("production search requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    vocab = MoveVocab.load(repo_config.vocab.path)
    model, _ = load_hstu_checkpoint(
        checkpoint_path=Path(checkpoint),
        repo_config=repo_config,
        move_vocab=vocab,
        device=device,
        compile_model=False,
        require_value_head=True,
    )
    runtime = InferenceRuntime(
        model=model,
        move_vocab=vocab,
        encoder=BoardStateEncoder(repo_config.board_state),
        device=device,
        algorithm=algorithm,
        root_batch_tokens=root_batch_tokens,
        stats=stats,
    )
    return runtime, model.config.max_position_embeddings


class InferenceRuntime:
    def __init__(
        self,
        *,
        model,
        move_vocab,
        encoder,
        device,
        root_batch_tokens=1024,
        algorithm="gumbel",
        stats=None,
    ):
        if algorithm not in ("gumbel", "value_search_halving"):
            raise ValueError("unsupported search algorithm")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("production search requires CUDA")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if model.training:
            raise ValueError("search requires an evaluation-mode model")
        if any(
            p.dtype != torch.float32 or p.device != device for p in model.parameters()
        ):
            raise ValueError("search requires FP32 weights on the runtime device")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model, self.move_vocab, self.encoder, self.device = (
            model,
            move_vocab,
            encoder,
            device,
        )
        self.algorithm = algorithm
        self._cache_token = object()
        self.options = dict(
            algorithm=algorithm,
            dtype="float32",
            tf32=False,
            runtime_revision="shared-search-v1",
        )
        self.waves = dict(root_eval=Counter(), decode_wave=Counter())
        self.seconds = Counter()
        self.inference_rows = Counter()
        self.executors = {}
        for kind, executor in dict(
            root_eval=_make_root_eval_executor(
                model=model,
                device=device,
                dtype=torch.float32,
                stats=stats,
                max_tokens=root_batch_tokens,
            ),
            decode_wave=_make_decode_wave_executor(
                model=model,
                device=device,
                dtype=torch.float32,
                stats=stats,
                algorithm=algorithm,
            ),
        ).items():
            self.executors[kind] = self._identified(kind, executor)

    def _identified(self, kind, executor):
        def run(payloads):
            owners = [p[0] for p in payloads]
            if len(set(owners)) != len(owners):
                raise ValueError("more than one outstanding request per game")
            if kind == "decode_wave" and any(
                getattr(payload[1][0], "_runtime_token", None) is not self._cache_token
                for payload in payloads
            ):
                raise RuntimeError("stale or foreign inference owner")
            start = time.perf_counter()
            result = executor([p[1] for p in payloads])
            if len(result) != len(owners):
                raise RuntimeError("inference result count mismatch")
            self.seconds[kind] += time.perf_counter() - start
            self.waves[kind][len(payloads)] += 1
            self.inference_rows[kind] += (
                len(payloads)
                if kind == "root_eval"
                else sum(len(p[1][1]) for p in payloads)
            )
            return list(zip(owners, result))

        run.clear_cache = getattr(executor, "clear_cache", lambda: None)
        run.workspace = getattr(executor, "workspace", None)
        return run

    def clear_caches(self):
        self._cache_token = object()
        for executor in self.executors.values():
            executor.clear_cache()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.clear_caches()

    def search(
        self,
        *,
        board,
        history,
        actor_id,
        game_id,
        config,
        rng=None,
        should_stop=lambda: False,
        noise=None,
    ):
        owner = (actor_id, game_id)
        batch = history.build_batch_for_current_position(board)
        batch["game_id"] = [game_id]
        return (
            yield from self.search_batch(
                board=board,
                batch=batch,
                owner=owner,
                config=config,
                rng=rng,
                should_stop=should_stop,
                noise=noise,
            )
        )

    def search_batch(
        self,
        *,
        board,
        batch,
        owner,
        config,
        rng=None,
        should_stop=lambda: False,
        noise=None,
        root_observer=None,
    ):
        if self.model.training:
            raise ValueError("search requires an evaluation-mode model")
        expected = GumbelConfig if self.algorithm == "gumbel" else HalvingConfig
        if not isinstance(config, expected):
            raise ValueError("search config does not match runtime algorithm")
        if isinstance(config, GumbelConfig) and config.max_depth > 32:
            raise ValueError("Gumbel workspace supports depth <= 32")
        if should_stop():
            raise InterruptedError("search cancelled")
        token = self._cache_token
        identity, output = yield WorkRequest("root_eval", (owner, batch))
        if token is not self._cache_token:
            raise RuntimeError("search invalidated by cache refresh")
        if identity != owner:
            raise RuntimeError("root result ownership mismatch")
        logits, moves, total, mapped = _project_legal_logits(
            logits=output["logits"][-1], board=board, move_vocab=self.move_vocab
        )
        if total != mapped:
            raise ValueError("incomplete legal vocabulary")
        if not moves:
            raise ValueError("cannot search a position without legal moves")
        if root_observer is not None:
            root_observer(moves, logits)
        if noise == 0.0:
            noise = [0.0] * len(moves)
        wdl = tuple(torch.softmax(output["value_logits"][-1].float(), -1).tolist())
        root = PositionEval(
            wdl[2] - wdl[0],
            [cozy_bridge.py_move_to_cozy(board, m) for m in moves],
            [m.uci() for m in moves],
            torch.log_softmax(logits.float(), 0).tolist(),
            [False] * len(moves),
            [self.move_vocab.encode(m.uci()) for m in moves],
        )
        evaluator = CachedPositionEvaluator(
            model=self.model,
            move_vocab=self.move_vocab,
            board_state_encoder=self.encoder,
            device=self.device,
            dtype=torch.float32,
            prefix_kv=output["kv_caches"],
            prefix_len=batch["total_tokens"],
            immutable_prefix=True,
        )
        evaluator._runtime_token = token
        if self.algorithm == "gumbel":
            gen = gumbel_stepwise(
                board=board,
                extend=evaluator.extend,
                config=config,
                rng=rng,
                root_eval=root,
                root_wdl=wdl,
                noise=noise,
                should_stop=should_stop,
            )
        else:
            if noise is not None:
                raise ValueError("explicit Gumbel noise is not a halving option")
            gen = search._halving_stepwise(
                extend=evaluator.extend,
                root_handle=None,
                board=board,
                legal_moves=moves,
                legal_log_priors=root.legal_log_priors,
                config=config,
                rng=rng,
            )
        del output, batch
        try:
            request = next(gen)
            while True:
                if should_stop():
                    raise InterruptedError("search cancelled")
                identity, result = yield WorkRequest(
                    "decode_wave", (owner, (evaluator, request.batch))
                )
                if token is not self._cache_token:
                    raise RuntimeError("search invalidated by cache refresh")
                if identity != owner:
                    raise RuntimeError("leaf result ownership mismatch")
                request = gen.send(result)
        except StopIteration as stop:
            if self.algorithm == "value_search_halving":
                chosen, rows = stop.value
                return HalvingResult(moves[chosen].uci(), rows)
            return stop.value
        finally:
            gen.close()
