"""Shared CUDA search inference (FP32 or BF16 weights) and per-model cache ownership."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import time

import chess
import torch

from . import cozy_bridge, search
from .batch_scheduler import WorkRequest
from .gumbel_search import gumbel_stepwise, GumbelConfig
from .puct_search import puct_stepwise, PuctConfig, PUCT_ALGORITHMS
from .merged_executors import _make_root_eval_executor, _make_decode_wave_executor
from .position_evaluator import CachedPositionEvaluator
from .search import PositionEval, HalvingConfig


INFERENCE_DTYPES = dict(float32=torch.float32, bfloat16=torch.bfloat16)
RUNTIME_REVISION = "shared-search-v2"
SINGLE_LEAF_ALGORITHMS = ("gumbel", *PUCT_ALGORITHMS)


@dataclass(frozen=True)
class HalvingResult:
    move_uci: str
    candidates: list[dict]


# KLENT's improved policy pi' ∝ exp((beta * logit + q) / (alpha + beta)); its
# argmax depends only on beta * logit + q. Reference beta (KazukiOhta/klent).
POLICY_Q_BETA = 0.1


@dataclass(frozen=True)
class _RootCarry:
    """The played child of the previous search, ready to be the next root.

    The child was decoded as the token right after the previous prefix, so it
    is exactly the next root token: its evaluation is the next root's, and the
    next prefix is the previous one plus the child's K/V row. Valid only for
    the same runtime weights (token), game owner and history position."""

    token: object
    owner: tuple
    fen: str
    move_id: int
    prefix_len: int
    prefix_kv: list
    evaluation: object


@dataclass(frozen=True)
class PolicyResult:
    """Greedy move with no search: argmax of the legal policy logits
    ("policy", the KLENT paper's eval) or of KLENT's improved policy pi',
    beta * logit + q ("policy_q", uses the action-value head)."""

    move_uci: str
    simulations: int = 0
    neural_evaluations: int = 1
    terminal_hits: int = 0
    depth_cutoffs: int = 0


def load_runtime(
    *,
    repo_config,
    checkpoint,
    device="cuda",
    algorithm="gumbel",
    root_batch_tokens=1024,
    stats=None,
    dtype=torch.float32,
    value_source="head",
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
    model.to(dtype)
    if value_source not in ("head", "pi_q"):
        raise ValueError("value_source must be 'head' or 'pi_q'")
    if value_source == "pi_q" and getattr(model, "action_value_head", None) is None:
        raise ValueError("value_source 'pi_q' needs a checkpoint with an action-value head")
    # Every root and leaf value the search reads comes from this source.
    model.value_source = value_source
    runtime = InferenceRuntime(
        model=model,
        move_vocab=vocab,
        encoder=BoardStateEncoder(repo_config.board_state),
        device=device,
        algorithm=algorithm,
        root_batch_tokens=root_batch_tokens,
        stats=stats,
        dtype=dtype,
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
        dtype=torch.float32,
    ):
        if algorithm not in (*SINGLE_LEAF_ALGORITHMS, "value_search_halving", "policy", "policy_q"):
            raise ValueError("unsupported search algorithm")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("production search requires CUDA")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if model.training:
            raise ValueError("search requires an evaluation-mode model")
        if dtype not in INFERENCE_DTYPES.values():
            raise ValueError("search runs in float32 or bfloat16")
        if any(p.dtype != dtype or p.device != device for p in model.parameters()):
            raise ValueError(f"search requires {dtype} weights on the runtime device")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model, self.move_vocab, self.encoder, self.device = (
            model,
            move_vocab,
            encoder,
            device,
        )
        self.algorithm, self.dtype = algorithm, dtype
        self._cache_token = object()
        root_executor = _make_root_eval_executor(
            model=model, device=device, dtype=dtype, stats=stats,
            max_tokens=root_batch_tokens,
        )
        self.options = dict(
            algorithm=algorithm,
            dtype=str(dtype).removeprefix("torch."),
            tf32=False,
            runtime_revision=RUNTIME_REVISION,
            root_compiled=root_executor.compile_enabled,
        )
        self.waves = dict(root_eval=Counter(), decode_wave=Counter())
        self.seconds = Counter()
        self.inference_rows = Counter()
        self.executors = {}
        for kind, executor in dict(
            root_eval=root_executor,
            **({} if algorithm in ("policy", "policy_q") else dict(decode_wave=_make_decode_wave_executor(
                model=model,
                device=device,
                dtype=dtype,
                stats=stats,
                algorithm=algorithm,
            ))),
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
        reused = None
        if self.algorithm in SINGLE_LEAF_ALGORITHMS:
            reused = self._take_root_carry(history, owner, board, *history.next_root_key())
        batch = None
        if reused is None:
            # Only a root prefill needs the whole history as tensors.
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
                carry=history,
                reused=reused,
            )
        )

    def _take_root_carry(self, holder, owner, board, total_tokens, prev_move_id):
        """Pop the holder's carry; return it if it is this exact root."""
        carry, holder.root_carry = holder.root_carry, None
        if (
            carry is None
            or carry.token is not self._cache_token
            or carry.owner != owner
            or carry.prefix_len != total_tokens
            or carry.move_id != prev_move_id
            or carry.fen != board.fen()
        ):
            return None
        return carry

    def _root_carry(self, owner, token, board, result, played, prefix_kv, prefix_len):
        child = played[0] if played else None
        if child is None or child.evaluation is None or not child.handle.arena_chain:
            return None
        workspace = self.executors["decode_wave"].workspace
        rows = workspace.arena_stack[:, :, child.handle.arena_chain[-1]]
        if len(prefix_kv) != rows.size(0) or any(
            k.dtype != rows.dtype or k.dim() != 3 for k, _ in prefix_kv
        ):
            raise RuntimeError("root prefix does not match the decode arena")
        # Copies, not views: the arena row is reclaimed with this search.
        prefix = [
            (torch.cat((k, rows[i, 0][:, None]), 1), torch.cat((v, rows[i, 1][:, None]), 1))
            for i, (k, v) in enumerate(prefix_kv)
        ]
        after = board.copy(stack=False)
        after.push_uci(result.move_uci)
        return _RootCarry(
            token, owner, after.fen(), int(result.move_id), prefix_len + 1, prefix,
            child.evaluation,
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
        carry=None,
        reused=None,
    ):
        """carry: the game's history object. Single-leaf searches leave the played
        child on it so the next search of the same game reuses that decode as
        its root instead of re-running the whole history (see _RootCarry).
        reused: a carry already validated for this root by search(); batch is
        then None."""
        if self.model.training:
            raise ValueError("search requires an evaluation-mode model")
        expected = (GumbelConfig if self.algorithm == "gumbel" else
                    PuctConfig if self.algorithm in PUCT_ALGORITHMS else HalvingConfig)
        if self.algorithm not in ("policy", "policy_q") and not isinstance(config, expected):
            raise ValueError("search config does not match runtime algorithm")
        if isinstance(config, PuctConfig) and config.algorithm != self.algorithm:
            raise ValueError("PUCT config does not match runtime algorithm")
        if isinstance(config, (GumbelConfig, PuctConfig)) and config.max_depth > 32:
            raise ValueError("single-leaf workspace supports depth <= 32")
        if should_stop():
            raise InterruptedError("search cancelled")
        token = self._cache_token
        if reused is not None and (reused.token is not token or root_observer is not None):
            raise ValueError("a reused root needs the current weights and no root observer")
        if reused is None:
            identity, output = yield WorkRequest("root_eval", (owner, batch))
            if token is not self._cache_token:
                raise RuntimeError("search invalidated by cache refresh")
            if identity != owner:
                raise RuntimeError("root result ownership mismatch")
            # Native projection: vocabulary ids, UCIs, playable moves and forcing
            # flags in canonical UCI order, the order python-chess projection used.
            legal = cozy_bridge.project_legal_lean(
                cozy_bridge.board_to_cozy(board), self.move_vocab
            )
            if legal.total != len(legal):
                raise ValueError("incomplete legal vocabulary")
            if not len(legal):
                raise ValueError("cannot search a position without legal moves")
            ids, ucis = legal.ids, legal.ucis
            row = output["logits"][-1]
            index = torch.tensor(ids, dtype=torch.long).to(row.device, non_blocking=True)
            logits = row.index_select(0, index)
            if root_observer is not None:
                root_observer([chess.Move.from_uci(uci) for uci in ucis], logits)
            if self.algorithm == "policy":
                return PolicyResult(ucis[int(torch.argmax(logits))])
            if self.algorithm == "policy_q":
                if "q" not in output:
                    raise ValueError("policy_q requires a checkpoint with an action-value head")
                q = output["q"][-1].index_select(0, index)
                scores = POLICY_Q_BETA * logits.float() + q.float()
                return PolicyResult(ucis[int(torch.argmax(scores))])
            if noise == 0.0:
                noise = [0.0] * len(ids)
            # One readback for the root's log priors and WDL.
            host = torch.cat(
                (
                    torch.log_softmax(logits.float(), 0),
                    torch.softmax(output["value_logits"][-1].float(), -1),
                )
            ).cpu()
            wdl = tuple(host[len(ids):].tolist())
            root = PositionEval(
                wdl[2] - wdl[0],
                legal.moves,
                ucis,
                host[: len(ids)].tolist(),
                legal.forcing,
                ids,
                wdl=wdl,
            )
            prefix_kv, prefix_len = output["kv_caches"], batch["total_tokens"]
        else:
            # The previous search already decoded this exact token (its played
            # child) against the same prefix with the same weights.
            root, prefix_kv, prefix_len = reused.evaluation, reused.prefix_kv, reused.prefix_len
            wdl = tuple(root.wdl)
            output = None
            if noise == 0.0:
                noise = [0.0] * len(root.legal_ids)
        evaluator = CachedPositionEvaluator(
            model=self.model,
            move_vocab=self.move_vocab,
            board_state_encoder=self.encoder,
            device=self.device,
            dtype=self.dtype,
            prefix_kv=prefix_kv,
            prefix_len=prefix_len,
            immutable_prefix=True,
        )
        evaluator._runtime_token = token
        played = []
        if self.algorithm in SINGLE_LEAF_ALGORITHMS:
            stepwise = gumbel_stepwise if self.algorithm == "gumbel" else puct_stepwise
            gen = stepwise(
                board=board,
                extend=evaluator.extend,
                config=config,
                rng=rng,
                root_eval=root,
                root_wdl=wdl,
                noise=noise,
                should_stop=should_stop,
                on_played=played.append if carry is not None else None,
            )
        else:
            if noise is not None:
                raise ValueError("explicit Gumbel noise is not a halving option")
            gen = search._halving_stepwise(
                extend=evaluator.extend,
                root_handle=None,
                board=board,
                legal_moves=[chess.Move.from_uci(uci) for uci in ucis],
                legal_log_priors=root.legal_log_priors,
                config=config,
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
                return HalvingResult(ucis[chosen], rows)
            if carry is not None:
                carry.root_carry = self._root_carry(
                    owner, token, board, stop.value, played, prefix_kv, prefix_len
                )
            return stop.value
        finally:
            gen.close()
