from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import chess
import chess.engine
import pytest

torch = pytest.importorskip("torch")
from imba_chess.config import ModelConfig, RepoConfig
from imba_chess.data.board_state import BoardStateEncoder
from imba_chess.data.move_vocab import MoveVocab, MoveVocabConfig
from imba_chess.eval.position_evaluator import _forward_model
from imba_chess.model import HSTUChessModel, build_hstu_chess_config

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_VOCAB_PATH = REPO_ROOT / "artifacts" / "move_vocab_static_uci.json"


def _load_eval_script_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "eval_vs_stockfish.py"
    )
    spec = importlib.util.spec_from_file_location(
        "eval_vs_stockfish_script", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load eval_vs_stockfish.py module for testing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "coverage,q,concurrency,use_cli",
    [
        (True, 2, 1, False),
        (False, 0, 2, False),
        (False, 0, 1, True),
        (True, 2, 2, True),
    ],
)
def test_tactical_config_cli_and_result_roundtrip(
    tmp_path, monkeypatch, coverage, q, concurrency, use_cli
):
    module = _load_eval_script_module()
    config_path = tmp_path / "experiment.toml"
    config_coverage = not coverage if use_cli else coverage
    config_q = (0 if q else 2) if use_cli else q
    config_path.write_text(
        f'[eval_vs_stockfish]\nmodel_move_policy = "value_search_halving"\nsearch_tactical_coverage = {str(config_coverage).lower()}\nsearch_quiescence_plies = {config_q}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "result.json"
    argv = [
        "eval_vs_stockfish.py",
        "--config",
        str(config_path),
        "--checkpoint",
        "unused.pt",
        "--stockfish-path",
        sys.executable,
        "--device",
        "cpu",
        "--no-save-games",
        "--ladder-elos",
        "2200",
        "--ladder-games-per-segment",
        "1",
        "--no-include-full-strength-segment",
        "--concurrent-games",
        str(concurrency),
        "--output-json",
        str(output_path),
    ]
    if use_cli:
        argv += [
            "--search-tactical-coverage"
            if coverage
            else "--no-search-tactical-coverage",
            "--search-quiescence-plies",
            str(q),
        ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        module,
        "load_runtime",
        lambda **kwargs: (
            SimpleNamespace(
                model=None, move_vocab=_mini_vocab(), encoder=BoardStateEncoder()
            ),
            128,
        ),
    )
    observed = []

    def run_segment(**kwargs):
        cfg = kwargs["halving_config"]
        observed.append(cfg)
        assert cfg.tactical_coverage is coverage
        assert cfg.quiescence_plies == q
        return module.EvalSummary(
            games=1,
            completed_games=1,
            draws=1,
            model_turns=2,
            model_selection_seconds=0.4,
            search_stats={"evals_spent": 9, "quiescence_evals": q, "max_depth": 4 + q},
        )

    monkeypatch.setattr(module, "_run_segment", run_segment)
    module.main()
    assert len(observed) == 1
    payload = json.loads(output_path.read_text())
    for result in (payload["segments"][0]["results"], payload["aggregate"]):
        assert result["run_config"]["search"]["search_tactical_coverage"] is coverage
        assert result["run_config"]["search"]["search_quiescence_plies"] == q
        assert result["search_stats"]["quiescence_evals"] == q
        assert result["search_stats"]["max_depth"] == 4 + q
        assert result["mean_model_selection_seconds"] == pytest.approx(0.2)


def test_search_stats_merge_sums_counts_but_takes_maximum_depth():
    module = _load_eval_script_module()
    merged = module._merge_summaries(
        [
            module.EvalSummary(search_stats={"evals_spent": 7, "max_depth": 4}),
            module.EvalSummary(search_stats={"evals_spent": 9, "max_depth": 3}),
        ]
    )
    assert merged.search_stats == {"evals_spent": 16, "max_depth": 4}


def test_negative_quiescence_rejected_before_loading_checkpoint(monkeypatch):
    module = _load_eval_script_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_vs_stockfish.py",
            "--checkpoint",
            "unused.pt",
            "--search-quiescence-plies",
            "-1",
        ],
    )
    with pytest.raises(ValueError, match="search-quiescence-plies"):
        module.main()


def _dummy_kv(total_tokens: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(torch.zeros(1, total_tokens, 1), torch.zeros(1, total_tokens, 1))]


def _dummy_decode_kv(batch_size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(torch.zeros(batch_size, 1, 1, 1), torch.zeros(batch_size, 1, 1, 1))]


class _DummyNoValueModel(torch.nn.Module):
    def __init__(self, move_vocab: MoveVocab) -> None:
        super().__init__()
        self.move_vocab = move_vocab

    def forward(self, batch, *, block_mask=None, return_loss=False, return_kv=False):
        total_tokens = int(batch["total_tokens"])
        logits = torch.zeros((total_tokens, len(self.move_vocab)), dtype=torch.float32)
        last = total_tokens - 1
        logits[last, self.move_vocab.token_to_id["e2e4"]] = 1.0
        logits[last, self.move_vocab.token_to_id["d2d4"]] = 0.5
        out = {"logits": logits}
        if return_kv:
            out["kv_caches"] = _dummy_kv(total_tokens)
        return out


def _mini_repo_config() -> RepoConfig:
    return RepoConfig(
        model=ModelConfig(
            model_dim=64,
            linear_hidden_dim=16,
            attention_dim=16,
            num_heads=1,
            num_layers=0,
            dropout=0.0,
            max_position_embeddings=128,
            enable_value_head=False,
        )
    )


def _mini_vocab() -> MoveVocab:
    return MoveVocab.build(
        ["e2e4", "d2d4", "e7e5", "d7d5"], config=MoveVocabConfig(include_unk=False)
    )


class _DummyMatePreferenceModel(torch.nn.Module):
    """Prefers a quiet move by policy logit; only the value modes should find mate."""

    def __init__(self, move_vocab: MoveVocab) -> None:
        super().__init__()
        self.move_vocab = move_vocab
        self.forward_calls = 0

    def forward(self, batch, *, block_mask=None, return_loss=False, return_kv=False):
        self.forward_calls += 1
        total_tokens = int(batch["total_tokens"])
        logits = torch.zeros((total_tokens, len(self.move_vocab)), dtype=torch.float32)
        value_logits = torch.zeros((total_tokens, 3), dtype=torch.float32)
        last = total_tokens - 1
        logits[last, self.move_vocab.token_to_id["a1b1"]] = 4.0
        logits[last, self.move_vocab.token_to_id["a1a8"]] = 1.0
        out = {"logits": logits, "value_logits": value_logits}
        if return_kv:
            out["kv_caches"] = _dummy_kv(total_tokens)
        return out

    def forward_decode(
        self,
        *,
        new_token_batch,
        positions,
        prefix_kv,
        suffix_kv=None,
        suffix_positions=None,
        suffix_mask=None,
    ):
        self.forward_calls += 1
        batch_size = int(positions.numel())
        return {
            "logits": torch.zeros(
                (batch_size, len(self.move_vocab)), dtype=torch.float32
            ),
            "value_logits": torch.zeros((batch_size, 3), dtype=torch.float32),
            "kv": _dummy_decode_kv(batch_size),
        }


def _mate_in_one_setup():
    module = _load_eval_script_module()
    move_vocab = MoveVocab.build(
        ["a1a8", "a1b1"], config=MoveVocabConfig(include_unk=False)
    )
    model = _DummyMatePreferenceModel(move_vocab)
    history = module._SequenceHistory(
        move_vocab=move_vocab, board_state_encoder=BoardStateEncoder()
    )
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1")
    batch = history.build_batch_for_current_position(board)
    return (module, move_vocab, model, history, board, batch)


def test_stockfish_label_formats_limited_and_full_strength():
    module = _load_eval_script_module()
    assert (
        module._stockfish_label(limit_strength=True, elo=1400) == "Stockfish (elo=1400)"
    )
    assert (
        module._stockfish_label(limit_strength=False, elo=None)
        == "Stockfish (full strength)"
    )


def test_outcome_label_covers_all_cases():
    module = _load_eval_script_module()
    assert (
        module._outcome_label(completed=False, result="*", model_color=chess.WHITE)
        == "incomplete"
    )
    assert (
        module._outcome_label(completed=True, result="1/2-1/2", model_color=chess.WHITE)
        == "draw"
    )
    assert (
        module._outcome_label(completed=True, result="1-0", model_color=chess.WHITE)
        == "model_win"
    )
    assert (
        module._outcome_label(completed=True, result="1-0", model_color=chess.BLACK)
        == "model_loss"
    )
    assert (
        module._outcome_label(completed=True, result="0-1", model_color=chess.BLACK)
        == "model_win"
    )


def test_save_traced_game_writes_pgn_and_html(tmp_path):
    module = _load_eval_script_module()
    board = chess.Board()
    for move_uci in ["e2e4", "e7e5"]:
        board.push_uci(move_uci)
    save_games_dir = tmp_path / "games"
    module._save_traced_game(
        board=board,
        model_color=chess.BLACK,
        result="*",
        completed=False,
        segment_name="sf_elo_1400",
        stockfish_label="Stockfish (elo=1400)",
        game_idx=1,
        save_games_dir=save_games_dir,
    )
    pgn_text = (save_games_dir / "sf_elo_1400_game002_incomplete.pgn").read_text(
        encoding="utf-8"
    )
    assert '[Event "sf_elo_1400"]' in pgn_text
    assert '[White "Stockfish (elo=1400)"]' in pgn_text
    assert '[Black "imba-chess"]' in pgn_text
    assert '[Result "*"]' in pgn_text
    assert "1. e4 e5" in pgn_text
    html_path = save_games_dir / "sf_elo_1400_game002_incomplete.html"
    assert html_path.read_text(encoding="utf-8").startswith("<!doctype html>")


class _DummyHalvingModel(torch.nn.Module):
    """Root policy prefers e2e4; value head says the d2d4 subtree is winning.

    Value is read from the side-to-move POV, so the sign is keyed on the new
    token's turn_id; the root move is recovered from the board's d4 square.
    """

    def __init__(self, move_vocab: MoveVocab) -> None:
        super().__init__()
        self.move_vocab = move_vocab
        self.forward_calls = 0

    def forward(self, batch, *, block_mask=None, return_loss=False, return_kv=False):
        self.forward_calls += 1
        total_tokens = int(batch["total_tokens"])
        logits = torch.zeros((total_tokens, len(self.move_vocab)), dtype=torch.float32)
        value_logits = torch.zeros((total_tokens, 3), dtype=torch.float32)
        last = total_tokens - 1
        logits[last, self.move_vocab.token_to_id["e2e4"]] = 4.0
        logits[last, self.move_vocab.token_to_id["d2d4"]] = 3.0
        out = {"logits": logits, "value_logits": value_logits}
        if return_kv:
            out["kv_caches"] = _dummy_kv(total_tokens)
        return out

    def forward_decode(
        self,
        *,
        new_token_batch,
        positions,
        prefix_kv,
        suffix_kv=None,
        suffix_positions=None,
        suffix_mask=None,
    ):
        self.forward_calls += 1
        batch_size = int(positions.numel())
        logits = torch.zeros((batch_size, len(self.move_vocab)), dtype=torch.float32)
        value_logits = torch.zeros((batch_size, 3), dtype=torch.float32)
        piece_ids = new_token_batch["piece_ids"]
        turn_ids = new_token_batch["turn_id"]
        for row in range(batch_size):
            logits[row, self.move_vocab.token_to_id["e2e4"]] = 4.0
            logits[row, self.move_vocab.token_to_id["d2d4"]] = 3.0
            good_for_white = int(piece_ids[row, chess.D4].item()) == 1
            stm_is_white = int(turn_ids[row].item()) == 0
            if good_for_white == stm_is_white:
                value_logits[row] = torch.tensor([0.0, 0.0, 3.0])
            else:
                value_logits[row] = torch.tensor([3.0, 0.0, 0.0])
        return {
            "logits": logits,
            "value_logits": value_logits,
            "kv": _dummy_decode_kv(batch_size),
        }


def test_value_search_halving_end_to_end_picks_value_backed_move():
    module = _load_eval_script_module()
    from imba_chess.eval.search import HalvingConfig

    move_vocab = MoveVocab.build_static()
    model = _DummyHalvingModel(move_vocab)
    history = module._SequenceHistory(
        move_vocab=move_vocab, board_state_encoder=BoardStateEncoder()
    )
    board = chess.Board()
    batch = history.build_batch_for_current_position(board)
    move, debug = module._select_model_move(
        runtime=_halving_runtime(model, move_vocab),
        batch=batch,
        board=board,
        config=HalvingConfig(budget=6, top_m=2, rounds=2, lam=0.05),
    )
    assert move.uci() == "d2d4"
    assert debug["policy"] == "value_search_halving"
    rows = debug["value_search_halving_candidates"]
    assert {row["move_uci"] for row in rows} == {"e2e4", "d2d4"}


def _drive_model_move_stepwise(gen, *, model, device, dtype):
    """Manual next()/send() driver for _select_model_move_stepwise.

    Answers WorkRequest("root_eval", batch) via _forward_model(...,
    return_kv=True) -- the merged root_eval executor's own hardcoded
    contract (imba_chess.eval.merged_executors._make_root_eval_executor) --
    and WorkRequest("decode_wave", (evaluator, batch)) via evaluator.
    evaluate(batch) directly -- the merged decode_wave executor's
    len(payloads)==1 passthrough. This is exactly the single-game codepath
    BatchScheduler exercises at --concurrent-games 1, without needing a real
    BatchScheduler in these equivalence tests.
    """
    try:
        request = next(gen)
        while True:
            if request.kind == "root_eval":
                response = _forward_model(
                    model=model,
                    batch=request.payload[1],
                    device=device,
                    dtype=dtype,
                    return_kv=True,
                )
            elif request.kind == "decode_wave":
                evaluator, batch = request.payload[1]
                response = evaluator.evaluate(batch)
            else:
                raise AssertionError(f"unexpected WorkRequest kind: {request.kind!r}")
            request = gen.send((request.payload[0], response))
    except StopIteration as stop:
        return stop.value


def test_select_model_move_stepwise_matches_sync_for_value_search_halving():
    module = _load_eval_script_module()
    from imba_chess.eval.search import HalvingConfig

    move_vocab = MoveVocab.build_static()
    history = module._SequenceHistory(
        move_vocab=move_vocab, board_state_encoder=BoardStateEncoder()
    )
    board = chess.Board()
    batch = history.build_batch_for_current_position(board)
    halving_config = HalvingConfig(budget=6, top_m=2, rounds=2, lam=0.05)
    model_sync = _DummyHalvingModel(move_vocab)
    move_sync, debug_sync = module._select_model_move(
        runtime=_halving_runtime(model_sync, move_vocab),
        batch=batch,
        board=board,
        config=halving_config,
    )
    model_stepwise = _DummyHalvingModel(move_vocab)
    gen = module._select_model_move_stepwise(
        runtime=_halving_runtime(model_stepwise, move_vocab),
        batch=batch,
        board=board,
        config=halving_config,
    )
    move_stepwise, debug_stepwise = _drive_model_move_stepwise(
        gen, model=model_stepwise, device=torch.device("cpu"), dtype=torch.float32
    )
    assert debug_stepwise == debug_sync
    assert move_stepwise.uci() == move_sync.uci() == "d2d4"
    assert model_stepwise.forward_calls == model_sync.forward_calls
    assert {
        row["move_uci"] for row in debug_stepwise["value_search_halving_candidates"]
    } == {row["move_uci"] for row in debug_sync["value_search_halving_candidates"]}


class _FakeSFEngine:
    """Fake `chess.engine.SimpleEngine` double for the scheduler-driver
    tests below: no subprocess, no real UCI protocol. Records every
    `configure`/`play`/`quit` call; `play` always returns the board's first
    legal move (deterministic, works for any position) unless `play_exc` is
    set, in which case every `play` call raises it instead.
    """

    def __init__(self, *, play_exc: BaseException | None = None) -> None:
        self.configure_calls: list[dict] = []
        self.play_calls: list[tuple[str, object]] = []
        self.quit_calls = 0
        self._play_exc = play_exc

    def configure(self, options):
        self.configure_calls.append(dict(options))

    def play(self, board, limit):
        self.play_calls.append((board.fen(), limit))
        if self._play_exc is not None:
            raise self._play_exc
        move = next(iter(board.legal_moves))
        return chess.engine.PlayResult(move, None)

    def quit(self):
        self.quit_calls += 1


def _patch_fake_stockfish(monkeypatch, *, play_exc: BaseException | None = None):
    """Monkeypatch `chess.engine.SimpleEngine.popen_uci` (called by
    `_run_segment`'s `_spawn_engine` closure) to hand out `_FakeSFEngine`
    instances instead of spawning a real Stockfish subprocess. Returns the
    list of spawned fake engines (append-order == EnginePool spawn order,
    i.e. slot index) so tests can assert on per-slot call counts.
    """
    spawned: list[_FakeSFEngine] = []

    def _fake_popen_uci(command, **kwargs):
        engine = _FakeSFEngine(play_exc=play_exc)
        spawned.append(engine)
        return engine

    monkeypatch.setattr(
        chess.engine.SimpleEngine, "popen_uci", staticmethod(_fake_popen_uci)
    )
    return spawned


def _run_fake_segment(
    module, *, games: int, concurrent_games: int, spawned, model=None
):
    move_vocab = MoveVocab.build_static()
    board_state_encoder = BoardStateEncoder()
    model = model if model is not None else _tiny_model(move_vocab)
    return module._run_segment(
        stockfish_path=Path("fake-stockfish-binary"),
        segment_options={"Threads": 1},
        segment_name="fake-segment",
        model=model,
        move_vocab=move_vocab,
        board_state_encoder=board_state_encoder,
        games=games,
        max_plies=2,
        engine_limit=chess.engine.Limit(time=0.01),
        device=torch.device("cpu"),
        dtype=torch.float32,
        model_move_policy="value_search_halving",
        search_lambda=0.0,
        opening_random_plies=0,
        debug_trace_games=0,
        debug_trace_max_plies=0,
        debug_topk=0,
        stockfish_label="fake",
        save_games_dir=None,
        concurrent_games=concurrent_games,
        halving_config=module.HalvingConfig(budget=2, top_m=2, max_depth=2),
        runtime=_halving_runtime(model, move_vocab),
    )


def test_run_segment_scheduler_g1_aggregates_alternates_colors_and_reuses_engine(
    monkeypatch,
):
    module = _load_eval_script_module()
    spawned = _patch_fake_stockfish(monkeypatch)
    summary = _run_fake_segment(module, games=3, concurrent_games=1, spawned=spawned)
    assert len(spawned) == 1
    assert spawned[0].configure_calls == [{"Threads": 1}]
    assert spawned[0].quit_calls == 1
    assert len(spawned[0].play_calls) == 3
    assert summary.games == 3
    assert summary.incomplete_games == 3
    assert summary.completed_games == 0
    assert summary.games_as_white == 2
    assert summary.games_as_black == 1
    assert summary.total_plies == 3 * 2
    assert summary.model_turns == 3


def test_run_segment_scheduler_engine_exception_aborts_run(monkeypatch):
    module = _load_eval_script_module()
    spawned = _patch_fake_stockfish(
        monkeypatch, play_exc=RuntimeError("engine crashed mid-game")
    )
    with pytest.raises(RuntimeError, match="engine crashed mid-game"):
        _run_fake_segment(module, games=3, concurrent_games=1, spawned=spawned)
    assert len(spawned) == 1
    assert spawned[0].quit_calls == 1


def _tiny_model(move_vocab: MoveVocab) -> HSTUChessModel:
    torch.manual_seed(3)
    config = build_hstu_chess_config(
        ModelConfig(
            model_dim=32,
            linear_hidden_dim=8,
            attention_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            max_position_embeddings=64,
            enable_value_head=True,
        ),
        move_vocab_size=len(move_vocab),
    )
    return HSTUChessModel(config).eval()


@pytest.mark.parametrize(
    "policy",
    [
        "greedy",
        "value_rerank",
        "value_search_d2",
        "value_search_alphabeta",
        "value_search_pvs",
    ],
)
def test_retired_search_policy_rejected_by_cli(monkeypatch, policy):
    module = _load_eval_script_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval", "--checkpoint", "unused.pt", "--model-move-policy", policy],
    )
    with pytest.raises(SystemExit) as exc:
        module._parse_args()
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "option",
    [
        "--value-rerank-top-k",
        "--value-rerank-lambda",
        "--dtype",
        "--compile",
        "--search-lmr",
        "--search-score-cache",
        "--search-iterative-deepening",
    ],
)
def test_retired_search_options_rejected_by_cli(monkeypatch, option):
    module = _load_eval_script_module()
    monkeypatch.setattr(sys, "argv", ["eval", "--checkpoint", "unused.pt", option])
    with pytest.raises(SystemExit) as exc:
        module._parse_args()
    assert exc.value.code == 2


def _halving_runtime(model, vocab):
    from tests.search_references import InferenceRuntime

    return InferenceRuntime(
        model=model.eval(),
        move_vocab=vocab,
        encoder=BoardStateEncoder(),
        device="cpu",
        algorithm="value_search_halving",
    )
