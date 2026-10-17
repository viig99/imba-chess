"""The played child of one search becomes the next search's root.

Its decode is exactly the next root token against the same prefix, so a game
played move by move must see each reused root equal a fresh full-history
prefill (to float32 tolerance), and must skip the prefill whenever the carry
is valid."""

from dataclasses import replace

import chess
import torch

from imba_chess.eval.gumbel_search import GumbelConfig
from imba_chess.eval.position_evaluator import _SequenceHistory
from tests.search_references import InferenceRuntime
from tests.test_self_play import ENCODER, VOCAB, tiny_model


def runtime():
    torch.manual_seed(3)
    return InferenceRuntime(
        model=tiny_model().eval(), move_vocab=VOCAB, encoder=ENCODER,
        device=torch.device("cpu"),
    )


def drive(rt, gen, roots):
    """Run one search generator synchronously, counting root prefills."""
    request = next(gen)
    while True:
        if request.kind == "root_eval":
            roots.append(request.payload[0])
        ((identity, output),) = rt.executors[request.kind]([request.payload])
        try:
            request = gen.send((identity, output))
        except StopIteration as stop:
            return stop.value


def prefill(rt, history, board, owner):
    batch = history.build_batch_for_current_position(board)
    ((_, output),) = rt.executors["root_eval"]([(owner, batch)])
    return output


def play(rt, history, board, owner, plies, roots, config):
    results = []
    for _ in range(plies):
        result = drive(
            rt,
            rt.search(board=board, history=history, actor_id=owner[0], game_id=owner[1],
                      config=config, rng=None, noise=0.0),
            roots,
        )
        results.append(result)
        history.append_observed_position(board)
        history.record_played_move(result.move_uci)
        board.push_uci(result.move_uci)
    return results


def test_played_child_is_reused_as_the_next_root_and_matches_a_prefill():
    rt = runtime()
    config = GumbelConfig(simulations=12, top_m=4, max_depth=4)
    board = chess.Board()
    history = _SequenceHistory(move_vocab=VOCAB, board_state_encoder=ENCODER)
    roots = []
    with torch.inference_mode():
        for ply in range(6):
            carry = history.root_carry
            expected = prefill(rt, history, board, ("a", "g"))
            (result,) = play(rt, history, board, ("a", "g"), 1, roots, config)
            assert len(roots) == 1, "only the first move may prefill the root"
            if ply == 0:
                assert carry is None
                continue
            # The reused root equals the full-history prefill of this position.
            ids = torch.tensor(result.legal_ids)
            log_priors = torch.log_softmax(expected["logits"][-1][ids].float(), 0)
            torch.testing.assert_close(
                torch.tensor(result.root_log_priors, dtype=torch.float64), log_priors.double(),
                atol=1e-5, rtol=1e-5,
            )
            wdl = torch.softmax(expected["value_logits"][-1].float(), -1)
            torch.testing.assert_close(
                torch.tensor(result.root_wdl, dtype=torch.float64), wdl.double(), atol=1e-5, rtol=1e-5
            )
            assert carry.prefix_len == len(history.seq_token_id)
            for (k, v), (ek, ev) in zip(carry.prefix_kv, expected["kv_caches"]):
                torch.testing.assert_close(k, ek, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(v, ev, atol=1e-5, rtol=1e-5)


def test_carry_is_dropped_for_new_weights_and_other_games():
    rt = runtime()
    config = GumbelConfig(simulations=8, top_m=4, max_depth=3)
    board = chess.Board()
    history = _SequenceHistory(move_vocab=VOCAB, board_state_encoder=ENCODER)
    roots = []
    with torch.inference_mode():
        play(rt, history, board, ("a", "g"), 2, roots, config)
        assert len(roots) == 1
        rt.clear_caches()  # New weights: the carried decode is stale.
        play(rt, history, board, ("a", "g"), 1, roots, config)
        assert len(roots) == 2
        # Another owner (e.g. the other actor of a two-model game) prefills.
        play(rt, history, board, ("b", "g"), 1, roots, config)
        assert len(roots) == 3
        play(rt, history, board, ("b", "g"), 1, roots, config)
        assert len(roots) == 3
        # A history that no longer matches the carried child prefills.
        history.root_carry = replace(history.root_carry, fen="8/8/8/8/8/8/8/8 w - - 0 1")
        play(rt, history, board, ("b", "g"), 1, roots, config)
        assert len(roots) == 4
