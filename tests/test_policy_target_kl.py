from __future__ import annotations

from imba_chess.data.move_vocab import MoveVocab, MoveVocabConfig
from imba_chess.data.policy_target_kl import POLICY_KL_MAX_ARMS, arm_vocab_ids_and_qhat
from imba_chess.data.rollout_store import RolloutRow


def _row(arm_move_uci: tuple[str, ...], arm_backed_value: tuple[float, ...]) -> RolloutRow:
    return RolloutRow(
        game_id="g1",
        ply=0,
        human_move_uci=arm_move_uci[0] if arm_move_uci else "e2e4",
        human_move_backed_value=0.1,
        real_outcome_stm=1,
        best_arm_move_uci=arm_move_uci[0] if arm_move_uci else "e2e4",
        best_arm_backed_value=arm_backed_value[0] if arm_backed_value else 0.0,
        root_wdl_unsearched=(0.2, 0.3, 0.5),
        arm_move_uci=arm_move_uci,
        arm_backed_value=arm_backed_value,
        arm_evals_spent=tuple(100 for _ in arm_move_uci),
        arm_log_prior=tuple(-0.5 for _ in arm_move_uci),
        search_budget=2048,
        search_top_m=16,
        search_max_depth=8,
        checkpoint="dummy.pt",
    )


def test_arm_vocab_ids_and_qhat_maps_known_moves():
    vocab = MoveVocab.build(["e2e4", "d2d4", "g1f3"], config=MoveVocabConfig(include_unk=True))
    row = _row(("e2e4", "d2d4"), (0.3, -0.1))

    arm_ids, arm_qhat, arm_mask = arm_vocab_ids_and_qhat(row, vocab, max_arms=4)

    assert arm_ids[0] == vocab.token_to_id["e2e4"]
    assert arm_ids[1] == vocab.token_to_id["d2d4"]
    assert arm_qhat[0] == 0.3
    assert arm_qhat[1] == -0.1
    assert arm_mask == [True, True, False, False]


def test_arm_vocab_ids_and_qhat_pads_short_rows():
    vocab = MoveVocab.build(["e2e4"], config=MoveVocabConfig(include_unk=True))
    row = _row(("e2e4",), (0.5,))

    arm_ids, arm_qhat, arm_mask = arm_vocab_ids_and_qhat(row, vocab, max_arms=3)

    assert len(arm_ids) == 3
    assert len(arm_qhat) == 3
    assert len(arm_mask) == 3
    assert arm_mask == [True, False, False]
    # Padding slots get a real (non-negative) dummy id, not None, so a
    # torch.gather using this id never indexes out of bounds; the mask is
    # what actually excludes them from the softmax.
    assert arm_ids[1] == 0
    assert arm_qhat[1] == 0.0


def test_arm_vocab_ids_and_qhat_truncates_long_rows_keeping_original_order():
    vocab = MoveVocab.build(
        ["e2e4", "d2d4", "g1f3", "b1c3", "f2f4"], config=MoveVocabConfig(include_unk=True)
    )
    row = _row(
        ("e2e4", "d2d4", "g1f3", "b1c3", "f2f4"),
        (0.1, 0.2, 0.3, 0.4, 0.5),
    )

    arm_ids, arm_qhat, arm_mask = arm_vocab_ids_and_qhat(row, vocab, max_arms=3)

    assert len(arm_ids) == 3
    assert arm_mask == [True, True, True]
    assert arm_ids == [
        vocab.token_to_id["e2e4"],
        vocab.token_to_id["d2d4"],
        vocab.token_to_id["g1f3"],
    ]


def test_arm_vocab_ids_and_qhat_excludes_unk_arms_with_include_unk_true():
    # A move string not in the vocab's build set falls back to <unk> when
    # include_unk=True -- must be excluded, not included with the <unk> id.
    vocab = MoveVocab.build(["e2e4", "d2d4"], config=MoveVocabConfig(include_unk=True))
    row = _row(("e2e4", "z9z9"), (0.3, -0.9))

    arm_ids, arm_qhat, arm_mask = arm_vocab_ids_and_qhat(row, vocab, max_arms=4)

    assert arm_mask == [True, False, False, False]
    assert arm_ids[0] == vocab.token_to_id["e2e4"]


def test_arm_vocab_ids_and_qhat_excludes_unmappable_arms_with_include_unk_false():
    # The production static vocab is built with include_unk=False (no <unk>
    # token at all -- MoveVocab.encode() would raise KeyError for an
    # unmappable move rather than silently return an unk id). This module
    # must not crash in that configuration either; it checks membership
    # directly rather than calling .encode().
    vocab = MoveVocab.build(["e2e4", "d2d4"], config=MoveVocabConfig(include_unk=False))
    row = _row(("e2e4", "z9z9"), (0.3, -0.9))

    arm_ids, arm_qhat, arm_mask = arm_vocab_ids_and_qhat(row, vocab, max_arms=4)

    assert arm_mask == [True, False, False, False]


def test_arm_vocab_ids_and_qhat_handles_empty_arms():
    vocab = MoveVocab.build(["e2e4"], config=MoveVocabConfig(include_unk=True))
    row = _row((), ())

    arm_ids, arm_qhat, arm_mask = arm_vocab_ids_and_qhat(row, vocab, max_arms=3)

    assert arm_mask == [False, False, False]


def test_policy_kl_max_arms_default_is_24():
    assert POLICY_KL_MAX_ARMS == 24
