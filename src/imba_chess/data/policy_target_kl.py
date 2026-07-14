from __future__ import annotations

from .move_vocab import MoveVocab
from .rollout_store import RolloutRow

# Safely above search_top_m=16 plus typical forcing-floor extras (see
# select_value_search_halving's root-level forcing floor in
# src/imba_chess/eval/search.py).
POLICY_KL_MAX_ARMS = 24


def arm_vocab_ids_and_qhat(
    row: RolloutRow,
    move_vocab: MoveVocab,
    max_arms: int = POLICY_KL_MAX_ARMS,
) -> tuple[list[int], list[float], list[bool]]:
    """Maps a rollout row's searched arms to move_vocab ids, pads/truncates to max_arms.

    Returns (arm_ids, arm_qhat, arm_mask), each length max_arms. Any arm
    whose move fails to map to a real vocab entry is excluded (not included
    with a bogus id) -- an <unk> or dummy slot in the target's softmax would
    corrupt it. This checks membership directly (move_uci in
    move_vocab.token_to_id) rather than calling move_vocab.encode(): the
    production static vocab is built with include_unk=False, where
    .encode() raises KeyError on an unmappable move rather than returning an
    <unk> id, so a membership check handles both vocab configurations
    (include_unk True or False) uniformly without relying on exception
    handling for routine control flow. Rows with more than max_arms real
    arms keep the first max_arms in the row's existing order (search's own
    return order -- Gumbel-sampled top_m first, forcing-floor extras
    appended after). Padding slots get a dummy id (0, always a valid vocab
    index) and arm_qhat=0.0, excluded from any loss via arm_mask=False.
    """
    arm_ids: list[int] = []
    arm_qhat: list[float] = []
    for move_uci, backed_value in zip(row.arm_move_uci, row.arm_backed_value):
        vocab_id = move_vocab.token_to_id.get(move_uci)
        if vocab_id is None or vocab_id == move_vocab.unk_id:
            continue
        arm_ids.append(vocab_id)
        arm_qhat.append(float(backed_value))
        if len(arm_ids) == max_arms:
            break

    num_real = len(arm_ids)
    pad_count = max_arms - num_real
    arm_ids.extend([0] * pad_count)
    arm_qhat.extend([0.0] * pad_count)
    arm_mask = [True] * num_real + [False] * pad_count
    return arm_ids, arm_qhat, arm_mask
