"""KLENT self-play: every slot plays one game; finished slots restart at once.

Mirrors the reference's auto-reset rollout: `positions` steps-by-slots are
played with fixed weights, games still running at the end are dropped (their
lambda-returns would need a bootstrap past the cut), and every played position
counts toward the simulator-evaluation budget.

A game may start from a human prefix (`starts`): its plies are played as forced
moves through the same decode, so the model sees the real history, but only
the plies after the takeover are supervised (policy target, Q return).
"""

from collections import Counter
from dataclasses import dataclass, field
import time

import imba_chess_native as cc
import numpy as np
import torch

from imba_chess.eval import cozy_bridge

from .engine import TOKEN_KEYS
from .targets import improved_policy, lambda_returns

_EP_MODES = {"fen": 0, "legal": 1, "xfen": 2}


@dataclass
class _Game:
    board: "cc.Board"
    prev_move_id: int
    forced: list = field(default_factory=list)  # remaining human prefix moves (UCI)
    history: list = field(default_factory=list)
    tokens: list = field(default_factory=list)  # per ply: (piece bytes, 6 ints)
    supervised: list = field(default_factory=list)  # per ply: False for prefix plies
    legal: list = field(default_factory=list)
    policy: list = field(default_factory=list)
    move_ids: list = field(default_factory=list)
    value_q: list = field(default_factory=list)
    value_head: list = field(default_factory=list)


class BoardCodec:
    """Native board -> model token fields and legal-move projection."""

    def __init__(self, move_vocab, board_config):
        self.move_vocab = move_vocab
        self.args = (
            _EP_MODES[board_config.en_passant],
            board_config.halfmove_max,
            board_config.halfmove_bucket_size,
            board_config.fullmove_max,
            board_config.fullmove_bucket_size,
        )

    def encode(self, board):
        return cc.encode_board_state(board, *self.args)

    def legal(self, board):
        ids, moves, _, _, total = cozy_bridge.project_legal_moves(board, self.move_vocab)
        if len(ids) != total or not ids:
            raise ValueError("legal moves missing from the move vocabulary")
        return ids, moves


def token_batch(states, prev_move_ids):
    """Encoded states for every slot -> CPU tensors keyed like TOKEN_KEYS."""
    pieces = np.frombuffer(b"".join(s[0] for s in states), dtype=np.uint8)
    scalars = np.array([s[1:] for s in states], dtype=np.int64)
    batch = dict(piece_ids=torch.from_numpy(pieces.reshape(len(states), 64).copy()))
    for column, key in enumerate(TOKEN_KEYS[1:-1]):
        batch[key] = torch.from_numpy(scalars[:, column].copy())
    batch["prev_move_id"] = torch.tensor(prev_move_ids, dtype=torch.long)
    return batch


def gather_legal(tensor, legal_ids, device):
    """[N, V] device tensor -> ([N, W] gathered values, [N, W] mask)."""
    width = max(map(len, legal_ids))
    index = torch.zeros(len(legal_ids), width, dtype=torch.long)
    mask = torch.zeros(len(legal_ids), width, dtype=torch.bool)
    for row, ids in enumerate(legal_ids):
        index[row, : len(ids)] = torch.tensor(ids)
        mask[row, : len(ids)] = True
    index, mask = index.to(device, non_blocking=True), mask.to(device, non_blocking=True)
    return tensor.float().gather(1, index), mask


def _wdl_value(value_logits):
    wdl = torch.softmax(value_logits.float(), -1)
    return wdl[:, 2] - wdl[:, 0]


def _finish(game, final_reward, *, lam, bootstrap, termination):
    values = game.value_q if bootstrap == "q" else game.value_head
    # Supervised plies are the self-played tail of the game (after any prefix).
    plies = len(game.move_ids)
    # Mover at supervised ply t is the last mover iff (plies - 1 - t) is even.
    sign = np.where((plies - 1 - np.arange(plies)) % 2 == 0, 1.0, -1.0)
    offsets = np.zeros(plies + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(ids) for ids in game.legal])
    return dict(
        # Token fields cover every ply, prefix included.
        piece_ids=np.frombuffer(b"".join(t[0] for t in game.tokens), np.uint8).reshape(
            len(game.tokens), 64
        ),
        supervised=np.asarray(game.supervised, dtype=bool),
        # TOKEN_KEYS[1:-1] columns, then prev_move_id.
        scalars=np.array([t[1:6] for t in game.tokens], dtype=np.int16),
        prev_move_id=np.array([t[6] for t in game.tokens], dtype=np.int16),
        legal_offsets=offsets,
        legal_ids=np.concatenate([np.asarray(ids, np.int16) for ids in game.legal]),
        policy=np.concatenate(game.policy).astype(np.float32),
        move_id=np.asarray(game.move_ids, np.int16),
        returns=lambda_returns(final_reward, values, lam),
        outcome=(final_reward * sign).astype(np.int8),
        termination=termination,
    )


class SelfPlay:
    def __init__(self, engine, codec, *, alpha, beta, lam, max_plies, start_id, generator,
                 advantage=False, starts=None):
        # starts: callable returning a list of UCI prefix moves for a new game
        # (empty = the initial position); None plays every game from move one.
        self.starts = starts
        # advantage: the action-value head outputs A(s, a), Q = V(s) + A(s, a).
        # V is constant across a position's moves, so pi' uses A unchanged;
        # only the bootstrap value needs V added back.
        self.advantage = advantage
        self.engine, self.codec = engine, codec
        self.alpha, self.beta, self.lam = alpha, beta, lam
        self.max_plies, self.start_id, self.generator = max_plies, start_id, generator

    def _new_game(self):
        prefix = list(self.starts()) if self.starts is not None else []
        return _Game(board=cc.Board.startpos(), prev_move_id=self.start_id, forced=prefix)

    def collect(self, positions, *, bootstrap):
        engine, device = self.engine, self.engine.device
        slots = engine.slots
        games = [self._new_game() for _ in range(slots)]
        engine.reset(range(slots))
        finished, terminations = [], Counter()
        stats = torch.zeros(5, dtype=torch.float64, device=device)
        timing = Counter()
        steps = -(-positions // slots)
        for _ in range(steps):
            start = time.perf_counter()
            states = [self.codec.encode(g.board) for g in games]
            legal = [self.codec.legal(g.board) for g in games]
            batch = token_batch(states, [g.prev_move_id for g in games])
            timing["cpu_prepare"] += time.perf_counter() - start

            start = time.perf_counter()
            out = engine.step(batch)
            legal_ids = [ids for ids, _ in legal]
            logits, mask = gather_legal(out["logits"], legal_ids, device)
            q, _ = gather_legal(out["q"], legal_ids, device)
            prior = torch.softmax(logits.masked_fill(~mask, -torch.inf), -1)
            policy = improved_policy(logits, q, mask, alpha=self.alpha, beta=self.beta)
            choice = torch.multinomial(policy, 1, generator=self.generator).squeeze(1)
            log_prior = torch.log(prior.clamp_min(1e-30))
            log_policy = torch.log(policy.clamp_min(1e-30))
            value_q = (policy * q).sum(-1)
            stats += torch.stack([
                (prior * q).sum(-1).sum(),
                value_q.sum(),
                (policy * (log_policy - log_prior)).masked_fill(~mask, 0).sum(),
                -(prior * log_prior).masked_fill(~mask, 0).sum(),
                -(policy * log_policy).masked_fill(~mask, 0).sum(),
            ]).double()
            value_head = _wdl_value(out["value_logits"]) if "value_logits" in out else value_q
            if self.advantage:
                value_q = value_q + value_head
            host = torch.cat([
                choice[:, None].float(), value_q[:, None], value_head[:, None], policy
            ], 1).cpu().numpy()
            timing["gpu"] += time.perf_counter() - start

            start = time.perf_counter()
            reset = []
            for slot, game in enumerate(games):
                ids, moves = legal[slot]
                state = states[slot]
                game.tokens.append((*state, game.prev_move_id))
                if game.forced:
                    # Human prefix ply: decoded for the history, not supervised.
                    uci = game.forced.pop(0)
                    pick = ids.index(self.codec.move_vocab.encode(uci))
                    game.supervised.append(False)
                    child, history, value = cc.push_and_classify(
                        game.board, moves[pick], game.history, True
                    )
                    if value is not None:
                        raise ValueError(f"start prefix reaches a terminal position at {uci}")
                    game.board, game.history, game.prev_move_id = child, history, ids[pick]
                    continue
                pick = int(host[slot, 0])
                game.supervised.append(True)
                game.legal.append(ids)
                game.policy.append(host[slot, 3 : 3 + len(ids)].copy())
                game.move_ids.append(ids[pick])
                game.value_q.append(float(host[slot, 1]))
                game.value_head.append(float(host[slot, 2]))
                child, history, value = cc.push_and_classify(
                    game.board, moves[pick], game.history, True
                )
                # The cap counts every ply (prefix included): it bounds the context.
                if value is not None or len(game.tokens) >= self.max_plies:
                    # value is from the child's side to move; the mover gets -value.
                    reward = 0.0 if value is None else -float(value)
                    termination = "game_limit" if value is None else (
                        "mate" if value != 0 else "draw"
                    )
                    terminations[termination] += 1
                    finished.append(
                        _finish(game, reward, lam=self.lam, bootstrap=bootstrap,
                                termination=termination)
                    )
                    games[slot] = self._new_game()
                    reset.append(slot)
                else:
                    game.board, game.history, game.prev_move_id = child, history, ids[pick]
            if reset:
                engine.reset(reset)
            timing["cpu_apply"] += time.perf_counter() - start
        played = steps * slots
        names = ("return_0", "return_1", "kl_1", "ent_0", "ent_1")
        metrics = {f"selfplay/{n}": v / played for n, v in zip(names, stats.tolist())}
        metrics.update({f"selfplay/termination_{k}": v for k, v in terminations.items()})
        kept = sum(len(g["move_id"]) for g in finished)
        prefix = sum(int((~g["supervised"]).sum()) for g in finished)
        metrics.update({
            "selfplay/positions_played": played,
            "selfplay/positions_kept": kept,
            "selfplay/games": len(finished),
            "selfplay/mean_plies": kept / max(len(finished), 1),
            "selfplay/mean_prefix_plies": prefix / max(len(finished), 1),
            **{f"time/{k}": v for k, v in timing.items()},
        })
        return finished, metrics
