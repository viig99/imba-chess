"""Evaluation-only, context-preserving negamax. No torch dependency.

Depth counts edges from the original board. Every nonterminal intermediate
node is decoded before its children, even when only its policy/KV is needed.
"""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any

import chess
import imba_chess_native as cc

from imba_chess.eval import cozy_bridge
from imba_chess.eval.search import EvalRequest, PositionEval, _drive, _root_hash_seed

POLICIES = {"value_search_alphabeta", "value_search_pvs"}
MATE = 10000.0


@dataclass(frozen=True)
class AlphaBetaConfig:
    budget: int = 2048
    max_depth: int = 9
    iterative_deepening: bool = True
    policy: str = "value_search_alphabeta"
    score_cache: str = "off"
    lmr: bool = False

    def __post_init__(self):
        if self.lmr and self.policy != "value_search_pvs":
            raise ValueError("LMR requires PVS")
        if self.score_cache not in {"off", "context"}:
            raise ValueError("score_cache must be off or context")
        if self.budget < 0:
            raise ValueError("search budget must be nonnegative")
        if not 1 <= self.max_depth <= 128:
            raise ValueError("alpha-beta/PVS depth must be in [1, 128]")
        if self.policy not in POLICIES:
            raise ValueError(f"Unknown search policy: {self.policy}")


@dataclass
class SearchReport:
    chosen_index: int | None
    score: float | None
    completed_depth: int
    attempted_depth: int
    pv: list[str]
    stop_reason: str
    stats: dict[str, int | float]
    selective: bool = False

    def debug(self):
        return {"search_report": asdict(self), "search_stats": self.stats}


@dataclass(eq=False)
class _Node:
    identity: int
    board: Any
    history: list[int]
    handle: Any
    ply: int
    terminal: float | None
    evaluation: PositionEval | None = None
    children: dict[int, _Node] = field(default_factory=dict)
    best: int | None = None


class _BudgetExhausted(Exception):
    pass


def _validate(board, evaluation):
    native = {cozy_bridge.cozy_move_to_uci(board, m) for m in board.generate_moves()}
    mapped = evaluation.legal_ucis
    if set(mapped) != native or len(mapped) != len(native):
        raise ValueError(f"Incomplete legal vocabulary coverage at {board.fen()}: "
                         f"missing={sorted(native - set(mapped))}, mapped={mapped}")
    if any(len(x) != len(mapped) for x in (
        evaluation.legal_moves, evaluation.legal_log_priors,
        evaluation.legal_forcing, evaluation.legal_ids,
    )):
        raise ValueError(f"Misaligned evaluator output at {board.fen()}")
    if [cozy_bridge.cozy_move_to_uci(board, m) for m in evaluation.legal_moves] != mapped:
        raise ValueError(f"Misaligned legal moves at {board.fen()}")
    if not math.isfinite(evaluation.value_stm) or not -1 <= evaluation.value_stm <= 1:
        raise ValueError(f"Invalid neural value at {board.fen()}: {evaluation.value_stm}")
    if not all(math.isfinite(p) for p in evaluation.legal_log_priors):
        raise ValueError(f"Non-finite policy at {board.fen()}")


@dataclass(frozen=True)
class _Entry:
    score: float
    bound: str
    best: int | None
    pv: tuple[str, ...]


class _ScoreCache:
    """Exact continuation identity, exact depth, exact search profile only."""
    def __init__(self, capacity=65536):
        self.capacity = capacity
        self.entries = OrderedDict()

    def get(self, key):
        entry = self.entries.get(key)
        if entry is not None:
            self.entries.move_to_end(key)
        return entry

    def put(self, key, entry):
        self.entries[key] = entry
        self.entries.move_to_end(key)
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)


class _Search:
    def __init__(self, extend, config):
        self.extend = extend
        self.config = config
        self.stats = dict(new_neural_evaluations=0, raw_eval_cache_hits=0,
                          recursive_visits=0, alpha_beta_cutoffs=0,
                          inference_requests=0, fallback_count=0, board_hash_repeats=0,
                          pvs_scout_calls=0, pvs_full_window_researches=0)
        self.cache = _ScoreCache()
        self.profile = (config.policy, config.lmr)
        self.stats.update(score_cache_probes=0, score_cache_hits=0, score_cache_cutoffs=0)
        self.stats.update(lmr_attempts=0, lmr_full_depth_verifications=0, selective_results=0)
        self.nodes = []
        self.hashes = set()
        self.pending_best = {}

    def node(self, board, history, handle, ply, terminal):
        if terminal:
            terminal = -(MATE - ply)
        node = _Node(len(self.nodes), board, history, handle, ply, terminal)
        self.nodes.append(node)
        board_hash = cozy_bridge.repetition_hash(board)
        if board_hash in self.hashes:
            self.stats['board_hash_repeats'] += 1
        self.hashes.add(board_hash)
        return node

    def child(self, node, index):
        if index not in node.children:
            ev = node.evaluation
            board, history, terminal = cc.push_and_classify(
                node.board, ev.legal_moves[index], node.history, True)
            handle = None if terminal is not None else self.extend(
                node.handle, ev.legal_ucis[index], ev.legal_ids[index])
            node.children[index] = self.node(board, history, handle, node.ply + 1, terminal)
        return node.children[index]

    def evaluate(self, node):
        if node.evaluation is not None:
            self.stats['raw_eval_cache_hits'] += 1
            return node.evaluation
        if self.stats['new_neural_evaluations'] >= self.config.budget:
            raise _BudgetExhausted
        self.stats['new_neural_evaluations'] += 1
        self.stats['inference_requests'] += 1
        rows = yield EvalRequest([(node.handle, node.board)])
        if len(rows) != 1:
            raise ValueError('Expected exactly one evaluation row')
        _validate(node.board, rows[0])
        node.evaluation = rows[0]
        return rows[0]

    def lmr_eligible(self, node, depth, move_number, index, pv_node):
        move = node.evaluation.legal_moves[index]
        return (self.config.lmr and not pv_node and node.ply > 0
                and not node.board.checkers() and depth >= 3 and move_number >= 3
                and move.promotion is None
                and not cozy_bridge.is_capture_cozy(node.board, move)
                and not cozy_bridge.gives_check(node.board, move))

    def visit(self, node, depth, alpha, beta, pv_node=True):
        self.stats['recursive_visits'] += 1
        key = f'visits_depth_{depth}'
        self.stats[key] = self.stats.get(key, 0) + 1
        if node.terminal is not None:
            return node.terminal, [], False
        original_alpha, original_beta = alpha, beta
        profile = self.profile + ((pv_node,) if self.config.lmr else ())
        cache_key = (node.identity, depth, profile)
        entry = None
        if self.config.score_cache == "context":
            self.stats['score_cache_probes'] += 1
            entry = self.cache.get(cache_key)
            if entry is not None:
                self.stats['score_cache_hits'] += 1
                if (entry.bound == 'exact'
                    or (entry.bound == 'lower' and entry.score >= beta)
                    or (entry.bound == 'upper' and entry.score <= alpha)):
                    self.stats['score_cache_cutoffs'] += 1
                    return entry.score, list(entry.pv), False
        ev = yield from self.evaluate(node)
        if depth == 0:
            return ev.value_stm, [], False
        order = sorted(range(len(ev.legal_ucis)), key=lambda i: (
            i != (node.best if node.best is not None else entry.best if entry else None),
            -ev.legal_log_priors[i], ev.legal_ucis[i]))
        best_score, best_pv, best_index = -math.inf, [], None
        selective = False
        for move_number, index in enumerate(order):
            child = self.child(node, index)
            if self.config.policy == "value_search_pvs" and move_number > 0:
                self.stats['pvs_scout_calls'] += 1
                reduced = self.lmr_eligible(node, depth, move_number, index, pv_node)
                if reduced:
                    self.stats['lmr_attempts'] += 1
                value, pv, child_selective = yield from self.visit(
                    child, depth - 2 if reduced else depth - 1,
                    -math.nextafter(alpha, math.inf), -alpha, False)
                score = -value
                if reduced and score > alpha:
                    self.stats['lmr_full_depth_verifications'] += 1
                    value, pv, child_selective = yield from self.visit(
                        child, depth - 1, -math.nextafter(alpha, math.inf), -alpha, False)
                    score = -value
                elif reduced:
                    child_selective = True
                if alpha < score < beta:
                    self.stats['pvs_full_window_researches'] += 1
                    value, pv, child_selective = yield from self.visit(child, depth - 1, -beta, -alpha, pv_node)
                    score = -value
            else:
                value, pv, child_selective = yield from self.visit(child, depth - 1, -beta, -alpha, pv_node)
                score = -value
            selective = selective or child_selective
            if score > best_score:
                best_score, best_pv, best_index = score, [ev.legal_ucis[index]] + pv, index
            alpha = max(alpha, score)
            if alpha >= beta:
                self.stats['alpha_beta_cutoffs'] += 1
                break
        self.pending_best[node.identity] = best_index
        if selective:
            self.stats["selective_results"] += 1
        if self.config.score_cache == "context":
            bound = ('upper' if best_score <= original_alpha else
                     'lower' if best_score >= original_beta else 'exact')
            self.cache.put(cache_key, _Entry(best_score, "selective" if selective else bound, best_index, tuple(best_pv)))
        return best_score, best_pv, selective


def search_stepwise(*, extend, root_handle, board: chess.Board,
                    legal_moves, legal_log_priors, config: AlphaBetaConfig):
    started = time.perf_counter()
    ctx = _Search(extend, config)
    native = cozy_bridge.board_to_cozy(board)
    history = _root_hash_seed(board)
    terminal = cozy_bridge.terminal_value_native(native, color_is_stm=True, hash_history=history)
    root = ctx.node(native, history, root_handle, 0, terminal)
    report = SearchReport(None, terminal, 0, 0, [], 'terminal_position', ctx.stats)
    if terminal is not None:
        report.score = root.terminal
        return report
    # Root KV/policy were already prefetched by the existing adapter.
    root.evaluation = PositionEval(0.0,
        [cozy_bridge.py_move_to_cozy(board, m) for m in legal_moves],
        [m.uci() for m in legal_moves], list(legal_log_priors),
        [False] * len(legal_moves), [None] * len(legal_moves))
    _validate(native, root.evaluation)
    order = sorted(range(len(legal_moves)), key=lambda i: (-legal_log_priors[i], legal_moves[i].uci()))
    report.chosen_index = order[0]
    report.score = None
    report.stop_reason = 'depth_ceiling'
    # All mating moves are inspected without decoding anything.
    for index in order:
        child = ctx.child(root, index)
        if child.terminal is not None and child.terminal < -1:
            report.chosen_index, report.score = index, -child.terminal
            report.completed_depth = report.attempted_depth = 1
            report.pv = [legal_moves[index].uci()]
            break
    else:
        depths = range(1, config.max_depth + 1) if config.iterative_deepening else [config.max_depth]
        for depth in depths:
            report.attempted_depth = depth
            ctx.pending_best.clear()
            try:
                score, pv, selective = yield from ctx.visit(root, depth, -math.inf, math.inf)
            except _BudgetExhausted:
                report.stop_reason = 'evaluation_budget' if report.completed_depth else 'no_completed_iteration_fallback'
                ctx.stats['fallback_count'] = int(not report.completed_depth)
                break
            report.score, report.pv, report.completed_depth = score, pv, depth
            report.selective = selective
            report.chosen_index = root.evaluation.legal_ucis.index(pv[0])
            for identity, best in ctx.pending_best.items():
                ctx.nodes[identity].best = best
    ctx.stats['max_completed_depth'] = report.completed_depth
    ctx.stats['max_attempted_depth'] = report.attempted_depth
    ctx.stats['move_selection_seconds'] = time.perf_counter() - started
    return report


def select_value_search(*, evaluator, **kwargs):
    return _drive(search_stepwise(extend=evaluator.extend, **kwargs), evaluator)
