"""Versioned, resumable Stockfish references with complete chess histories."""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
import threading
from pathlib import Path

import chess
import chess.engine
import chess.pgn


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def human_games(path):
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            game = chess.pgn.read_game(io.StringIO(row["movetext"]))
            if game is None or game.errors or game.board().fen() != chess.STARTING_FEN:
                continue
            yield dict(source="human", source_id=row.get("Site", ""),
                       moves=[m.uci() for m in game.mainline_moves()],
                       metadata={k: row.get(k) for k in ("WhiteElo", "BlackElo", "Result", "TimeControl")})


def replay_games(path, seed):
    from imba_chess.data.self_play_store import SelfPlayStore

    store = SelfPlayStore(path, read_only=True)
    ids = sorted(store.manifest["active"])
    random.Random(seed).shuffle(ids)
    for gid in ids:
        game = store.read_game(gid)
        if game["status"] == "completed":
            yield dict(source="self_play", source_id=gid,
                       moves=list(game["prefix_moves"]) + list(game["moves"]),
                       metadata=dict(iteration=store.index[gid][1]["iteration"],
                                     previously_in_replay=True))


def pgn_games(paths):
    for path in sorted(paths):
        with Path(path).open() as stream:
            while (game := chess.pgn.read_game(stream)) is not None:
                if game.errors or game.board().fen() != chess.STARTING_FEN:
                    continue
                yield dict(source="stockfish_match", source_id=str(path),
                           moves=[m.uci() for m in game.mainline_moves()], metadata=dict(game.headers))


def sample_games(sources, *, per_game=5, seed=42, max_history=477):
    """Select well-spaced nonterminal roots, deduplicating across source games.

    sources is an iterable of (game_iterator, required_game_count). Entire games
    stay in one split. Board duplicates are removed across both splits while the
    selected root retains its original full history for model inference.
    """
    if per_game < 1 or max_history < 12:
        raise ValueError("invalid sampling sizes")
    seen_games, seen_boards, positions, selected_games = set(), set(), [], []
    counts = {}
    for games, required in sources:
        accepted = 0
        for game in games:
            game_id = digest(game["moves"])
            if game_id in seen_games:
                continue
            board, candidates, local_boards = chess.Board(), [], set()
            try:
                for ply, move in enumerate(game["moves"][:max_history], 1):
                    board.push_uci(move)
                    if ply >= 12 and not board.is_game_over(claim_draw=True):
                        key = " ".join(board.fen().split()[:4])
                        if key not in seen_boards and key not in local_boards and board.legal_moves.count() > 1:
                            candidates.append((ply, key, board.fen(), len(board.piece_map())))
                            local_boards.add(key)
            except ValueError:
                continue
            if len(candidates) < per_game:
                continue
            rng = random.Random(f"{seed}:{game_id}")
            chosen = []
            # Equal segments of each trajectory cover early and late play;
            # one position per segment avoids adjacent near-duplicates.
            for i in range(per_game):
                segment = candidates[i * len(candidates) // per_game:(i + 1) * len(candidates) // per_game]
                chosen.append(rng.choice(segment))
            split = "tune" if int(digest([seed, game_id])[:8], 16) % 5 == 0 else "test"
            for ply, key, fen, pieces in chosen:
                seen_boards.add(key)
                positions.append(dict(position_id=digest([game_id, ply])[:24], game_id=game_id,
                                      source=game["source"], source_id=game["source_id"],
                                      source_metadata=game["metadata"], split=split,
                                      prefix_moves=game["moves"][:ply], fen=fen, pieces=pieces,
                                      phase="endgame" if pieces <= 12 else "opening" if ply <= 24 else "middlegame"))
            seen_games.add(game_id)
            selected_games.append(dict(game, game_id=game_id, split=split))
            counts[game["source"]] = counts.get(game["source"], 0) + 1
            accepted += 1
            if accepted == required:
                break
        if accepted != required:
            raise ValueError(f"requested {required} games, found only {accepted}")
    random.Random(seed).shuffle(positions)
    return dict(schema_version=1, seed=seed, per_game=per_game, game_counts=counts,
                positions=positions, positions_sha256=digest(positions),
                games=selected_games, games_sha256=digest(selected_games))


def engine_identity(path):
    with chess.engine.SimpleEngine.popen_uci(str(path)) as engine:
        identity = dict(engine.id)
        defaults = {k: v.default for k, v in engine.options.items() if k.startswith("EvalFile")}
    return dict(path=str(Path(path).resolve()), binary_sha256=file_sha256(path),
                identity=identity, network_defaults=defaults)


def score_info(info, turn):
    score = info["score"].pov(turn)
    wdl = info["wdl"].pov(turn)
    if info.get("lowerbound") or info.get("upperbound"):
        raise ValueError("bounded engine score cannot be a reference label")
    return dict(cp=score.score(), mate=score.mate(),
                wdl=[wdl.losses / 1000, wdl.draws / 1000, wdl.wins / 1000],
                expectation=wdl.expectation(), pv=[m.uci() for m in info.get("pv", [])],
                nodes=info.get("nodes"), depth=info.get("depth"),
                seldepth=info.get("seldepth"), seconds=info.get("time"))


def analyse_exact(engine, board, nodes, root_moves=None):
    """Keep bound flags from the last score report, not an earlier iteration.

    python-chess analyse() merges information dictionaries across reports, so
    a lowerbound/upperbound key from an aspiration search can survive a later
    exact score. Read the stream to retain each score report's own flags.
    A node limit can interrupt an aspiration search. In that case retain the
    last completed exact iteration. Retry only if no exact iteration exists.
    """
    for attempt in range(4):
        engine.configure({"Clear Hash": None})
        last, exact = None, None
        with engine.analysis(board, chess.engine.Limit(nodes=nodes * 2 ** attempt),
                             root_moves=root_moves) as analysis:
            for info in analysis:
                if "score" in info:
                    last = info
                    if not (info.get("lowerbound") or info.get("upperbound")):
                        exact = info
        if exact is not None:
            return score_info(exact, board.turn) | dict(budget_multiplier=2 ** attempt,
                        searched_nodes=last.get("nodes"), final_iteration_bounded=exact is not last)
    raise ValueError("engine did not produce an exact score after four analysis budgets")


_local = threading.local()
_engines = []


def start_engine(path, hash_mb):
    engine = chess.engine.SimpleEngine.popen_uci(str(path))
    _engines.append(engine)
    _local.engine = engine
    engine.configure({"Threads": 1, "Hash": hash_mb, "UCI_LimitStrength": False,
                       "Skill Level": 20, "UCI_ShowWDL": True})


def close_engines():
    while _engines:
        engine = _engines.pop()
        try:
            engine.quit()
        except chess.engine.EngineTerminatedError:
            pass


def label_position(job):
    engine = _local.engine
    position, nodes, root_nodes = job
    board = chess.Board()
    for move in position["prefix_moves"]:
        board.push_uci(move)
    if board.fen() != position["fen"] or board.is_game_over(claim_draw=True):
        raise ValueError("reference root does not match its history or is terminal")
    root = analyse_exact(engine, board, root_nodes)
    scores = {}
    for move in sorted(board.legal_moves, key=lambda m: m.uci()):
        # Each root action gets the same independent search budget. Retaining
        # the board stack keeps repetition and fifty-move information intact.
        scores[move.uci()] = analyse_exact(engine, board, nodes, root_moves=[move])
    return dict(position_id=position["position_id"], root=root, moves=scores)


def read_labels(path, *, recover_partial=False):
    """Recover only a partial final append; never silently accept duplicates."""
    rows = {}
    if not Path(path).exists():
        return rows
    with Path(path).open("rb+" if recover_partial else "rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                if recover_partial:
                    stream.truncate(offset)
                break
            row = json.loads(line)
            pid = row["position_id"]
            if pid in rows:
                raise ValueError(f"duplicate reference position {pid}")
            rows[pid] = row
    return rows


def validate_labels(manifest, labels):
    expected = {p["position_id"] for p in manifest["positions"]}
    if set(labels) != expected:
        raise ValueError(f"reference incomplete: {len(labels)}/{len(expected)}")
    for position in manifest["positions"]:
        board = chess.Board(position["fen"])
        row = labels[position["position_id"]]
        if set(row["moves"]) != {m.uci() for m in board.legal_moves}:
            raise ValueError("reference legal-move coverage mismatch")
        for score in [row["root"], *row["moves"].values()]:
            if (len(score["wdl"]) != 3 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in score["wdl"])
                    or abs(sum(score["wdl"]) - 1) > 1e-8):
                raise ValueError("invalid reference WDL")
            if not math.isfinite(score["expectation"]) or not 0 <= score["expectation"] <= 1:
                raise ValueError("invalid reference expectation")
            if abs(score["expectation"] - score["wdl"][2] - 0.5 * score["wdl"][1]) > 1e-8:
                raise ValueError("inconsistent reference expectation")
