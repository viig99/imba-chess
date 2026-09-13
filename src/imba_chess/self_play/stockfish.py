"""Stockfish opponent adapter for the same terminal/context-safe evaluator."""

import chess.engine

from imba_chess.eval.batch_scheduler import WorkRequest
from imba_chess.eval.gumbel_search import GumbelResult


class StockfishRuntime:
    def __init__(
        self,
        reference_runtime,
        *,
        path="/usr/bin/stockfish",
        nodes=40000,
        elo=2400,
        limit_strength=False,
        threads=1,
        hash_mb=64,
    ):
        self.move_vocab = reference_runtime.move_vocab
        self.encoder = reference_runtime.encoder
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure(
            {
                "Threads": threads,
                "Hash": hash_mb,
                "UCI_LimitStrength": limit_strength,
                "UCI_Elo": elo,
            }
        )
        self.nodes = nodes
        self.executors = {"stockfish": self._execute}
        self.protocol = dict(
            engine=self.engine.id,
            path=path,
            nodes=nodes,
            elo=elo,
            limit_strength=limit_strength,
            threads=threads,
            hash_mb=hash_mb,
        )

    def _execute(self, payloads):
        results = []
        for owner, board in payloads:
            # Reset engine state for every position so restarted evaluation does
            # not depend on a previous game's transposition table.
            move = self.engine.play(
                board, chess.engine.Limit(nodes=self.nodes), game=object()
            ).move
            results.append((owner, move))
        return results

    def search(self, *, board, actor_id, game_id, **kwargs):
        owner = (actor_id, game_id)
        identity, move = yield WorkRequest("stockfish", (owner, board.copy(stack=True)))
        if identity != owner or move is None:
            raise RuntimeError("invalid Stockfish response")
        ids = [self.move_vocab.encode(m.uci()) for m in board.legal_moves]
        # Evaluation never writes these opponent placeholders to training replay.
        return GumbelResult(
            move.uci(),
            self.move_vocab.encode(move.uci()),
            ids,
            [1 / len(ids)] * len(ids),
            0.0,
            None,
            [0] * len(ids),
            [0.0] * len(ids),
            0,
            0,
            0,
            0,
            0,
        )

    def close(self):
        self.engine.quit()
