"""Deterministic G-game lockstep scheduler.

Runs G game coroutines concurrently: each tick, every live slot advances to
its next WorkRequest; requests are grouped by kind and executed as one merged
call per kind; results are scattered back. Completed games are reported in
dataset-stream order via a hold-back buffer so downstream flush/resume
semantics (--flush-every-games, progress sidecar, --skip-games) are
unchanged. Torch-free by design: payloads/results are opaque.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Generator, Iterator, NamedTuple


class WorkRequest(NamedTuple):
    kind: str  # e.g. "root_eval" | "decode_wave" — opaque to the scheduler
    payload: Any  # opaque; executor understands it


@dataclass
class _Slot:
    stream_idx: int
    game_id: str
    gen: Generator
    pending: WorkRequest | None = None


class BatchScheduler:
    """Runs up to `concurrent_games` game coroutines in lockstep ticks.

    Each tick: every slot lacking a pending request is advanced to its next
    WorkRequest (or finishes/errors); pending requests are grouped by kind
    and dispatched as one merged executor call per kind; results are
    scattered back to the originating slots. Finished games are reported via
    `on_game_done` in dataset-stream order regardless of completion order.
    """

    def __init__(
        self,
        *,
        game_factory: Iterator[tuple[str, Generator]],
        executors: dict[str, Callable[[list[Any]], list[Any]]],
        concurrent_games: int,
        on_game_done: Callable[[str, list[Any] | None], None],
        on_game_error: Callable[[str, BaseException], None],
    ) -> None:
        self._game_factory = game_factory
        self._executors = executors
        self._concurrent_games = concurrent_games
        self._on_game_done = on_game_done
        self._on_game_error = on_game_error

        self._next_stream_idx = 0
        self._next_slot_id = 0
        # Finished-but-not-yet-emitted games, keyed by stream index.
        self._held_back: dict[int, tuple[str, list[Any] | None]] = {}
        self._next_expected_idx = 0

    def run(self) -> None:
        slots: dict[int, _Slot] = {}
        self._fill_slots(slots)
        while slots:
            # Phase 1: advance every slot lacking a pending request.
            for slot_id in list(slots):
                slot = slots.get(slot_id)
                if slot is not None and slot.pending is None:
                    self._advance(slot_id, slots, send_value=None, first=True)

            # Phase 2: group pending requests by kind (stable slot order).
            by_kind: dict[str, list[tuple[int, WorkRequest]]] = defaultdict(list)
            for slot_id in sorted(slots):
                slot = slots[slot_id]
                if slot.pending is not None:
                    by_kind[slot.pending.kind].append((slot_id, slot.pending))

            if not by_kind:
                break

            # Phase 3: one merged executor call per kind; scatter results.
            for kind, entries in by_kind.items():
                results = self._executors[kind]([req.payload for _, req in entries])
                assert len(results) == len(entries)
                for (slot_id, _), result in zip(entries, results):
                    self._advance(slot_id, slots, send_value=result)

            self._fill_slots(slots)

    def _fill_slots(self, slots: dict[int, _Slot]) -> None:
        while len(slots) < self._concurrent_games:
            try:
                game_id, gen = next(self._game_factory)
            except StopIteration:
                return
            slot_id = self._next_slot_id
            self._next_slot_id += 1
            slots[slot_id] = _Slot(
                stream_idx=self._next_stream_idx,
                game_id=game_id,
                gen=gen,
                pending=None,
            )
            self._next_stream_idx += 1

    def _advance(
        self,
        slot_id: int,
        slots: dict[int, _Slot],
        send_value: Any,
        first: bool = False,
    ) -> None:
        slot = slots[slot_id]
        try:
            if first:
                request = next(slot.gen)
            else:
                request = slot.gen.send(send_value)
            slot.pending = request
        except StopIteration as stop:
            self._finish_slot(slot_id, slots, stop.value)
        except Exception as exc:  # noqa: BLE001 - isolate per-game failures
            self._on_game_error(slot.game_id, exc)
            self._finish_slot(slot_id, slots, None)

    def _finish_slot(
        self, slot_id: int, slots: dict[int, _Slot], rows: list[Any] | None
    ) -> None:
        slot = slots.pop(slot_id)
        self._held_back[slot.stream_idx] = (slot.game_id, rows)
        self._emit_ready()

    def _emit_ready(self) -> None:
        while self._next_expected_idx in self._held_back:
            game_id, rows = self._held_back.pop(self._next_expected_idx)
            self._on_game_done(game_id, rows)
            self._next_expected_idx += 1
