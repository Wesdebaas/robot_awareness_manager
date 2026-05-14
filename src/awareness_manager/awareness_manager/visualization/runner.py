"""
runner.py - background thread that drives AwarenessManager.tick() at real-time speed.

All AM state reads (snapshot, history) and writes (tick, observe) share a
threading.Lock so the Dash callback thread and the simulation thread never race.

Phase 2 addition: rolling attention history buffer.
    After each tick the runner appends an extended HistoryEntry to a per-concept
    deque of fixed length. The hover callback reads this via runner.history(concept_id)
    to build sparklines.

    Memory bound: history_maxlen=300 (30 s at dt=0.1 s).
    18 nodes x 300 entries x 8 floats x 8 B ≈ 346 KB. Trivially safe.

Phase 3 addition: TraceLogger integration + flush_trace().
    An optional TraceLogger is called after each tick to stream a complete
    activation trace to disk. flush_trace() dumps the rolling history buffer
    to a trace directory for the "Save last Ns" live-export button.
"""

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from awareness_manager.visualization.snapshot import compute_positions, get_snapshot

if TYPE_CHECKING:
    from awareness_manager.baselines.strategy import AttentionStrategy
    from awareness_manager.visualization.trace_logger import TraceLogger

# Type alias: each history entry is
#   (elapsed, total_attn, mission, relational, surprise, epistemic_error, prediction_error, ch_anticipatory)
# Indices 0-4 are read by the sparkline; indices 5-7 are used by flush_trace.
HistoryEntry = tuple[float, float, float, float, float, float, float, float]


class SimulationRunner:
    """
    Real-time simulation driver.

    Calls am.tick(dt) every dt real seconds so that simulated time advances
    at 1:1 real-time speed. After each tick it observes the top-priority
    concept (if observe_top=True), mirroring the existing visualiser behaviour.

    snapshot() returns a serialisable dict safe to pass directly to the
    Dash callback - it copies all necessary state under the lock so the
    callback thread holds no reference into live AM state.

    history(concept_id) returns a snapshot of the rolling attention time
    series for one concept, safe to call from the Dash callback thread.

    flush_trace(output_dir, scenario) dumps the rolling history buffer to a
    trace directory for offline replay - the "Save last Ns" live-export feature.
    """

    def __init__(
        self,
        am: 'AttentionStrategy',
        dt: float = 0.1,
        observe_top: bool = True,
        history_maxlen: int = 300,
        logger: 'TraceLogger | None' = None,
        tick_hook: 'Callable[[AttentionStrategy, float], None] | None' = None,
        observe_interval: float | None = None,
    ) -> None:
        self._am = am
        self._dt = dt
        self._observe_top = observe_top
        self._history_maxlen = history_maxlen
        self._logger = logger
        self._tick_hook = tick_hook
        self._observe_interval = observe_interval
        self._time_since_observe: float = 0.0
        self._lock = threading.Lock()
        self._schedule: list[str] = []
        self._last_observed: list[str] = []
        self._elapsed: float = 0.0
        self._running = False
        self._thread: threading.Thread | None = None

        # Rolling history: concept_id → deque of HistoryEntry tuples.
        self._history: dict[str, deque[HistoryEntry]] = {}

        # Positions are fixed for the lifetime of the runner - computed once.
        self._positions = compute_positions(am)

    def start(self) -> None:
        """Start the background tick thread (non-blocking, daemon)."""
        if self._logger is not None:
            self._logger.open(positions=self._positions)
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the tick thread to stop after the current tick."""
        self._running = False
        if self._logger is not None:
            self._logger.close()

    @property
    def available_seconds(self) -> float:
        """
        How many seconds of history are available in the rolling buffer.
        Grows from 0 to history_maxlen * dt, then stays constant.
        """
        with self._lock:
            return min(self._elapsed, self._history_maxlen * self._dt)

    def snapshot(self, threshold: float = 0.0) -> dict:
        """
        Return a serialisable snapshot of the current AM state.

        Safe to call from any thread. Holds the lock only long enough to
        copy the snapshot dict; does not hold it while the callback renders.
        """
        with self._lock:
            return get_snapshot(
                self._am,
                self._last_observed,
                self._elapsed,
                self._positions,
                threshold=threshold,
            )

    def history(self, concept_id: str) -> list[HistoryEntry]:
        """
        Return a snapshot of the attention history for one concept.

        Each entry is (elapsed_s, total_attn, mission_attn, relational_attn,
        surprise_attn, epistemic_error, prediction_error, ch_anticipatory),
        where mission_attn = F1 + F2 combined.

        Returns an empty list if the concept has no history yet.
        Safe to call from any thread.
        """
        with self._lock:
            return list(self._history.get(concept_id, []))

    def flush_trace(self, output_dir: str | Path, scenario: str) -> Path:
        """
        Dump the rolling history buffer to a trace directory.

        Writes meta.json (with truncated=True if elapsed > history window) and
        ticks.jsonl reconstructed from the per-concept history deques.

        Note: the flushed trace omits goal, queue, schedule, and events per tick
        because these are not stored in the rolling history buffer. For a complete
        trace, use TraceLogger from session start via --log.

        Returns the output_dir path (for display in the UI).
        """
        from awareness_manager.visualization.trace_logger import TraceLogger as _TL

        output_dir = Path(output_dir)

        with self._lock:
            history_snap = {cid: list(dq) for cid, dq in self._history.items()}
            elapsed_snap = self._elapsed
            positions_snap = dict(self._positions)
            am = self._am
            goal_id_snap = self._am.goal_id

        truncated = elapsed_snap > self._history_maxlen * self._dt

        # Build meta via a temporary logger instance (reuses _build_meta logic)
        tmp_logger = _TL(
            am, scenario=scenario, output_dir=output_dir,  # type: ignore[arg-type]
            dt=self._dt, sample_rate=1,
            history_maxlen=self._history_maxlen,
        )
        meta = tmp_logger._build_meta(positions=positions_snap, truncated=truncated)

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        # Group history entries by elapsed time → reconstruct tick records
        time_concepts: dict[float, dict[str, dict]] = {}
        for cid, entries in history_snap.items():
            for entry in entries:
                t = round(entry[0], 3)
                if t not in time_concepts:
                    time_concepts[t] = {}
                a, mission, rel, surp = entry[1], entry[2], entry[3], entry[4]
                e, pe, ch_an = entry[5], entry[6], entry[7]
                ch_m = max(0.0, mission - ch_an)
                time_concepts[t][cid] = {
                    "a":     round(a, 4),
                    "e":     round(e, 4),
                    "pe":    round(pe, 4),
                    "ch_m":  round(ch_m, 4),
                    "ch_an": round(ch_an, 4),
                    "ch_r":  round(rel, 4),
                    "ch_s":  round(surp, 4),
                }

        with open(output_dir / "ticks.jsonl", "w", encoding="utf-8") as f:
            for t in sorted(time_concepts.keys()):
                record = {
                    "t": t,
                    "goal": goal_id_snap,
                    "queue": [],
                    "schedule": [],
                    "events": [],
                    "concepts": time_concepts[t],
                }
                f.write(json.dumps(record, separators=(",", ":")) + "\n")

        return output_dir

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while self._running:
            t0 = time.perf_counter()
            with self._lock:
                self._schedule = self._am.tick(dt=self._dt)
                self._time_since_observe += self._dt
                if self._observe_top and self._schedule:
                    if (self._observe_interval is None
                            or self._time_since_observe >= self._observe_interval):
                        for cid in self._schedule:
                            self._am.observe(cid)
                        self._last_observed = list(self._schedule)
                        self._time_since_observe = 0.0
                self._elapsed += self._dt
                if self._tick_hook is not None:
                    self._tick_hook(self._am, self._elapsed)
                self._record_history()
                if self._logger is not None:
                    self._logger.record(self._elapsed, self._schedule)
            elapsed = time.perf_counter() - t0
            sleep_time = self._dt - elapsed
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    def _record_history(self) -> None:
        """
        Append current attention values to each concept's history deque.

        Called under the lock after every tick. Stores 8 values per concept:
        (elapsed, total_attn, mission, relational, surprise,
         epistemic_error, prediction_error, ch_anticipatory).

        Indices 0-4 are used by the live sparkline; indices 5-7 are used by
        flush_trace() to reconstruct full trace records from the buffer.
        """
        attn = self._am.attention()
        ch = self._am.attention_channels()
        kb = self._am.kb
        ikb = self._am.instance_kb

        for cid, a in attn.items():
            if cid not in self._history:
                self._history[cid] = deque(maxlen=self._history_maxlen)

            mission = ch['mission'].get(cid, 0.0) + ch['anticipatory'].get(cid, 0.0)
            relational = ch['relational'].get(cid, 0.0)
            surprise = ch['surprise'].get(cid, 0.0)
            ch_an = ch['anticipatory'].get(cid, 0.0)

            if cid in kb.concept_ids():
                concept = kb.get_concept(cid)
                e = concept.epistemic_error
                pe = concept.prediction_error
            elif ikb is not None and cid in ikb.instance_ids():
                inst = ikb.get_instance(cid)
                e = inst.epistemic_error
                pe = inst.prediction_error
            else:
                e = 0.0
                pe = 0.0

            self._history[cid].append(
                (self._elapsed, a, mission, relational, surprise, e, pe, ch_an)
            )
