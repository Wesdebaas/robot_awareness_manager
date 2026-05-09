"""
replay_reader.py - loads a trace directory produced by TraceLogger and exposes
the same snapshot / history interface as SimulationRunner for use by the
dashboard in replay mode.

The dashboard's build_app() accepts either a SimulationRunner (live mode) or a
ReplayReader (replay mode) and builds the appropriate layout and callbacks.
"""

import json
from pathlib import Path

from awareness_manager.visualization.snapshot import (
    _channel_color,
    compute_positions_from_structure,
)

# Matches runner.HistoryEntry indices 0-4 (the sparkline-relevant subset)
HistoryEntry = tuple[float, float, float, float, float]


class ReplayReader:
    """
    Loads a trace directory and provides tick-indexed access to snapshots and
    history windows, mirroring the SimulationRunner interface.

    snapshot_at(tick_index) returns a dict in exactly the same shape as
    SimulationRunner.snapshot(), so the dashboard graph-update callback works
    unchanged in replay mode.

    history_at(concept_id, tick_index) returns a rolling window centred on
    tick_index - both past and future within the window - giving the user a
    richer view than the live one-sided rolling buffer.
    """

    def __init__(self, trace_dir: str | Path) -> None:
        self._dir = Path(trace_dir)
        self._meta: dict = json.loads(
            (self._dir / "meta.json").read_text(encoding="utf-8")
        )
        self._ticks: list[dict] = []
        with open(self._dir / "ticks.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._ticks.append(json.loads(line))

        # Positions: use stored positions (written by TraceLogger) or recompute
        if "positions" in self._meta:
            self._positions: dict = self._meta["positions"]
        else:
            self._positions = compute_positions_from_structure(
                self._meta["structure"]
            )

        # Pre-collect all events across the trace for the events strip
        self._all_events: list[dict] = []
        for tick in self._ticks:
            for ev in tick.get("events", []):
                self._all_events.append({**ev, "t": tick["t"]})

        # Derived constants - handle schema v1 (am_params) and v2 (budget at top level)
        if self._meta.get("schema_version", 1) >= 2:
            self._budget: int = self._meta["budget"]
            self._strategy_name: str = self._meta.get("strategy", "awareness_manager")
        else:
            self._budget = self._meta["am_params"]["budget"]
            self._strategy_name = "awareness_manager"

    # ------------------------------------------------------------------
    # Replay interface
    # ------------------------------------------------------------------

    @property
    def tick_count(self) -> int:
        return len(self._ticks)

    @property
    def meta(self) -> dict:
        return self._meta

    @property
    def all_events(self) -> list[dict]:
        """All events in the trace: [{type, t, ...}, ...]"""
        return self._all_events

    def elapsed_at(self, tick_index: int) -> float:
        """Simulated time at tick_index, clamped to trace bounds."""
        if not self._ticks:
            return 0.0
        tick_index = max(0, min(tick_index, len(self._ticks) - 1))
        return self._ticks[tick_index]["t"]

    def total_duration(self) -> float:
        """Total simulated duration covered by the trace."""
        return self._ticks[-1]["t"] if self._ticks else 0.0

    def goal_transition_count(self) -> int:
        return sum(1 for ev in self._all_events if ev["type"] == "goal_transition")

    def surprise_event_count(self) -> int:
        return sum(1 for ev in self._all_events if ev["type"] == "surprise")

    def snapshot_at(self, tick_index: int, threshold: float = 0.0) -> dict:
        """
        Build a Cytoscape-ready snapshot from the trace record at tick_index.

        Returns a dict in the same shape as SimulationRunner.snapshot():
        {"elements": [...], "goal_id": ..., "queue_text": ..., "n_scheduled": ...,
         "budget": ..., "elapsed": ..., "schedule": [...]}
        """
        tick_index = max(0, min(tick_index, len(self._ticks) - 1))
        record = self._ticks[tick_index]
        return self._build_snapshot(record, threshold)

    def history_at(
        self,
        concept_id: str,
        tick_index: int,
        half_window: int = 150,
    ) -> list[HistoryEntry]:
        """
        Attention history centred on tick_index (±half_window ticks).

        Returns list of (elapsed, total_attn, mission, relational, surprise)
        matching runner.HistoryEntry indices 0-4, compatible with _make_sparkline.
        In replay mode this shows both past and future within the window, so the
        sparkline gives full context around the current position.
        """
        lo = max(0, tick_index - half_window)
        hi = min(len(self._ticks), tick_index + half_window + 1)
        result: list[HistoryEntry] = []
        for rec in self._ticks[lo:hi]:
            c = rec["concepts"].get(concept_id)
            if c is None:
                continue
            mission = c["ch_m"] + c["ch_an"]
            result.append((rec["t"], c["a"], mission, c["ch_r"], c["ch_s"]))
        return result

    # ------------------------------------------------------------------
    # Element builder (mirrors get_snapshot() from snapshot.py)
    # ------------------------------------------------------------------

    def _build_snapshot(self, record: dict, threshold: float) -> dict:
        structure = self._meta["structure"]
        positions = self._positions
        concepts_data = record["concepts"]
        schedule_set = set(record.get("schedule") or [])
        goal = record.get("goal")
        queue = record.get("queue") or []

        populated_classes = {inst["class_id"] for inst in structure["instances"]}

        elements: list[dict] = []

        # ── Class nodes ──────────────────────────────────────────────────
        for cls in structure["classes"]:
            cid = cls["id"]
            c = concepts_data.get(cid, {})
            a   = c.get("a",     0.0)
            e   = c.get("e",     0.0)
            pe  = c.get("pe",    0.0)
            cm  = c.get("ch_m",  0.0)
            ca  = c.get("ch_an", 0.0)
            cs  = c.get("ch_s",  0.0)
            pos = positions.get(cid, {"x": 0.0, "y": 0.0})

            elements.append({
                "data": {
                    "id": cid,
                    "label": f"{cid}\nA={a:.2f}  E={e:.2f}",
                    "type": "class",
                    "concept_type": cls["concept_type"],
                    "attention": round(a, 4),
                    "epistemic_error": round(e, 4),
                    "prediction_error": round(pe, 4),
                    "decay_rate": cls["decay_rate"],
                    "in_schedule": 1 if cid in schedule_set else 0,
                    "below_threshold": 1 if a < threshold else 0,
                    "is_active_goal": 1 if cid == goal else 0,
                    "has_instances": 1 if cid in populated_classes else 0,
                    "ch_mission": round(cm, 4),
                    "ch_anticipatory": round(ca, 4),
                    "ch_relational": 0.0,
                    "ch_surprise": round(cs, 4),
                    "color": _channel_color(cm, ca, 0.0, cs, e),
                },
                "position": pos,
            })

        # ── Instance nodes ────────────────────────────────────────────────
        for inst in structure["instances"]:
            iid = inst["id"]
            c = concepts_data.get(iid, {})
            a   = c.get("a",     0.0)
            e   = c.get("e",     0.0)
            pe  = c.get("pe",    0.0)
            cm  = c.get("ch_m",  0.0)
            ca  = c.get("ch_an", 0.0)
            cr  = c.get("ch_r",  0.0)
            cs  = c.get("ch_s",  0.0)
            pos = positions.get(iid, {"x": 0.0, "y": 0.0})

            elements.append({
                "data": {
                    "id": iid,
                    "label": f"{iid}\nA={a:.2f}\nE={e:.2f}",
                    "parent": inst["class_id"],
                    "type": "instance",
                    "class_id": inst["class_id"],
                    "attention": round(a, 4),
                    "epistemic_error": round(e, 4),
                    "prediction_error": round(pe, 4),
                    "decay_rate": inst["decay_rate"],
                    "in_schedule": 1 if iid in schedule_set else 0,
                    "below_threshold": 1 if a < threshold else 0,
                    "is_active_goal": 0,
                    "has_instances": 0,
                    "ch_mission": round(cm, 4),
                    "ch_anticipatory": round(ca, 4),
                    "ch_relational": round(cr, 4),
                    "ch_surprise": round(cs, 4),
                    "color": _channel_color(cm, ca, cr, cs, e),
                },
                "position": pos,
            })

        # ── Semantic edges ────────────────────────────────────────────────
        for id_a, id_b, weight in structure["semantic_edges"]:
            elements.append({
                "data": {
                    "id": f"sem__{id_a}__{id_b}",
                    "source": id_a,
                    "target": id_b,
                    "edge_type": "semantic",
                    "weight": weight,
                    "label": f"{weight:.1f}",
                }
            })

        # ── Relational edges ──────────────────────────────────────────────
        for edge in structure["relational_edges"]:
            id_a, id_b, weight = edge[0], edge[1], edge[2]
            rtype = (edge[3] if len(edge) > 3 else None) or "related"
            elements.append({
                "data": {
                    "id": f"rel__{id_a}__{id_b}",
                    "source": id_a,
                    "target": id_b,
                    "edge_type": "relational",
                    "weight": weight,
                    "relation_type": rtype,
                    "label": rtype,
                }
            })

        # ── Top-bar scalars ───────────────────────────────────────────────
        if queue:
            next_goal, next_eta, next_level = queue[0][0], queue[0][1], queue[0][2]
            queue_text = f"→ {next_goal} in {next_eta:.1f}s ({next_level})"
        else:
            queue_text = "no queued goals"

        return {
            "elements": elements,
            "goal_id": goal if goal else "-",
            "queue_text": queue_text,
            "n_scheduled": len(schedule_set),
            "budget": self._budget,
            "budget_unlimited": self._budget < 0,
            "strategy": self._strategy_name,
            "elapsed": record["t"],
            "schedule": list(schedule_set),
        }
