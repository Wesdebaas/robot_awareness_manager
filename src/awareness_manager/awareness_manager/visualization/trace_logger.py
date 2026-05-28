"""
trace_logger.py - per-tick trace writer for offline replay and analysis.

Produces a trace directory with two files:
  meta.json   - static structure + run hyperparameters (written once at open())
  ticks.jsonl - one compact JSON line per sampled tick (streamed by record())

Attach to SimulationRunner by passing a TraceLogger to its constructor.
The runner calls record() from within _run() under the AM lock, after each
strategy.tick() call.

Private-state coupling note (intentional per design decision):
    Surprise events are read via getattr(strategy, '_channel_surprise', {}) -
    the private dict that AwarenessManager's _recompute_attention() populates
    from _violation_boosts before clearing them. Baselines without violation
    boosts simply return an empty dict. No new public API is added on AM.

Schema versions:
    v1: meta contains "am_params" dict (AwarenessManager only, legacy)
    v2: meta contains "strategy", "strategy_params", "budget" at top level
"""

import datetime
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from awareness_manager.baselines.strategy import AttentionStrategy


def _git_commit() -> str:
    """Return the short HEAD commit hash, or 'unknown' if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[4]),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


class TraceLogger:
    """
    Streams a complete activation trace to disk during a live run.

    Accepts any AttentionStrategy (AwarenessManager, AlwaysOnBaseline,
    ReactiveBaseline) - uses only the public strategy interface.

    Lifecycle:
        logger = TraceLogger(strategy, scenario="pv_inspection",
                             output_dir=Path("traces/run_001"), dt=0.1)
        logger.open(positions=runner._positions)   # writes meta.json
        # inside runner._run():
        logger.record(elapsed, schedule)           # appends to ticks.jsonl
        logger.close()                             # flushes and closes the file
    """

    def __init__(
        self,
        strategy: 'AttentionStrategy',
        scenario: str,
        output_dir: Path,
        dt: float,
        sample_rate: int = 1,
        history_maxlen: int = 300,
    ) -> None:
        self._strategy = strategy
        self._scenario = scenario
        self._output_dir = Path(output_dir)
        self._dt = dt
        self._sample_rate = sample_rate
        self._history_maxlen = history_maxlen
        self._tick_count = 0
        self._prev_goal: str = strategy.goal_id
        self._file = None
        self._is_open = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, positions: dict | None = None, truncated: bool = False) -> None:
        """Write meta.json and open ticks.jsonl for streaming writes."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        meta = self._build_meta(positions=positions, truncated=truncated)
        (self._output_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        self._file = open(self._output_dir / "ticks.jsonl", "w", encoding="utf-8")
        self._is_open = True

    def close(self) -> None:
        """Flush and close the tick file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
        self._is_open = False

    # ------------------------------------------------------------------
    # Per-tick recording (called from runner._run() under the AM lock)
    # ------------------------------------------------------------------

    def record(
        self,
        elapsed: float,
        schedule: list[str],
        extra_events: list[dict] | None = None,
    ) -> None:
        """
        Append one tick record to ticks.jsonl.

        Called after strategy.tick() has returned. Reads _channel_surprise via
        getattr for surprise events - see module docstring for rationale.

        extra_events: additional event dicts appended verbatim (e.g. volatility
        injections from the batch runner).

        Skipped silently if sample_rate > 1 and this tick is not sampled.
        """
        if not self._is_open or self._file is None:
            return

        self._tick_count += 1
        if self._tick_count % self._sample_rate != 0:
            return

        strategy = self._strategy
        cur_goal = strategy.goal_id
        events: list[dict] = []

        # Goal-transition detection
        if cur_goal != self._prev_goal:
            events.append({
                "type": "goal_transition",
                "from": self._prev_goal,
                "to": cur_goal,
            })
            self._prev_goal = cur_goal

        # Surprise events: _channel_surprise is AwarenessManager-specific;
        # baselines return {} via getattr default.
        channel_surprise: dict = getattr(strategy, '_channel_surprise', {})
        for cid, boost in channel_surprise.items():
            if boost > 0.0:
                events.append({
                    "type": "surprise",
                    "concept": cid,
                    "boost": round(boost, 4),
                })

        attn = strategy.attention()
        ch = strategy.attention_channels()
        kb = strategy.kb
        ikb = strategy.instance_kb

        concepts: dict[str, dict] = {}
        for cid in kb.concept_ids():
            concept = kb.get_concept(cid)
            concepts[cid] = {
                "a":     round(attn.get(cid, 0.0), 4),
                "e":     round(concept.epistemic_error, 4),
                "pe":    round(concept.prediction_error, 4),
                "ch_m":  round(ch["mission"].get(cid, 0.0), 4),
                "ch_an": round(ch["anticipatory"].get(cid, 0.0), 4),
                "ch_r":  0.0,   # class nodes receive no relational boost
                "ch_s":  round(ch["surprise"].get(cid, 0.0), 4),
            }
        if ikb is not None:
            for iid in ikb.instance_ids():
                inst = ikb.get_instance(iid)
                concepts[iid] = {
                    "a":     round(attn.get(iid, 0.0), 4),
                    "e":     round(inst.epistemic_error, 4),
                    "pe":    round(inst.prediction_error, 4),
                    "ch_m":  round(ch["mission"].get(inst.class_id, 0.0), 4),
                    "ch_an": round(ch["anticipatory"].get(inst.class_id, 0.0), 4),
                    "ch_r":  round(ch["relational"].get(iid, 0.0), 4),
                    "ch_s":  round(ch["surprise"].get(iid, 0.0), 4),
                }

        if extra_events:
            events.extend(extra_events)

        record = {
            "t": round(elapsed, 3),
            "goal": cur_goal,
            "queue": [[gid, round(eta, 3), lvl] for gid, eta, lvl in strategy.mission_queue],
            "schedule": list(schedule),
            "events": events,
            "concepts": concepts,
        }
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")

    # ------------------------------------------------------------------
    # Meta builder (reused by runner.flush_trace for the save-button feature)
    # ------------------------------------------------------------------

    def _build_meta(self, positions: dict | None = None, truncated: bool = False) -> dict:
        strategy = self._strategy
        kb = strategy.kb
        ikb = strategy.instance_kb

        classes = [
            {
                "id": cid,
                "concept_type": kb.get_concept(cid).concept_type,
                "decay_rate": kb.get_concept(cid).decay_rate,
            }
            for cid in kb.concept_ids()
        ]
        instances = []
        if ikb is not None:
            for iid in ikb.instance_ids():
                inst = ikb.get_instance(iid)
                instances.append({
                    "id": iid,
                    "class_id": inst.class_id,
                    "decay_rate": inst.decay_rate,
                })

        meta: dict = {
            "schema_version": 2,
            "scenario": self._scenario,
            "wall_start": datetime.datetime.utcnow().isoformat() + "Z",
            "git_commit": _git_commit(),
            "strategy": strategy.strategy_name(),
            "strategy_params": strategy.strategy_params(),
            "budget": strategy.budget,
            "dt": self._dt,
            "sample_rate": self._sample_rate,
            "structure": {
                "classes": classes,
                "instances": instances,
                "semantic_edges": [
                    [id_a, id_b, weight]
                    for id_a, id_b, weight in kb.semantic_edges()
                ],
                "relational_edges": [
                    [id_a, id_b, weight, rtype]
                    for id_a, id_b, weight, rtype
                    in (ikb.relational_edges() if ikb is not None else [])
                ],
            },
            "history_maxlen": self._history_maxlen,
            "truncated": truncated,
        }
        if positions is not None:
            meta["positions"] = positions
        return meta
