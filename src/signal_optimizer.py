"""
signal_optimizer.py — Dynamic traffic signal timing decision engine.

Given real-time density measurements and ML congestion predictions for
multiple lanes/junctions, computes optimal green-phase durations using
a weighted proportional allocation strategy with safety constraints.

Algorithm
---------
1. For each lane, compute a *pressure* score combining:
     - EMA vehicle count (normalised)
     - Predicted congestion probability (weighted)
     - Trend multiplier (rising → increase pressure)
2. Distribute total cycle time proportionally to lane pressures
3. Apply hard floor (min_green) and ceiling (max_green) safety clamps
4. Emit a PhaseSchedule containing per-lane green times + advisory messages

Design notes
------------
• No global state — optimiser is a pure function of its inputs
• All timing is in seconds; durations are integers for real-world controller compatibility
• Extensible: subclass SignalOptimizer and override _compute_pressure()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from src.density_analyzer import FrameDensity
from src.predictor import PredictionResult
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LaneSignalInput:
    """
    All information needed to optimise the signal for one lane.

    Parameters
    ----------
    lane_name       : Matches DensityAnalyzer.lanes[i].name.
    density         : Current FrameDensity for this lane.
    prediction      : CongestionPredictor.predict() result for this lane.
    trend           : One of "rising", "stable", "falling" from DensityAnalyzer.trend().
    waiting_time_s  : Optional: how long vehicles have been waiting (seconds).
                      Boosts pressure to prevent starvation.
    """
    lane_name:      str
    density:        FrameDensity
    prediction:     PredictionResult
    trend:          str             = "stable"
    waiting_time_s: float           = 0.0


@dataclass
class LaneSignalOutput:
    """Recommended signal timing for one lane."""
    lane_name:       str
    green_time_s:    int            # recommended green phase duration (seconds)
    pressure:        float          # raw pressure score (diagnostic)
    advisory:        str            # human-readable explanation


@dataclass
class PhaseSchedule:
    """Complete signal timing schedule for one optimisation cycle."""
    cycle_time_s:    int
    lanes:           list[LaneSignalOutput]
    total_pressure:  float
    dominant_lane:   str            # lane with highest pressure
    notes:           list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

class SignalOptimizer:
    """
    Proportional traffic signal optimiser.

    Parameters
    ----------
    cycle_time_s    : Total signal cycle duration (all lanes) in seconds.
    min_green_s     : Hard minimum green time per lane (pedestrian safety).
    max_green_s     : Hard maximum green time per lane (starvation prevention).
    high_prob_weight: Weight applied to High-congestion probability in pressure calc.
    trend_multiplier: Factor applied to pressure when trend is "rising".
    starvation_k    : Pressure bonus per second of waiting time.
    """

    def __init__(
        self,
        cycle_time_s:     int   = 120,
        min_green_s:      int   = 10,
        max_green_s:      int   = 90,
        high_prob_weight: float = 0.50,
        trend_multiplier: float = 1.25,
        starvation_k:     float = 0.005,
    ) -> None:
        if min_green_s >= max_green_s:
            raise ValueError("min_green_s must be strictly less than max_green_s.")

        self.cycle_time_s     = cycle_time_s
        self.min_green_s      = min_green_s
        self.max_green_s      = max_green_s
        self.high_prob_weight = high_prob_weight
        self.trend_multiplier = trend_multiplier
        self.starvation_k     = starvation_k

    # ------------------------------------------------------------------
    def optimise(self, lane_inputs: Sequence[LaneSignalInput]) -> PhaseSchedule:
        """
        Compute optimal green-phase durations for all lanes.

        Parameters
        ----------
        lane_inputs : One LaneSignalInput per lane, ordered by evaluation priority.

        Returns
        -------
        PhaseSchedule with per-lane timings.
        """
        if not lane_inputs:
            raise ValueError("lane_inputs must not be empty.")

        n = len(lane_inputs)

        # Validate feasibility
        if self.min_green_s * n > self.cycle_time_s:
            raise ValueError(
                f"cycle_time_s ({self.cycle_time_s}s) is too short to satisfy "
                f"min_green_s={self.min_green_s}s for {n} lanes. "
                f"Increase cycle_time_s to at least {self.min_green_s * n}s."
            )

        # --- Pressure scores per lane -------------------------------------
        pressures = [self._compute_pressure(li) for li in lane_inputs]
        total_pressure = sum(pressures)

        # Proportional green times
        raw_greens: list[float] = []
        if total_pressure <= 0:
            # No signal → distribute evenly
            equal = self.cycle_time_s / n
            raw_greens = [equal] * n
        else:
            for p in pressures:
                raw_greens.append((p / total_pressure) * self.cycle_time_s)

        # Clamp to [min_green_s, max_green_s]
        clamped = [
            max(self.min_green_s, min(int(round(g)), self.max_green_s))
            for g in raw_greens
        ]

        # Re-balance: ensure sum equals cycle_time_s exactly
        clamped = self._rebalance(clamped)

        # --- Build outputs ------------------------------------------------
        outputs: list[LaneSignalOutput] = []
        for li, p, gt in zip(lane_inputs, pressures, clamped):
            advisory = self._advisory(li, gt)
            outputs.append(LaneSignalOutput(
                lane_name    = li.lane_name,
                green_time_s = gt,
                pressure     = round(p, 4),
                advisory     = advisory,
            ))

        dominant = max(outputs, key=lambda o: o.pressure).lane_name
        notes    = self._global_notes(lane_inputs, outputs)

        schedule = PhaseSchedule(
            cycle_time_s   = sum(o.green_time_s for o in outputs),
            lanes          = outputs,
            total_pressure = round(total_pressure, 4),
            dominant_lane  = dominant,
            notes          = notes,
        )

        logger.info(
            "Signal optimised | cycle=%ds | dominant=%s | pressures=%s",
            schedule.cycle_time_s,
            dominant,
            {o.lane_name: round(o.pressure, 2) for o in outputs},
        )

        return schedule

    # ------------------------------------------------------------------
    # Pressure calculation (override in subclasses)
    # ------------------------------------------------------------------

    def _compute_pressure(self, li: LaneSignalInput) -> float:
        """
        Compute a non-negative pressure score for a lane.

        Components
        ----------
        count_pressure : EMA vehicle count normalised by the "High" threshold (25).
        prob_pressure  : Weighted probability of High congestion.
        trend_factor   : Multiplier for rising traffic.
        starvation     : Bonus to prevent a lane from being starved of green time.
        """
        count_pressure = li.density.ema_count / 25.0  # normalise to [0, ∞)

        prob_high      = li.prediction.probabilities.get("High", 0.0)
        prob_medium    = li.prediction.probabilities.get("Medium", 0.0)
        prob_pressure  = (
            self.high_prob_weight * prob_high
            + (1 - self.high_prob_weight) * prob_medium
        )

        # Congestion score contributes a small continuous signal
        cong_pressure  = li.density.congestion_score / 100.0

        raw = (
            0.50 * count_pressure
            + 0.30 * prob_pressure
            + 0.20 * cong_pressure
        )

        # Trend multiplier
        if li.trend == "rising":
            raw *= self.trend_multiplier
        elif li.trend == "falling":
            raw *= (2.0 - self.trend_multiplier)   # symmetric de-boost

        # Starvation prevention
        raw += self.starvation_k * li.waiting_time_s

        return max(raw, 0.0)

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def _rebalance(self, clamped: list[int]) -> list[int]:
        """
        Adjust clamped green times so their sum equals cycle_time_s.

        Strategy: distribute surplus/deficit starting with the lane
        furthest from its clamp boundary.
        """
        target = self.cycle_time_s
        diff   = target - sum(clamped)
        if diff == 0:
            return clamped

        result = list(clamped)
        step   = 1 if diff > 0 else -1

        for _ in range(abs(diff)):
            # Find the lane with most room to absorb ±1 second
            best_idx, best_room = 0, -1
            for i, g in enumerate(result):
                if step > 0:
                    room = self.max_green_s - g
                else:
                    room = g - self.min_green_s

                if room > best_room:
                    best_room, best_idx = room, i

            if best_room > 0:
                result[best_idx] += step
            else:
                # Fallback: ignore max limits if we MUST fill the target to avoid cycle shrinkage
                if step > 0:
                    result[0] += 1
                else:
                    # Should not happen given min_green_s * n <= cycle_time_s check
                    result[0] -= 1


        return result

    # ------------------------------------------------------------------
    # Advisory text
    # ------------------------------------------------------------------

    def _advisory(self, li: LaneSignalInput, green_time: int) -> str:
        label  = li.prediction.label
        trend  = li.trend

        if label == "High" and trend == "rising":
            return (
                f"CRITICAL: {li.lane_name} is severely congested and worsening. "
                f"Green extended to {green_time}s. Consider incident alert."
            )
        if label == "High":
            return (
                f"WARNING: {li.lane_name} is heavily congested. "
                f"Green extended to {green_time}s."
            )
        if label == "Medium" and trend == "rising":
            return (
                f"CAUTION: {li.lane_name} congestion is rising. "
                f"Green pre-emptively extended to {green_time}s."
            )
        if label == "Low" and trend == "falling":
            return (
                f"OK: {li.lane_name} clearing. "
                f"Green reduced to {green_time}s to free other lanes."
            )
        return (
            f"NORMAL: {li.lane_name} at {label.lower()} density. "
            f"Green time {green_time}s."
        )

    # ------------------------------------------------------------------
    # Global notes
    # ------------------------------------------------------------------

    def _global_notes(
        self,
        inputs:  Sequence[LaneSignalInput],
        outputs: list[LaneSignalOutput],
    ) -> list[str]:
        notes: list[str] = []

        all_high = all(
            li.prediction.label == "High" for li in inputs
        )
        if all_high:
            notes.append(
                "All lanes critically congested. Consider activating overflow routing."
            )

        at_max = [o for o in outputs if o.green_time_s == self.max_green_s]
        if len(at_max) == len(outputs):
            notes.append(
                "All lanes clamped at max_green_s. Increase cycle_time_s for better resolution."
            )

        at_min = [o for o in outputs if o.green_time_s == self.min_green_s]
        if at_min:
            lanes = ", ".join(o.lane_name for o in at_min)
            notes.append(
                f"Lanes at minimum green ({self.min_green_s}s): {lanes}. "
                "Monitor for starvation."
            )

        return notes