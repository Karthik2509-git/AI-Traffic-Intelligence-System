"""
signal_optimizer.py — Dynamic traffic signal timing and optimization engine.

Maintains optimal throughput at intersections by calculating adaptive 
green-phase durations based on real-time vehicle density, ML-based 
congestion forecasts, and temporal traffic trends.
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
    lane_name : str
        Matches DensityAnalyzer.lanes[i].name.
    density : FrameDensity
        Current spatial density snapshot for this lane.
    prediction : PredictionResult
        Congestion forecast for this lane.
    trend : str
        Temporal volume trend ('rising', 'stable', 'falling').
    waiting_time_s : float
        Cumulative wait duration for vehicles in this lane (seconds).
        Used to prevent lane starvation.
    """
    lane_name: str
    density: FrameDensity
    prediction: PredictionResult
    trend: str = "stable"
    waiting_time_s: float = 0.0


@dataclass
class LaneSignalOutput:
    """Recommended signal timing for one lane."""
    lane_name: str
    green_time_s: int
    pressure: float
    advisory: str


@dataclass
class PhaseSchedule:
    """Complete signal timing schedule for one optimisation cycle."""
    cycle_time_s: int
    lanes: list[LaneSignalOutput]
    total_pressure: float
    dominant_lane: str
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

class SignalOptimizer:
    """
    Proportional-weighted signal optimization engine.

    Computes ideal green-time distributions across multiple lanes by 
    calculating a 'pressure' score for each segment. Pressure is derived 
    from volume, congestion probability, and temporal acceleration.

    Parameters
    ----------
    cycle_time_s : int
        Total duration for one complete signal cycle across all lanes.
    min_green_s : int
        Safety floor for green time per lane (e.g., pedestrian clearance).
    max_green_s : int
        Ceiling for green time to prevent starvation in other lanes.
    high_prob_weight : float
        Contribution of high-congestion forecasts to the pressure score.
    trend_multiplier : float
        Scalar applied when traffic volume trend is 'rising'.
    starvation_k : float
        Penalty factor applied based on vehicle waiting duration.
    """

    def __init__(
        self,
        cycle_time_s: int = 120,
        min_green_s: int = 10,
        max_green_s: int = 90,
        high_prob_weight: float = 0.50,
        trend_multiplier: float = 1.25,
        starvation_k: float = 0.005,
    ) -> None:
        if min_green_s >= max_green_s:
            raise ValueError("min_green_s must be strictly less than max_green_s.")

        self.cycle_time_s = cycle_time_s
        self.min_green_s = min_green_s
        self.max_green_s = max_green_s
        self.high_prob_weight = high_prob_weight
        self.trend_multiplier = trend_multiplier
        self.starvation_k = starvation_k

    def optimise(self, lane_inputs: Sequence[LaneSignalInput]) -> PhaseSchedule:
        """
        Derive the optimal green-phase schedule for all configured lanes.

        Parameters
        ----------
        lane_inputs : Sequence[LaneSignalInput]
            Telemetry and forecasts for each lane in the intersection.

        Returns
        -------
        PhaseSchedule
            A complete timing schedule with advisory metadata.
        """
        if not lane_inputs:
            raise ValueError("lane_inputs must not be empty.")

        n = len(lane_inputs)

        # Validate feasibility against safety constraints
        if self.min_green_s * n > self.cycle_time_s:
            raise ValueError(
                f"cycle_time_s ({self.cycle_time_s}s) is too short to satisfy "
                f"min_green_s={self.min_green_s}s for {n} lanes. "
                f"Increase cycle_time_s to at least {self.min_green_s * n}s."
            )

        # Calculate demand pressure for each lane node
        pressures = [self._compute_pressure(li) for li in lane_inputs]
        total_pressure = sum(pressures)

        # Map pressure to proportional green-time slices
        raw_greens: list[float] = []
        if total_pressure <= 0:
            equal = self.cycle_time_s / n
            raw_greens = [equal] * n
        else:
            for p in pressures:
                raw_greens.append((p / total_pressure) * self.cycle_time_s)

        # Boundary enforcement
        clamped = [
            max(self.min_green_s, min(int(round(g)), self.max_green_s))
            for g in raw_greens
        ]

        # Cycle re-balancing for integer accuracy
        clamped = self._rebalance(clamped)

        # Final allocation assembly
        outputs: list[LaneSignalOutput] = []
        for li, p, gt in zip(lane_inputs, pressures, clamped):
            outputs.append(LaneSignalOutput(
                lane_name=li.lane_name,
                green_time_s=gt,
                pressure=round(p, 4),
                advisory=self._advisory(li, gt),
            ))

        dominant = max(outputs, key=lambda o: o.pressure).lane_name
        notes = self._global_notes(lane_inputs, outputs)

        schedule = PhaseSchedule(
            cycle_time_s=sum(o.green_time_s for o in outputs),
            lanes=outputs,
            total_pressure=round(total_pressure, 4),
            dominant_lane=dominant,
            notes=notes,
        )

        logger.info(
            "Signal optimised | cycle=%ds | dominant=%s | max_pressure=%.2f",
            schedule.cycle_time_s,
            dominant,
            max(pressures) if pressures else 0.0,
        )

        return schedule

    def _compute_pressure(self, li: LaneSignalInput) -> float:
        """
        Derive a composite pressure score for a specific lane.

        Heuristic weighting of volume, forecasted risk, and cumulative 
        waiting duration.

        Parameters
        ----------
        li : LaneSignalInput
            Current telemetry and forecasts for the lane.

        Returns
        -------
        float
            Demand pressure scalar.
        """
        # Volume pressure normalised against typical threshold
        count_pressure = li.density.ema_count / 25.0

        probs = li.prediction.probabilities
        prob_high = probs.get("High", 0.0)
        prob_medium = probs.get("Medium", 0.0)
        prob_pressure = (
            self.high_prob_weight * prob_high
            + (1 - self.high_prob_weight) * prob_medium
        )

        cong_pressure = li.density.congestion_score / 100.0

        # Weighted demand model
        raw = (
            0.50 * count_pressure
            + 0.30 * prob_pressure
            + 0.20 * cong_pressure
        )

        # Temporal acceleration boosters
        if li.trend == "rising":
            raw *= self.trend_multiplier
        elif li.trend == "falling":
            raw *= (2.0 - self.trend_multiplier)

        # Starvation prevention
        raw += self.starvation_k * li.waiting_time_s

        return max(raw, 0.0)

    def _rebalance(self, clamped: list[int]) -> list[int]:
        """
        Adjust green-time allocations to meet strict cycle duration constraints.

        Parameters
        ----------
        clamped : list[int]
            Initial clipped allocations.

        Returns
        -------
        list[int]
            Balanced integer seconds for each lane.
        """
        target = self.cycle_time_s
        diff = target - sum(clamped)
        if diff == 0:
            return clamped

        result = list(clamped)
        step = 1 if diff > 0 else -1

        for _ in range(abs(diff)):
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
                result[0] += step

        return result

    def _advisory(self, li: LaneSignalInput, green_time: int) -> str:
        """
        Generate a human-readable advisory message for a lane's timing decision.

        Parameters
        ----------
        li : LaneSignalInput
            Telemetry for the lane.
        green_time : int
            Final green duration.

        Returns
        -------
        str
            Professional status message.
        """
        label = li.prediction.label
        trend = li.trend

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

    def _global_notes(
        self,
        inputs: Sequence[LaneSignalInput],
        outputs: list[LaneSignalOutput],
    ) -> list[str]:
        """
        Synthesize system-wide operational observations.

        Parameters
        ----------
        inputs : Sequence[LaneSignalInput]
            Input data for all lanes.
        outputs : list[LaneSignalOutput]
            Recommended results.

        Returns
        -------
        list[str]
            High-level alerts.
        """
        notes: list[str] = []

        all_high = all(li.prediction.label == "High" for li in inputs)
        if all_high:
            notes.append(
                "CRITICAL: All lanes congested. Activating overflow routing recommended."
            )

        at_max = [o for o in outputs if o.green_time_s == self.max_green_s]
        if at_max and len(at_max) == len(outputs):
            notes.append(
                f"Operational Bottleneck: All lanes at max threshold ({self.max_green_s}s). "
                "Consider increasing total cycle_time_s."
            )

        at_min = [o for o in outputs if o.green_time_s == self.min_green_s]
        if at_min:
            lanes = ", ".join(o.lane_name for o in at_min)
            notes.append(
                f"Information: Safety minimum green ({self.min_green_s}s) active for: {lanes}. "
                "Monitoring for potential starvation."
            )

        return notes