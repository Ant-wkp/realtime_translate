"""DOA angle to zone labels for display and beam reference.

A zone is a label plus an explicit azimuth range in degrees. Ranges are
user-editable because the labels are only meaningful relative to how the
array is actually mounted: the same angle is "front" on a desk and "left"
when the array sits sideways.

Angles are display-only metadata on each segment; they never gate audio.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

# (label, start, end) in degrees, clockwise, 0 = the array's own front.
# Ranges may wrap past 360 and may leave gaps; an angle in no range gets "".
DEFAULT_ZONES: list[tuple[str, float, float]] = [
    ("front", 315.0, 45.0),
    ("right", 45.0, 135.0),
    ("back", 135.0, 225.0),
    ("left", 225.0, 315.0),
]

# the evenly-divided presets the UI offers as starting points
ZONE_MAPS = {
    2: ["front", "back"],
    4: ["front", "right", "back", "left"],
    8: ["front", "front-right", "right", "back-right",
        "back", "back-left", "left", "front-left"],
}


@dataclass
class Zone:
    label: str
    start: float
    end: float

    def as_dict(self) -> dict:
        return {"label": self.label, "start": self.start, "end": self.end}

    def contains(self, angle: float) -> bool:
        """Is `angle` inside [start, end), allowing a range that wraps 360?"""
        angle = (angle % 360 + 360) % 360
        start = (self.start % 360 + 360) % 360
        end = (self.end % 360 + 360) % 360
        if start == end:
            return False  # empty range: covers nothing rather than everything
        if start < end:
            return start <= angle < end
        return angle >= start or angle < end   # wraps past 360


def zones_from_labels(labels: list[str]) -> list[Zone]:
    """Build equal ranges from bare labels (the legacy even-split behavior)."""
    n = len(labels)
    if n == 0:
        return []
    step = 360.0 / n
    return [Zone(label, i * step, (i + 1) * step) for i, label in enumerate(labels)]


class SpeakerLocator:
    def __init__(self, zones: list[Zone] | None = None):
        self.zones: list[Zone] = list(zones) if zones else [
            Zone(lbl, s, e) for lbl, s, e in DEFAULT_ZONES]
        self._counter: Counter = Counter()
        self._last_angle: float | None = None

    # ------------------------------------------------------------------
    def label_for(self, angle: float) -> str:
        """First zone whose range contains `angle`, else "" (a gap)."""
        for z in self.zones:
            if z.contains(angle):
                return z.label
        return ""

    def observe(self, angle: float) -> None:
        """Accumulate one DOA reading for the running segment."""
        self._last_angle = angle
        label = self.label_for(angle)
        if label:
            self._counter[label] += 1

    def mode(self) -> str:
        """Most frequent zone during the current segment."""
        return self._counter.most_common(1)[0][0] if self._counter else ""

    def reset(self) -> None:
        """Forget the accumulated readings; called once per segment so the
        reported direction describes this utterance rather than the session."""
        self._counter.clear()
        self._last_angle = None

    # config --------------------------------------------------------------
    def set_zones(self, zones: list[Zone]) -> None:
        if zones:
            self.zones = list(zones)

    def set_zones_from_dicts(self, zones: list[dict]) -> None:
        """Accept the wire format: [{label, start, end}, ...]."""
        parsed: list[Zone] = []
        for z in zones:
            label = str(z.get("label", "")).strip()
            if not label:
                continue
            parsed.append(Zone(label, float(z.get("start", 0.0)),
                               float(z.get("end", 0.0))))
        if not parsed:
            raise ValueError("no usable zones (each needs a label)")
        self.zones = parsed

    def get_zones(self) -> list[dict]:
        return [z.as_dict() for z in self.zones]
