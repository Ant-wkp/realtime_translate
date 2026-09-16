"""DOA angle to zone labels for display and beam reference.

Zones are evenly divided rings around the array (supporting 0/360 wrap).
Angles are display-only metadata on each segment; they never gate audio.
"""

from __future__ import annotations

from collections import Counter

ZONE_MAPS = {
    2: ["front", "back"],
    4: ["front", "right", "back", "left"],
    8: ["front", "front-right", "right", "back-right",
        "back", "back-left", "left", "front-left"],
}


class SpeakerLocator:
    def __init__(self, zone_map: list[str] | None = None):
        self.zones = list(zone_map or ZONE_MAPS[4])
        self._counter: Counter = Counter()

    # ------------------------------------------------------------------
    def label_for(self, angle: float) -> str:
        """Map an azimuth (deg, 0 = front) to the nearest zone label."""
        n = len(self.zones)
        if n == 0:
            return ""
        angle = (angle % 360 + 360) % 360
        idx = int((angle + 360 / (2 * n)) % 360 // (360 / n))
        return self.zones[min(idx, n - 1)]

    def observe(self, angle: float) -> None:
        """Accumulate one DOA reading for the running segment."""
        self._counter[self.label_for(angle)] += 1

    def mode(self) -> str:
        """Most frequent zone during the current segment."""
        return self._counter.most_common(1)[0][0] if self._counter else ""

    def reset(self) -> None:
        self._counter.clear()

    # config --------------------------------------------------------------
    def set_zones(self, labels: list[str]) -> None:
        if labels:
            self.zones = list(labels)

    def get_zones(self) -> list[str]:
        return list(self.zones)
