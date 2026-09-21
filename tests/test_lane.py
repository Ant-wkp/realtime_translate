"""Tests for the per-lane pieces: the energy gate's hold and the Lane object.

The energy gate is the one part of the dual-lane work that cannot be checked
by listening, because it lives in the interaction between a device reading that
is zero most of the time and a segmenter that stays open through silence. These
tests drive that interaction directly, with no hardware.

    python -m unittest tests.test_lane -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realtime_subtitle.main import (          # noqa: E402
    BEAM_ENERGY_THRESHOLD, Lane,
)

# The array's own numbers, from measurement on the unit: a quiet room peaks
# around 43000 and clapping reaches 2.4M..4.3M. A gate that cannot separate
# those two is the whole feature failing.
QUIET_ROOM = 42916.0
CLAP = 2500000.0


class ScriptedEnergyLane(Lane):
    """A Lane whose energy reading can be set directly."""

    def set_energy(self, beam0: float, beam1: float, beam_index: int = 0):
        self.beam_index = beam_index
        self.energy = [beam0, beam1, 0.0, 0.0]


class TestEnergyGate(unittest.TestCase):
    def test_quiet_room_does_not_open(self):
        ln = ScriptedEnergyLane(index=0, channel=0)
        ln.set_energy(QUIET_ROOM, QUIET_ROOM)
        self.assertFalse(ln.energy_active(100.0, hold=0.6))

    def test_clap_opens(self):
        ln = ScriptedEnergyLane(index=0, channel=0)
        ln.set_energy(CLAP, 0.0)
        self.assertTrue(ln.energy_active(100.0, hold=0.6))

    def test_default_threshold_sits_between_them(self):
        # guards against someone retuning the default into the noise band
        self.assertGreater(BEAM_ENERGY_THRESHOLD, QUIET_ROOM)
        self.assertLess(BEAM_ENERGY_THRESHOLD, CLAP)

    def test_hold_bridges_a_pause(self):
        """SPENERGY reads exactly 0 between words; without the hold the lane
        would be dropped mid-sentence, which is the failure this test exists
        to prevent."""
        ln = ScriptedEnergyLane(index=0, channel=0)
        ln.set_energy(CLAP, 0.0)
        self.assertTrue(ln.energy_active(100.0, hold=0.6))
        ln.set_energy(0.0, 0.0)                      # a pause
        self.assertTrue(ln.energy_active(100.3, hold=0.6))
        self.assertTrue(ln.energy_active(100.59, hold=0.6))

    def test_hold_expires(self):
        ln = ScriptedEnergyLane(index=0, channel=0)
        ln.set_energy(CLAP, 0.0)
        self.assertTrue(ln.energy_active(100.0, hold=0.6))
        ln.set_energy(0.0, 0.0)
        self.assertFalse(ln.energy_active(100.61, hold=0.6))

    def test_hold_refreshes_on_speech(self):
        ln = ScriptedEnergyLane(index=0, channel=0)
        for t, e in [(100.0, CLAP), (100.4, 0.0), (100.5, CLAP)]:
            ln.set_energy(e, 0.0)
            ln.energy_active(t, hold=0.6)
        ln.set_energy(0.0, 0.0)
        # the 100.5 clap restarts the window, so 100.9 is still inside it
        self.assertTrue(ln.energy_active(100.9, hold=0.6))

    def test_beam_index_selects_the_right_beam(self):
        """The gate must read this lane's own beam, not beam 1 always: with
        two lanes that would gate both on one beam."""
        ln = ScriptedEnergyLane(index=1, channel=1)
        ln.set_energy(CLAP, 0.0, beam_index=0)
        self.assertTrue(ln.energy_active(100.0, hold=0.6))
        ln = ScriptedEnergyLane(index=1, channel=1)
        ln.set_energy(CLAP, 0.0, beam_index=1)       # reading beam 2, which is quiet
        self.assertFalse(ln.energy_active(100.0, hold=0.6))

    def test_no_control_plane_never_blocks(self):
        """Audio-only runs (no USB control plane) leave energy empty. The gate
        must then defer to the software VAD rather than closing forever."""
        ln = ScriptedEnergyLane(index=0, channel=0)
        ln.energy = []
        self.assertTrue(ln.energy_active(100.0, hold=0.6))

    def test_short_energy_list_does_not_raise(self):
        ln = ScriptedEnergyLane(index=3, channel=0)
        ln.energy = [CLAP]                            # only beam 1 present
        self.assertTrue(ln.energy_active(100.0, hold=0.6))


class TestLaneIsolation(unittest.TestCase):
    def test_gate_state_is_per_lane(self):
        """Shared gate state is a real hazard: a hardware tick in one lane
        would advance the other lane's confirmation counter."""
        a = Lane(index=0, channel=0)
        b = Lane(index=1, channel=1)
        a.confirm_pending = True
        a.confirm_sw_reject = 4
        self.assertFalse(b.confirm_pending)
        self.assertEqual(b.confirm_sw_reject, 0)

    def test_queues_are_not_shared(self):
        a = Lane(index=0, channel=0)
        b = Lane(index=1, channel=1)
        a.queue.put("x")
        self.assertEqual(b.queue.qsize(), 0)

    def test_energy_hold_is_per_lane(self):
        """The hold timestamps must be independent too, or one lane going
        quiet would keep the other's window alive."""
        a = ScriptedEnergyLane(index=0, channel=0)
        b = ScriptedEnergyLane(index=1, channel=1)
        a.set_energy(CLAP, 0.0)
        a.energy_active(100.0, hold=0.6)
        b.set_energy(0.0, 0.0)
        self.assertFalse(b.energy_active(100.0, hold=0.6))

    def test_defaults(self):
        ln = Lane(index=0, channel=2)
        self.assertEqual(ln.energy_threshold, BEAM_ENERGY_THRESHOLD)
        self.assertEqual(ln.energy, [])
        self.assertEqual(ln.tag, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)