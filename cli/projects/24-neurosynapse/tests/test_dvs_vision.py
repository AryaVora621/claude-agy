"""Unit tests for neuromorphic DVS vision, event filtering, and optical flow."""

import unittest
from neurosynapse.dvs import (
    BackgroundActivityFilter,
    DVSEvent,
    DVSStimulusGenerator,
    DVSStream,
    OpticalFlowEstimator,
    RefractoryFilter,
    SurfaceOfActiveEvents,
)


class TestDVSVision(unittest.TestCase):
    """Test suite for Address-Event Representation and neuromorphic vision algorithms."""

    def test_dvs_stream_slicing(self) -> None:
        """Verify stream creation, slicing, and polarity counts."""
        events = [
            DVSEvent(x=10, y=10, timestamp_us=1000, polarity=1),
            DVSEvent(x=12, y=10, timestamp_us=2000, polarity=-1),
            DVSEvent(x=14, y=10, timestamp_us=3000, polarity=1),
        ]
        stream = DVSStream(events, width=32, height=32)
        self.assertEqual(len(stream), 3)

        sub = stream.slice_time(1500, 3500)
        self.assertEqual(len(sub), 2)
        on, off = sub.count_polarities()
        self.assertEqual(on, 1)
        self.assertEqual(off, 1)

    def test_refractory_filter(self) -> None:
        """Verify rejection of events within refractory window at identical pixels."""
        rf = RefractoryFilter(width=32, height=32, tau_ref_us=5000)
        ev1 = DVSEvent(x=5, y=5, timestamp_us=1000, polarity=1)
        ev2 = DVSEvent(x=5, y=5, timestamp_us=3000, polarity=1)  # 2000us diff < 5000us
        ev3 = DVSEvent(x=5, y=5, timestamp_us=7000, polarity=1)  # 4000us from ev1, but 4000 from dropped ev2? No, 6000us from ev1

        self.assertTrue(rf.filter_event(ev1))
        self.assertFalse(rf.filter_event(ev2))
        self.assertTrue(rf.filter_event(ev3))

    def test_background_activity_filter(self) -> None:
        """Verify rejection of isolated thermal noise events."""
        baf = BackgroundActivityFilter(width=32, height=32, window_us=5000)
        # Isolated event
        e_isolated = DVSEvent(x=20, y=20, timestamp_us=1000, polarity=1)
        self.assertFalse(baf.filter_event(e_isolated))

        # Neighbor event at (21, 20) within 2000us
        e_neighbor = DVSEvent(x=21, y=20, timestamp_us=3000, polarity=1)
        self.assertTrue(baf.filter_event(e_neighbor))

    def test_surface_of_active_events(self) -> None:
        """Verify exponential time surface decay."""
        sae = SurfaceOfActiveEvents(width=16, height=16, tau_decay_us=20000.0)
        sae.update(DVSEvent(x=4, y=4, timestamp_us=10000, polarity=1))

        # Check surface immediately at t=10000
        surf_now = sae.get_decayed_surface(current_time_us=10000, polarity=1)
        self.assertAlmostEqual(surf_now[4][4], 1.0, places=3)

        # Check surface decayed at t=30000 (dt = 20000us = tau -> exp(-1) = 0.3678)
        surf_later = sae.get_decayed_surface(current_time_us=30000, polarity=1)
        self.assertAlmostEqual(surf_later[4][4], 0.3678, delta=0.05)

    def test_optical_flow_estimation(self) -> None:
        """Verify plane-fitting optical flow on moving bar stimulus."""
        stream = DVSStimulusGenerator.moving_vertical_bar(
            width=32, height=32, speed_px_s=50.0, duration_s=0.1
        )
        estimator = OpticalFlowEstimator(width=32, height=32)

        flows = []
        for ev in stream.events:
            f = estimator.estimate_flow(ev)
            if f is not None:
                flows.append(f)

        self.assertGreater(len(flows), 10)
        avg_vx = sum(f[0] for f in flows) / len(flows)
        # Bar is moving rightward (+x), so vx should be positive
        self.assertGreater(avg_vx, 10.0)


if __name__ == "__main__":
    unittest.main()
