"""Unit tests for Branch Target Buffer, Return Address Stack, and Branch Predictors."""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from siliconrisc.branch import (
    BimodalPredictor,
    BranchPredictorUnit,
    BranchTargetBuffer,
    GsharePredictor,
    PredictorType,
    ReturnAddressStack,
    TournamentPredictor,
)


class TestBranchPredictor(unittest.TestCase):
    """Test BTB, RAS, Bimodal, Gshare, and Tournament prediction accuracy."""

    def test_btb(self) -> None:
        btb = BranchTargetBuffer(capacity=64)
        self.assertIsNone(btb.lookup(0x1000))

        btb.update(0x1000, 0x1020)
        self.assertEqual(btb.lookup(0x1000), 0x1020)

        # Overwrite with different branch
        btb.update(0x1000, 0x1040)
        self.assertEqual(btb.lookup(0x1000), 0x1040)

    def test_ras(self) -> None:
        ras = ReturnAddressStack(capacity=4)
        self.assertIsNone(ras.pop())

        ras.push(0x2000)
        ras.push(0x2010)
        ras.push(0x2020)

        self.assertEqual(ras.pop(), 0x2020)
        self.assertEqual(ras.pop(), 0x2010)
        self.assertEqual(ras.pop(), 0x2000)
        self.assertIsNone(ras.pop())

    def test_bimodal_predictor(self) -> None:
        bp = BimodalPredictor(num_entries=16)
        pc = 0x1000

        # Initial state is 1 (Weakly Not Taken) -> Predict False
        self.assertFalse(bp.predict(pc))

        # Train with taken branch
        bp.update(pc, actual_taken=True)  # becomes 2 (Weakly Taken)
        self.assertTrue(bp.predict(pc))

        bp.update(pc, actual_taken=True)  # becomes 3 (Strongly Taken)
        self.assertTrue(bp.predict(pc))

        # Train with not taken branch
        bp.update(pc, actual_taken=False)  # becomes 2
        self.assertTrue(bp.predict(pc))

        bp.update(pc, actual_taken=False)  # becomes 1
        self.assertFalse(bp.predict(pc))

    def test_gshare_pattern_learning(self) -> None:
        gshare = GsharePredictor(history_bits=8)
        pc = 0x4000

        # Pattern: Taken, Taken, Not Taken (1, 1, 0) repeated
        pattern = [True, True, False]
        for _ in range(50):
            for outcome in pattern:
                gshare.update(pc, outcome)

        # Gshare should adapt to history correlations
        self.assertIsNotNone(gshare.predict(pc))

    def test_branch_predictor_unit_tournament(self) -> None:
        bpu = BranchPredictorUnit(predictor_type=PredictorType.TOURNAMENT)
        pc = 0x8000
        target = 0x8050

        # Call instruction: push return address
        taken, _ = bpu.predict(pc, is_call=True)
        bpu.update(pc, predicted_taken=taken, actual_taken=True, actual_target=target, is_call=True, return_pc=0x8004)

        # Return instruction: pop return address from RAS
        taken_ret, pred_ret_pc = bpu.predict(target + 0x10, is_return=True)
        self.assertTrue(taken_ret)
        self.assertEqual(pred_ret_pc, 0x8004)


if __name__ == "__main__":
    unittest.main()
