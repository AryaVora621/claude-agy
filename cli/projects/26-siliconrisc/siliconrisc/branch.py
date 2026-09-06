"""Branch Prediction Unit, BTB, RAS, Gshare, and Tournament Predictors.

Provides:
1. Branch Target Buffer (BTB) for fast target address caching
2. Return Address Stack (RAS) for zero-latency subroutine call/return pairing
3. 2-Bit Saturating Counter Bimodal Local Predictor
4. Gshare Predictor with XOR-folded Global History Register (GHR)
5. Tournament (Meta) Predictor arbitrating between Local and Global predictors
6. Comprehensive prediction accuracy telemetry
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


class PredictorType(Enum):
    ALWAYS_NOT_TAKEN = auto()
    ALWAYS_TAKEN = auto()
    BIMODAL = auto()
    GSHARE = auto()
    TOURNAMENT = auto()


# =====================================================================
# Branch Target Buffer (BTB)
# =====================================================================

@dataclass
class BTBEntry:
    tag: int
    target_pc: int
    valid: bool = True


class BranchTargetBuffer:
    """Direct-mapped Branch Target Buffer caching predicted target PCs."""

    def __init__(self, capacity: int = 512) -> None:
        self.capacity = capacity
        assert (capacity & (capacity - 1)) == 0, "Capacity must be power of 2"
        self.index_mask = capacity - 1
        self.entries: List[Optional[BTBEntry]] = [None] * capacity

    def lookup(self, pc: int) -> Optional[int]:
        idx = (pc >> 2) & self.index_mask
        entry = self.entries[idx]
        if entry is not None and entry.valid and entry.tag == pc:
            return entry.target_pc
        return None

    def update(self, pc: int, target_pc: int) -> None:
        idx = (pc >> 2) & self.index_mask
        self.entries[idx] = BTBEntry(tag=pc, target_pc=target_pc, valid=True)

    def flush(self) -> None:
        for i in range(self.capacity):
            self.entries[i] = None


# =====================================================================
# Return Address Stack (RAS)
# =====================================================================

class ReturnAddressStack:
    """Fixed-capacity LIFO stack predicting function return target addresses."""

    def __init__(self, capacity: int = 16) -> None:
        self.capacity = capacity
        self.stack: List[int] = []

    def push(self, return_pc: int) -> None:
        if len(self.stack) >= self.capacity:
            self.stack.pop(0)  # Drop oldest entry on overflow
        self.stack.append(return_pc)

    def pop(self) -> Optional[int]:
        if not self.stack:
            return None
        return self.stack.pop()

    def flush(self) -> None:
        self.stack.clear()


# =====================================================================
# 2-Bit Saturating Counter Predictors (Bimodal, Gshare, Tournament)
# =====================================================================

class BimodalPredictor:
    """2-bit saturating counter local branch predictor."""

    def __init__(self, num_entries: int = 1024) -> None:
        self.num_entries = num_entries
        self.mask = num_entries - 1
        # Initialize to 01 (Weakly Not Taken)
        self.counters: List[int] = [1] * num_entries

    def predict(self, pc: int) -> bool:
        idx = (pc >> 2) & self.mask
        return self.counters[idx] >= 2

    def update(self, pc: int, actual_taken: bool) -> None:
        idx = (pc >> 2) & self.mask
        val = self.counters[idx]
        if actual_taken:
            self.counters[idx] = min(3, val + 1)
        else:
            self.counters[idx] = max(0, val - 1)


class GsharePredictor:
    """Gshare predictor XORing Global History Register with PC bits."""

    def __init__(self, history_bits: int = 10) -> None:
        self.history_bits = history_bits
        self.num_entries = 1 << history_bits
        self.mask = self.num_entries - 1
        self.ghr: int = 0  # Global history register
        self.pht: List[int] = [1] * self.num_entries  # Pattern history table

    def get_index(self, pc: int) -> int:
        pc_bits = (pc >> 2) & self.mask
        return pc_bits ^ (self.ghr & self.mask)

    def predict(self, pc: int) -> bool:
        idx = self.get_index(pc)
        return self.pht[idx] >= 2

    def update(self, pc: int, actual_taken: bool) -> None:
        idx = self.get_index(pc)
        val = self.pht[idx]
        if actual_taken:
            self.pht[idx] = min(3, val + 1)
        else:
            self.pht[idx] = max(0, val - 1)

        # Shift in actual branch outcome to GHR
        self.ghr = ((self.ghr << 1) | (1 if actual_taken else 0)) & self.mask


class TournamentPredictor:
    """Meta-predictor selecting between Bimodal (Local) and Gshare (Global) predictors."""

    def __init__(self, size: int = 1024) -> None:
        self.local_pred = BimodalPredictor(num_entries=size)
        self.global_pred = GsharePredictor(history_bits=10)
        self.meta_counters: List[int] = [1] * size  # 0,1: prefer local; 2,3: prefer global
        self.mask = size - 1

    def predict(self, pc: int) -> Tuple[bool, bool]:
        """Returns (predicted_direction, used_global)."""
        idx = (pc >> 2) & self.mask
        use_global = self.meta_counters[idx] >= 2
        pred = self.global_pred.predict(pc) if use_global else self.local_pred.predict(pc)
        return pred, use_global

    def update(self, pc: int, actual_taken: bool) -> None:
        idx = (pc >> 2) & self.mask
        local_pred = self.local_pred.predict(pc)
        global_pred = self.global_pred.predict(pc)

        # Train meta-predictor only when local and global predictions differ
        if local_pred != global_pred:
            if global_pred == actual_taken:
                # Global was right -> increment towards global
                self.meta_counters[idx] = min(3, self.meta_counters[idx] + 1)
            else:
                # Local was right -> decrement towards local
                self.meta_counters[idx] = max(0, self.meta_counters[idx] - 1)

        # Train both underlying predictors
        self.local_pred.update(pc, actual_taken)
        self.global_pred.update(pc, actual_taken)


# =====================================================================
# Unified Branch Predictor Engine
# =====================================================================

class BranchPredictorUnit:
    """Complete Branch Predictor combining Direction Predictor, BTB, and RAS."""

    def __init__(
        self,
        predictor_type: PredictorType = PredictorType.TOURNAMENT,
        btb_entries: int = 512,
        ras_entries: int = 16,
    ) -> None:
        self.predictor_type = predictor_type
        self.btb = BranchTargetBuffer(capacity=btb_entries)
        self.ras = ReturnAddressStack(capacity=ras_entries)

        self.bimodal = BimodalPredictor()
        self.gshare = GsharePredictor()
        self.tournament = TournamentPredictor()

        # Telemetry Statistics
        self.total_branches: int = 0
        self.correct_predictions: int = 0
        self.mispredictions: int = 0
        self.btb_hits: int = 0
        self.ras_hits: int = 0

    @property
    def accuracy(self) -> float:
        if self.total_branches == 0:
            return 1.0
        return self.correct_predictions / self.total_branches

    def predict(
        self,
        pc: int,
        is_conditional: bool = False,
        is_call: bool = False,
        is_return: bool = False,
    ) -> Tuple[bool, Optional[int]]:
        """Predict whether branch/jump is taken and predict target address.

        Returns: (predict_taken, predicted_target_pc)
        """
        # Return instruction -> consult RAS first
        if is_return:
            ret_target = self.ras.pop()
            if ret_target is not None:
                self.ras_hits += 1
                return True, ret_target

        # Unconditional Jumps (JAL / JALR) are always taken
        if not is_conditional:
            btb_target = self.btb.lookup(pc)
            if btb_target is not None:
                self.btb_hits += 1
            return True, btb_target

        # Conditional branch direction prediction
        taken = False
        if self.predictor_type == PredictorType.ALWAYS_NOT_TAKEN:
            taken = False
        elif self.predictor_type == PredictorType.ALWAYS_TAKEN:
            taken = True
        elif self.predictor_type == PredictorType.BIMODAL:
            taken = self.bimodal.predict(pc)
        elif self.predictor_type == PredictorType.GSHARE:
            taken = self.gshare.predict(pc)
        elif self.predictor_type == PredictorType.TOURNAMENT:
            taken, _ = self.tournament.predict(pc)

        target = self.btb.lookup(pc) if taken else None
        if target is not None:
            self.btb_hits += 1

        return taken, target

    def update(
        self,
        pc: int,
        predicted_taken: bool,
        actual_taken: bool,
        actual_target: int,
        is_conditional: bool = False,
        is_call: bool = False,
        return_pc: Optional[int] = None,
    ) -> None:
        """Update branch predictor tables with actual branch outcome."""
        if is_call and return_pc is not None:
            self.ras.push(return_pc)

        if actual_taken:
            self.btb.update(pc, actual_target)

        if is_conditional:
            self.total_branches += 1
            if predicted_taken == actual_taken:
                self.correct_predictions += 1
            else:
                self.mispredictions += 1

            if self.predictor_type == PredictorType.BIMODAL:
                self.bimodal.update(pc, actual_taken)
            elif self.predictor_type == PredictorType.GSHARE:
                self.gshare.update(pc, actual_taken)
            elif self.predictor_type == PredictorType.TOURNAMENT:
                self.tournament.update(pc, actual_taken)
