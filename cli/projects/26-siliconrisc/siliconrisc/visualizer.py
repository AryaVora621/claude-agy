"""Sub-Pixel Braille Hardware Visualizer, Pipeline HUD, and Telemetry Inspector.

Provides:
1. BrailleCanvas: 2x4 sub-pixel resolution Unicode Braille (U+2800..U+28FF) plotting engine
2. PipelineHUD: 5-stage classic RISC pipeline visualizer displaying IF, ID, EX, MEM, WB
3. RegisterInspector: 32-register formatted grid with ABI names and decimal/hex values
4. CacheTelemetryHUD: L1I, L1D, L2 cache hit-rates, MESI coherence states, and MMU TLB stats
5. SystemWorkbenchDashboard: Unified terminal telemetry screen for interactive execution
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .core import CPU
from .isa import ABI_REG_NAMES, DecodedInstruction, Disassembler, InstructionDecoder
from .pipeline import PipelineStatus, PipelinedCore


# ANSI Color Codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
BG_BLUE = "\033[44m"
BG_DARK = "\033[48;5;236m"


# =====================================================================
# Sub-Pixel Unicode Braille Canvas (2x4 dots per character)
# =====================================================================

class BrailleCanvas:
    """Sub-pixel resolution canvas rendering 2x4 dot matrices into Unicode Braille characters."""

    DOT_MAP = (
        (0x01, 0x08),  # row 0: dot 0 (left), dot 3 (right)
        (0x02, 0x10),  # row 1: dot 1 (left), dot 4 (right)
        (0x04, 0x20),  # row 2: dot 2 (left), dot 5 (right)
        (0x40, 0x80),  # row 3: dot 6 (left), dot 7 (right)
    )

    def __init__(self, char_width: int, char_height: int) -> None:
        self.char_width = char_width
        self.char_height = char_height
        self.pixel_width = char_width * 2
        self.pixel_height = char_height * 4
        self.grid = [[0 for _ in range(char_width)] for _ in range(char_height)]

    def clear(self) -> None:
        for r in range(self.char_height):
            for c in range(self.char_width):
                self.grid[r][c] = 0

    def set_pixel(self, x: int, y: int) -> None:
        """Set a single sub-pixel dot at (x, y)."""
        if 0 <= x < self.pixel_width and 0 <= y < self.pixel_height:
            cx = x // 2
            cy = y // 4
            sub_x = x % 2
            sub_y = y % 4
            self.grid[cy][cx] |= self.DOT_MAP[sub_y][sub_x]

    def draw_line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Draw line between (x0, y0) and (x1, y1) using Bresenham's algorithm."""
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        while True:
            self.set_pixel(x0, y0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def plot_sparkline(self, data: List[float], min_val: Optional[float] = None, max_val: Optional[float] = None) -> None:
        """Plot continuous timeseries data across the width of the canvas."""
        if not data:
            return
        min_v = min_val if min_val is not None else min(data)
        max_v = max_val if max_val is not None else max(data)
        span = max_v - min_v if max_v > min_v else 1.0

        n = len(data)
        pts: List[Tuple[int, int]] = []
        for i, val in enumerate(data):
            x = int(i * (self.pixel_width - 1) / (n - 1)) if n > 1 else 0
            norm = (val - min_v) / span
            norm = max(0.0, min(1.0, norm))
            y = int((1.0 - norm) * (self.pixel_height - 1))
            pts.append((x, y))

        for i in range(len(pts) - 1):
            self.draw_line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])

    def render(self) -> str:
        """Render grid into multi-line string of Braille characters (U+2800..U+28FF)."""
        lines = []
        for r in range(self.char_height):
            line_chars = []
            for c in range(self.char_width):
                dots = self.grid[r][c]
                line_chars.append(chr(0x2800 + dots))
            lines.append("".join(line_chars))
        return "\n".join(lines)


# =====================================================================
# Pipeline Stage HUD Visualizer
# =====================================================================

class PipelineHUD:
    """Visualizes the 5 active stages of the in-order RISC pipeline."""

    @staticmethod
    def render(core: PipelinedCore, status: Optional[PipelineStatus] = None) -> str:
        lines = []
        sep = f"{CYAN}+" + "-" * 76 + f"+{RESET}"
        title = f"{BOLD}{CYAN}| 5-STAGE IN-ORDER PIPELINE HUD (CYCLE {core.cycle_count:06d}) {' ' * 36}|{RESET}"

        lines.append(sep)
        lines.append(title)
        lines.append(sep)

        # 1. Fetch Stage
        if_str = f"PC: 0x{core.pc:08X}" if not core.halted else "HALTED"
        lines.append(f"| {BOLD}[IF] FETCH  {RESET} : {if_str:<62} |")

        # 2. Decode Stage
        if_id = core.if_id
        if if_id.valid and if_id.inst is not None:
            d_str = f"0x{if_id.pc:08X} [{if_id.raw:08X}] -> {if_id.inst.mnemonic:<7} {Disassembler.disassemble(if_id.inst)}"
        else:
            d_str = f"{DIM}(bubble / empty){RESET}"
        lines.append(f"| {BOLD}[ID] DECODE {RESET} : {d_str:<70} |")

        # 3. Execute Stage
        id_ex = core.id_ex
        if id_ex.valid and id_ex.inst:
            fwd_info = ""
            if status and status.forward_a > 0:
                fwd_info += f" [FWD_A:{'EX' if status.forward_a == 2 else 'MEM'}]"
            if status and status.forward_b > 0:
                fwd_info += f" [FWD_B:{'EX' if status.forward_b == 2 else 'MEM'}]"
            ex_str = f"0x{id_ex.pc:08X} {id_ex.inst.mnemonic:<6} op_a=0x{id_ex.val_rs1:X} op_b=0x{id_ex.val_rs2:X}{fwd_info}"
        else:
            ex_str = f"{DIM}(bubble / stall){RESET}"
        lines.append(f"| {BOLD}[EX] EXEC   {RESET} : {ex_str:<70} |")

        # 4. Memory Stage
        ex_mem = core.ex_mem
        if ex_mem.valid and ex_mem.inst:
            mem_act = "NOP"
            if ex_mem.inst.is_load:
                mem_act = f"LOAD [0x{ex_mem.alu_result:08X}]"
            elif ex_mem.inst.is_store:
                mem_act = f"STORE [0x{ex_mem.alu_result:08X}] <= 0x{ex_mem.val_rs2:X}"
            mem_str = f"0x{ex_mem.pc:08X} {ex_mem.inst.mnemonic:<6} {mem_act}"
        else:
            mem_str = f"{DIM}(bubble / empty){RESET}"
        lines.append(f"| {BOLD}[MEM] MEMORY{RESET} : {mem_str:<70} |")

        # 5. Writeback Stage
        mem_wb = core.mem_wb
        if mem_wb.valid and mem_wb.inst:
            wb_str = f"0x{mem_wb.pc:08X} {mem_wb.inst.mnemonic:<6}"
            if mem_wb.inst.writes_rd and mem_wb.inst.rd != 0:
                wb_val = mem_wb.mem_data if mem_wb.inst.is_load else mem_wb.alu_result
                reg_name = ABI_REG_NAMES[mem_wb.inst.rd] if 0 <= mem_wb.inst.rd < len(ABI_REG_NAMES) else f"x{mem_wb.inst.rd}"
                wb_str += f" -> {reg_name} (x{mem_wb.inst.rd}) <= 0x{wb_val:X} ({wb_val})"
        else:
            wb_str = f"{DIM}(bubble / empty){RESET}"
        lines.append(f"| {BOLD}[WB] WRITE  {RESET} : {wb_str:<70} |")

        # Hazard and Control Indicators
        flags = []
        if status:
            if status.stalled:
                flags.append(f"{YELLOW}[LOAD-USE STALL]{RESET}")
            if status.flushed:
                flags.append(f"{RED}[BRANCH FLUSH]{RESET}")
            if status.forward_a > 0 or status.forward_b > 0:
                flags.append(f"{GREEN}[FORWARDING ACTIVE]{RESET}")
        if core.halted:
            flags.append(f"{MAGENTA}[CPU HALTED]{RESET}")

        flag_str = " ".join(flags) if flags else f"{DIM}Optimal Flow (No Hazards){RESET}"
        lines.append(sep)
        lines.append(f"| Status: {flag_str:<74} |")
        lines.append(sep)

        return "\n".join(lines)


# =====================================================================
# Register File Inspector (32 Integer Registers Grid)
# =====================================================================

class RegisterInspector:
    """Formatted 4-column register inspection grid."""

    @staticmethod
    def render(core_reg_file: Any) -> str:
        lines = []
        sep = f"{YELLOW}+" + "-" * 76 + f"+{RESET}"
        lines.append(sep)
        lines.append(f"{BOLD}{YELLOW}| 64-BIT GENERAL PURPOSE INTEGER REGISTERS (RV64I) {' ' * 24}|{RESET}")
        lines.append(sep)

        # Render 8 rows x 4 columns
        for row in range(8):
            row_items = []
            for col in range(4):
                reg_idx = row + col * 8
                name = ABI_REG_NAMES[reg_idx] if 0 <= reg_idx < len(ABI_REG_NAMES) else f"x{reg_idx}"
                val = core_reg_file.read_x(reg_idx)
                # Show register name and lower 32-bits/64-bits
                item = f"{name:>4} (x{reg_idx:02d}): 0x{val:08X}"
                row_items.append(item)
            line = "| " + " | ".join(row_items) + " |"
            lines.append(line)

        lines.append(sep)
        return "\n".join(lines)


# =====================================================================
# Cache & Memory Telemetry HUD
# =====================================================================

class CacheTelemetryHUD:
    """Telemetry HUD for L1I, L1D, L2 Caches, MMU TLBs, and Branch Predictor."""

    @staticmethod
    def render(cpu: CPU) -> str:
        lines = []
        sep = f"{GREEN}+" + "-" * 76 + f"+{RESET}"
        lines.append(sep)
        lines.append(f"{BOLD}{GREEN}| MEMORY HIERARCHY, MESI COHERENCE & BPU TELEMETRY {' ' * 24}|{RESET}")
        lines.append(sep)

        # Cache Row
        l1i = cpu.cache.l1i
        l1d = cpu.cache.l1d
        l2 = cpu.cache.l2

        lines.append(
            f"| L1-I Cache: {l1i.hits:5d} Hits / {l1i.misses:4d} Misses | Hit Rate: {l1i.hit_rate * 100:6.2f}% | Latency: 1 cycle  |"
        )
        lines.append(
            f"| L1-D Cache: {l1d.hits:5d} Hits / {l1d.misses:4d} Misses | Hit Rate: {l1d.hit_rate * 100:6.2f}% | Latency: 2 cycles |"
        )
        lines.append(
            f"| L2 Unified: {l2.hits:5d} Hits / {l2.misses:4d} Misses | Hit Rate: {l2.hit_rate * 100:6.2f}% | Latency: 8 cycles |"
        )
        lines.append(sep)

        # MMU & Branch Row
        bpu = cpu.bpu
        lines.append(
            f"| Branch Unit: {bpu.correct_predictions:5d} Hits / {bpu.mispredictions:4d} Misses | Accuracy: {bpu.accuracy * 100:6.2f}% | BTB Hits: {bpu.btb_hits:4d} |"
        )
        lines.append(
            f"| SV39 MMU   : ITLB Hits: {cpu.mmu.itlb_hits:4d} | DTLB Hits: {cpu.mmu.dtlb_hits:4d} | Page Faults: 0        |"
        )
        lines.append(sep)

        return "\n".join(lines)


# =====================================================================
# System Workbench Dashboard
# =====================================================================

class SystemWorkbenchDashboard:
    """Consolidated terminal dashboard rendering pipeline, registers, cache, and Braille sparkline."""

    def __init__(self, cpu: CPU) -> None:
        self.cpu = cpu
        self.ipc_history: List[float] = []
        self.canvas = BrailleCanvas(char_width=38, char_height=4)

    def update_history(self) -> None:
        ipc = self.cpu.instructions_executed / self.cpu.total_cycles if self.cpu.total_cycles > 0 else 1.0
        self.ipc_history.append(ipc)
        if len(self.ipc_history) > 76:
            self.ipc_history.pop(0)

    def render(self, status: Optional[PipelineStatus] = None) -> str:
        self.update_history()
        parts = []

        # 1. Header Banner
        mode_str = self.cpu.mode.name
        hdr = (
            f"{BOLD}{MAGENTA}"
            f"==============================================================================\n"
            f"  SILICONRISC RV64GC HARDWARE WORKBENCH : [{mode_str} EXECUTION]\n"
            f"  Cycles: {self.cpu.total_cycles:06d} | Insts: {self.cpu.instructions_executed:06d} | IPC: {self.ipc_history[-1]:.3f} | CPI: {(1.0 / self.ipc_history[-1]) if self.ipc_history[-1] > 0 else 0:.3f}\n"
            f"=============================================================================={RESET}"
        )
        parts.append(hdr)

        # 2. Pipeline HUD
        if self.cpu.mode.name == "PIPELINED":
            parts.append(PipelineHUD.render(self.cpu.pipeline, status))

        # 3. Register Inspector
        reg_file = self.cpu.pipeline.reg_file if self.cpu.mode.name == "PIPELINED" else self.cpu.reg_file
        parts.append(RegisterInspector.render(reg_file))

        # 4. Cache & Memory Telemetry
        parts.append(CacheTelemetryHUD.render(self.cpu))

        # 5. Braille IPC Waveform
        self.canvas.clear()
        self.canvas.plot_sparkline(self.ipc_history, min_val=0.0, max_val=2.0)
        spark_title = f"{BOLD}{BLUE}--- REAL-TIME IPC WAVEFORM (BRAILLE SUB-PIXEL 0.0 TO 2.0 IPC) ---{RESET}"
        parts.append(spark_title)
        parts.append(self.canvas.render())

        return "\n".join(parts)
