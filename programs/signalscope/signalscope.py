#!/usr/bin/env python3
"""SignalScope - Standalone Desktop DSP Virtual Oscilloscope & Audio Synthesizer.

Pure Python standard library (Tkinter) application featuring:
- Time-Domain Dual-Trace Oscilloscope with calibrated graticule and edge-triggering.
- Real-time FFT Frequency Spectrum Analyzer with decibel magnitude and harmonic peaks.
- Lissajous Phase XY Trajectory mode for phase-difference and frequency-ratio analysis.
- Multi-waveform synthesizer with FM synthesis, AM ring modulation, and resonant biquad filtering.
- 16-bit PCM WAV audio exporting.
"""

from __future__ import annotations
import math
import sys
import time
import os
from typing import List, Tuple, Optional, Dict, Any

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except ImportError:
    print("Error: Tkinter is required to run SignalScope Desktop GUI.")
    sys.exit(1)

try:
    from .dsp import SignalEngine, ChannelParams, BiquadFilter
except ImportError:
    from dsp import SignalEngine, ChannelParams, BiquadFilter


class SignalScopeApp:
    """Desktop Virtual Oscilloscope and Audio Synthesizer Studio."""

    def __init__(self, root: Optional[tk.Tk] = None):
        self.root = root if root is not None else tk.Tk()
        self.root.title("SignalScope - Virtual Oscilloscope & Audio Synthesizer")
        self.root.geometry("1340x880")
        self.root.minsize(1100, 720)
        self.root.configure(bg="#0B0F19")

        # DSP Signal Engine
        self.engine = SignalEngine(sample_rate=44100)

        # Oscilloscope State
        self.running = True
        self.display_mode = "Oscilloscope"  # "Oscilloscope", "Spectrum", "Lissajous"
        self.time_div = 1.0  # ms per horizontal division (10 divisions total)
        self.volts_div_ch1 = 0.5  # Volts per division (8 divisions total)
        self.volts_div_ch2 = 0.5
        self.persistency = False

        # Simulation clock
        self.sim_time = 0.0
        self.fps = 60.0
        self.last_frame_time = time.time()
        self._loop_running = True

        # Pre-allocate buffer sizes
        self.scope_samples = 1024

        # Apply dark theme and layout
        self._apply_theme()
        self._build_layout()

        # Window protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(16, self._update_loop)

    def _apply_theme(self) -> None:
        """Configure clean dark-mode widget styling."""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TFrame", background="#111827")
        style.configure("TLabel", background="#111827", foreground="#F9FAFB", font=("Helvetica", 10))
        style.configure("Header.TLabel", background="#111827", foreground="#06B6D4", font=("Helvetica", 10, "bold"))
        style.configure("Muted.TLabel", background="#111827", foreground="#9CA3AF", font=("Helvetica", 9))
        style.configure("TButton", background="#1F293D", foreground="#F9FAFB", font=("Helvetica", 9, "bold"), padding=5)
        style.map("TButton", background=[("active", "#06B6D4"), ("pressed", "#0891B2")], foreground=[("active", "#0B0F19")])
        style.configure("TCombobox", fieldbackground="#1F293D", background="#1F293D", foreground="#F9FAFB")
        style.configure("Horizontal.TScale", background="#111827", troughcolor="#1F293D")

    def _build_layout(self) -> None:
        """Construct top control ribbon, phosphor CRT screen, and parameter sidebar."""
        # Top Ribbon
        top_bar = tk.Frame(self.root, bg="#111827", height=50, highlightthickness=1, highlightbackground="#1F293D")
        top_bar.pack(side=tk.TOP, fill=tk.X)

        # Brand Badge
        brand_frame = tk.Frame(top_bar, bg="#111827")
        brand_frame.pack(side=tk.LEFT, padx=16, pady=6)

        logo_lbl = tk.Label(brand_frame, text="SIGNALSCOPE", bg="#06B6D4", fg="#0B0F19", font=("Helvetica", 10, "bold"), padx=8, pady=2)
        logo_lbl.pack(side=tk.LEFT, padx=(0, 10))

        sub_lbl = tk.Label(brand_frame, text="Dual-Trace DSP Oscilloscope & Synthesizer", bg="#111827", fg="#9CA3AF", font=("Helvetica", 10))
        sub_lbl.pack(side=tk.LEFT)

        # Ribbon Mode Buttons
        mode_frame = tk.Frame(top_bar, bg="#111827")
        mode_frame.pack(side=tk.RIGHT, padx=16)

        self.btn_mode_scope = tk.Button(mode_frame, text="Time Oscilloscope", bg="#06B6D4", fg="#0B0F19",
                                        font=("Helvetica", 9, "bold"), relief="flat", padx=10, pady=4,
                                        command=lambda: self.set_display_mode("Oscilloscope"))
        self.btn_mode_scope.pack(side=tk.LEFT, padx=3)

        self.btn_mode_spec = tk.Button(mode_frame, text="FFT Spectrum", bg="#1F293D", fg="#F9FAFB",
                                       font=("Helvetica", 9, "bold"), relief="flat", padx=10, pady=4,
                                       command=lambda: self.set_display_mode("Spectrum"))
        self.btn_mode_spec.pack(side=tk.LEFT, padx=3)

        self.btn_mode_liss = tk.Button(mode_frame, text="Lissajous (X-Y)", bg="#1F293D", fg="#F9FAFB",
                                       font=("Helvetica", 9, "bold"), relief="flat", padx=10, pady=4,
                                       command=lambda: self.set_display_mode("Lissajous"))
        self.btn_mode_liss.pack(side=tk.LEFT, padx=3)

        self.btn_run = tk.Button(mode_frame, text="❚❚ Hold", bg="#1F293D", fg="#F9FAFB",
                                 font=("Helvetica", 9, "bold"), relief="flat", padx=8, pady=4,
                                 command=self.toggle_run)
        self.btn_run.pack(side=tk.LEFT, padx=(8, 3))

        btn_export = tk.Button(mode_frame, text="💾 Save WAV", bg="#10B981", fg="#0B0F19",
                               font=("Helvetica", 9, "bold"), relief="flat", padx=8, pady=4,
                               command=self.export_audio_wav)
        btn_export.pack(side=tk.LEFT, padx=3)

        # Main Split Workspace
        workspace = tk.Frame(self.root, bg="#080C14")
        workspace.pack(fill=tk.BOTH, expand=True)

        # Left: CRT Canvas
        self.canvas_frame = tk.Frame(workspace, bg="#05080E", highlightthickness=1, highlightbackground="#1F293D")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#05080E", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Overlay HUD on CRT
        self.crt_hud = tk.Label(self.canvas, text="CH1: 0.50 V/div • CH2: 0.50 V/div • TIME: 1.0 ms/div • TRIG: AUTO 0.00V",
                                bg="#0B132B", fg="#06B6D4", font=("Menlo", 9), padx=8, pady=4,
                                highlightthickness=1, highlightbackground="#1F293D")
        self.crt_hud.place(x=16, y=16)

        # Right: Sidebar Controls
        sidebar = tk.Frame(workspace, bg="#111827", width=380, highlightthickness=1, highlightbackground="#1F293D")
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Scrollable sidebar wrapper
        side_canvas = tk.Canvas(sidebar, bg="#111827", highlightthickness=0)
        side_scroll = ttk.Scrollbar(sidebar, orient=tk.VERTICAL, command=side_canvas.yview)
        self.side_content = tk.Frame(side_canvas, bg="#111827")

        self.side_content.bind(
            "<Configure>",
            lambda e: side_canvas.configure(scrollregion=side_canvas.bbox("all"))
        )
        side_canvas.create_window((0, 0), window=self.side_content, anchor="nw", width=360)
        side_canvas.configure(yscrollcommand=side_scroll.set)

        side_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        side_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._build_sidebar_sections()

    def _build_sidebar_sections(self) -> None:
        """Create Channel 1, Channel 2, Modulation, Filter, and Scope Timebase decks."""
        p = self.side_content

        # --- SECTION 1: CHANNEL 1 (Primary Trace / X-Axis) ---
        sec1 = tk.Frame(p, bg="#161F30", highlightthickness=1, highlightbackground="#1F293D", padx=12, pady=10)
        sec1.pack(fill=tk.X, padx=12, pady=(10, 5))

        tk.Label(sec1, text="CHANNEL 1 (CH1 - CYAN)", bg="#161F30", fg="#06B6D4", font=("Helvetica", 10, "bold")).pack(anchor="w")

        # Waveform selector
        tk.Label(sec1, text="Waveform Shape:", bg="#161F30", fg="#9CA3AF", font=("Helvetica", 8)).pack(anchor="w", pady=(4, 0))
        self.cb_wave1 = ttk.Combobox(sec1, values=["Sine", "Square", "Triangle", "Sawtooth", "Noise"], state="readonly")
        self.cb_wave1.set("Sine")
        self.cb_wave1.pack(fill=tk.X, pady=(2, 6))
        self.cb_wave1.bind("<<ComboboxSelected>>", lambda e: setattr(self.engine.ch1, "waveform", self.cb_wave1.get()))

        # Frequency Slider
        tk.Label(sec1, text="Frequency (Hz):", bg="#161F30", fg="#9CA3AF", font=("Helvetica", 8)).pack(anchor="w")
        f1_bar = tk.Frame(sec1, bg="#161F30")
        f1_bar.pack(fill=tk.X)
        self.scale_f1 = ttk.Scale(f1_bar, from_=20.0, to=2000.0, value=440.0, orient=tk.HORIZONTAL, command=self._on_f1_change)
        self.scale_f1.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_f1 = tk.Label(f1_bar, text="440.0 Hz", bg="#161F30", fg="#06B6D4", font=("Menlo", 8, "bold"), width=8)
        self.lbl_f1.pack(side=tk.RIGHT)

        # Amplitude Slider
        tk.Label(sec1, text="Amplitude (V):", bg="#161F30", fg="#9CA3AF", font=("Helvetica", 8)).pack(anchor="w", pady=(4, 0))
        a1_bar = tk.Frame(sec1, bg="#161F30")
        a1_bar.pack(fill=tk.X)
        self.scale_a1 = ttk.Scale(a1_bar, from_=0.0, to=1.5, value=1.0, orient=tk.HORIZONTAL, command=self._on_a1_change)
        self.scale_a1.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_a1 = tk.Label(a1_bar, text="1.00 V", bg="#161F30", fg="#06B6D4", font=("Menlo", 8, "bold"), width=8)
        self.lbl_a1.pack(side=tk.RIGHT)

        # Phase Offset
        tk.Label(sec1, text="Phase Offset (deg):", bg="#161F30", fg="#9CA3AF", font=("Helvetica", 8)).pack(anchor="w", pady=(4, 0))
        ph1_bar = tk.Frame(sec1, bg="#161F30")
        ph1_bar.pack(fill=tk.X)
        self.scale_ph1 = ttk.Scale(ph1_bar, from_=0.0, to=360.0, value=0.0, orient=tk.HORIZONTAL, command=self._on_ph1_change)
        self.scale_ph1.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_ph1 = tk.Label(ph1_bar, text="0.0°", bg="#161F30", fg="#06B6D4", font=("Menlo", 8, "bold"), width=8)
        self.lbl_ph1.pack(side=tk.RIGHT)

        # --- SECTION 2: CHANNEL 2 (Secondary Trace / Y-Axis) ---
        sec2 = tk.Frame(p, bg="#161F30", highlightthickness=1, highlightbackground="#1F293D", padx=12, pady=10)
        sec2.pack(fill=tk.X, padx=12, pady=5)

        tk.Label(sec2, text="CHANNEL 2 (CH2 - AMBER)", bg="#161F30", fg="#F59E0B", font=("Helvetica", 10, "bold")).pack(anchor="w")

        # Waveform selector
        tk.Label(sec2, text="Waveform Shape:", bg="#161F30", fg="#9CA3AF", font=("Helvetica", 8)).pack(anchor="w", pady=(4, 0))
        self.cb_wave2 = ttk.Combobox(sec2, values=["Sine", "Square", "Triangle", "Sawtooth", "Noise"], state="readonly")
        self.cb_wave2.set("Sine")
        self.cb_wave2.pack(fill=tk.X, pady=(2, 6))
        self.cb_wave2.bind("<<ComboboxSelected>>", lambda e: setattr(self.engine.ch2, "waveform", self.cb_wave2.get()))

        # Frequency Slider
        tk.Label(sec2, text="Frequency (Hz):", bg="#161F30", fg="#9CA3AF", font=("Helvetica", 8)).pack(anchor="w")
        f2_bar = tk.Frame(sec2, bg="#161F30")
        f2_bar.pack(fill=tk.X)
        self.scale_f2 = ttk.Scale(f2_bar, from_=20.0, to=2000.0, value=880.0, orient=tk.HORIZONTAL, command=self._on_f2_change)
        self.scale_f2.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_f2 = tk.Label(f2_bar, text="880.0 Hz", bg="#161F30", fg="#F59E0B", font=("Menlo", 8, "bold"), width=8)
        self.lbl_f2.pack(side=tk.RIGHT)

        # Amplitude Slider
        tk.Label(sec2, text="Amplitude (V):", bg="#161F30", fg="#9CA3AF", font=("Helvetica", 8)).pack(anchor="w", pady=(4, 0))
        a2_bar = tk.Frame(sec2, bg="#161F30")
        a2_bar.pack(fill=tk.X)
        self.scale_a2 = ttk.Scale(a2_bar, from_=0.0, to=1.5, value=0.5, orient=tk.HORIZONTAL, command=self._on_a2_change)
        self.scale_a2.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_a2 = tk.Label(a2_bar, text="0.50 V", bg="#161F30", fg="#F59E0B", font=("Menlo", 8, "bold"), width=8)
        self.lbl_a2.pack(side=tk.RIGHT)

        # Phase Offset
        tk.Label(sec2, text="Phase Offset (deg):", bg="#161F30", fg="#9CA3AF", font=("Helvetica", 8)).pack(anchor="w", pady=(4, 0))
        ph2_bar = tk.Frame(sec2, bg="#161F30")
        ph2_bar.pack(fill=tk.X)
        self.scale_ph2 = ttk.Scale(ph2_bar, from_=0.0, to=360.0, value=90.0, orient=tk.HORIZONTAL, command=self._on_ph2_change)
        self.scale_ph2.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_ph2 = tk.Label(ph2_bar, text="90.0°", bg="#161F30", fg="#F59E0B", font=("Menlo", 8, "bold"), width=8)
        self.lbl_ph2.pack(side=tk.RIGHT)

        # --- SECTION 3: MODULATION & BIQUAD FILTER DECK ---
        sec3 = tk.Frame(p, bg="#161F30", highlightthickness=1, highlightbackground="#1F293D", padx=12, pady=10)
        sec3.pack(fill=tk.X, padx=12, pady=5)

        tk.Label(sec3, text="SYNTHESIS & MODULATION", bg="#161F30", fg="#8B5CF6", font=("Helvetica", 10, "bold")).pack(anchor="w")

        # FM Modulation
        self.var_fm = tk.BooleanVar(value=False)
        tk.Checkbutton(sec3, text="FM: Modulate CH1 Freq by CH2", variable=self.var_fm, bg="#161F30", fg="#F9FAFB",
                       selectcolor="#0B0F19", activebackground="#161F30", font=("Helvetica", 8),
                       command=self._on_toggle_fm).pack(anchor="w", pady=(4, 0))

        # AM Modulation
        self.var_am = tk.BooleanVar(value=False)
        tk.Checkbutton(sec3, text="AM: Ring Modulate Amplitude", variable=self.var_am, bg="#161F30", fg="#F9FAFB",
                       selectcolor="#0B0F19", activebackground="#161F30", font=("Helvetica", 8),
                       command=self._on_toggle_am).pack(anchor="w", pady=2)

        # Biquad Filter Mode
        tk.Label(sec3, text="Biquad Filter:", bg="#161F30", fg="#9CA3AF", font=("Helvetica", 8)).pack(anchor="w", pady=(4, 0))
        self.cb_filter = ttk.Combobox(sec3, values=["Bypass", "Lowpass", "Highpass", "Bandpass"], state="readonly")
        self.cb_filter.set("Bypass")
        self.cb_filter.pack(fill=tk.X, pady=(2, 4))
        self.cb_filter.bind("<<ComboboxSelected>>", self._on_filter_type_change)

        # Filter Cutoff
        tk.Label(sec3, text="Cutoff Frequency (Hz):", bg="#161F30", fg="#9CA3AF", font=("Helvetica", 8)).pack(anchor="w")
        fc_bar = tk.Frame(sec3, bg="#161F30")
        fc_bar.pack(fill=tk.X)
        self.scale_fc = ttk.Scale(fc_bar, from_=100.0, to=8000.0, value=3000.0, orient=tk.HORIZONTAL, command=self._on_cutoff_change)
        self.scale_fc.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_fc = tk.Label(fc_bar, text="3000 Hz", bg="#161F30", fg="#8B5CF6", font=("Menlo", 8, "bold"), width=8)
        self.lbl_fc.pack(side=tk.RIGHT)

        # --- SECTION 4: OSCILLOSCOPE TIMEBASE & TRIGGER ---
        sec4 = tk.Frame(p, bg="#161F30", highlightthickness=1, highlightbackground="#1F293D", padx=12, pady=10)
        sec4.pack(fill=tk.X, padx=12, pady=5)

        tk.Label(sec4, text="TIMEBASE & TRIGGER", bg="#161F30", fg="#10B981", font=("Helvetica", 10, "bold")).pack(anchor="w")

        # Time/div
        tk.Label(sec4, text="Time / Div (ms):", bg="#161F30", fg="#9CA3AF", font=("Helvetica", 8)).pack(anchor="w", pady=(4, 0))
        tdiv_bar = tk.Frame(sec4, bg="#161F30")
        tdiv_bar.pack(fill=tk.X)
        self.scale_tdiv = ttk.Scale(tdiv_bar, from_=0.2, to=10.0, value=1.0, orient=tk.HORIZONTAL, command=self._on_tdiv_change)
        self.scale_tdiv.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_tdiv = tk.Label(tdiv_bar, text="1.0 ms", bg="#161F30", fg="#10B981", font=("Menlo", 8, "bold"), width=8)
        self.lbl_tdiv.pack(side=tk.RIGHT)

        # Trigger controls
        trig_bar = tk.Frame(sec4, bg="#161F30")
        trig_bar.pack(fill=tk.X, pady=(4, 0))

        tk.Label(trig_bar, text="Trigger Mode:", bg="#161F30", fg="#9CA3AF", font=("Helvetica", 8)).pack(side=tk.LEFT)
        self.cb_trig = ttk.Combobox(trig_bar, values=["Auto", "Rising Slope", "Falling Slope"], state="readonly", width=14)
        self.cb_trig.set("Rising Slope")
        self.cb_trig.pack(side=tk.RIGHT)
        self.cb_trig.bind("<<ComboboxSelected>>", self._on_trig_mode_change)

        # --- SECTION 5: WAVEFORM METRICS HUD ---
        sec5 = tk.Frame(p, bg="#161F30", highlightthickness=1, highlightbackground="#1F293D", padx=12, pady=10)
        sec5.pack(fill=tk.X, padx=12, pady=(5, 12))

        tk.Label(sec5, text="TELEMETRY READOUT", bg="#161F30", fg="#F59E0B", font=("Helvetica", 10, "bold")).pack(anchor="w")

        self.lbl_vpp = tk.Label(sec5, text="CH1 Vpp: 2.00 V", bg="#161F30", fg="#06B6D4", font=("Menlo", 8), anchor="w")
        self.lbl_vpp.pack(fill=tk.X, pady=1)

        self.lbl_vrms = tk.Label(sec5, text="CH1 Vrms: 0.71 V", bg="#161F30", fg="#9CA3AF", font=("Menlo", 8), anchor="w")
        self.lbl_vrms.pack(fill=tk.X, pady=1)

        self.lbl_fpeak = tk.Label(sec5, text="Peak Freq: 440.0 Hz", bg="#161F30", fg="#F9FAFB", font=("Menlo", 8, "bold"), anchor="w")
        self.lbl_fpeak.pack(fill=tk.X, pady=1)

        self.lbl_thd = tk.Label(sec5, text="THD: < 0.1% (Pure Tone)", bg="#161F30", fg="#10B981", font=("Menlo", 8), anchor="w")
        self.lbl_thd.pack(fill=tk.X, pady=1)

    # --- Parameter Handlers ---

    def set_display_mode(self, mode: str) -> None:
        """Switch visual display mode."""
        self.display_mode = mode
        self.btn_mode_scope.configure(bg="#06B6D4" if mode == "Oscilloscope" else "#1F293D",
                                      fg="#0B0F19" if mode == "Oscilloscope" else "#F9FAFB")
        self.btn_mode_spec.configure(bg="#06B6D4" if mode == "Spectrum" else "#1F293D",
                                     fg="#0B0F19" if mode == "Spectrum" else "#F9FAFB")
        self.btn_mode_liss.configure(bg="#06B6D4" if mode == "Lissajous" else "#1F293D",
                                     fg="#0B0F19" if mode == "Lissajous" else "#F9FAFB")
        self.render_frame()

    def toggle_run(self) -> None:
        """Toggle run / hold waveform display."""
        self.running = not self.running
        self.btn_run.configure(text="▶ Run" if not self.running else "❚❚ Hold")

    def _on_f1_change(self, val: str) -> None:
        f = float(val)
        self.engine.ch1.frequency = f
        self.lbl_f1.configure(text=f"{f:5.1f} Hz")

    def _on_a1_change(self, val: str) -> None:
        a = float(val)
        self.engine.ch1.amplitude = a
        self.lbl_a1.configure(text=f"{a:4.2f} V")

    def _on_ph1_change(self, val: str) -> None:
        ph = float(val)
        self.engine.ch1.phase = ph
        self.lbl_ph1.configure(text=f"{ph:4.1f}°")

    def _on_f2_change(self, val: str) -> None:
        f = float(val)
        self.engine.ch2.frequency = f
        self.lbl_f2.configure(text=f"{f:5.1f} Hz")

    def _on_a2_change(self, val: str) -> None:
        a = float(val)
        self.engine.ch2.amplitude = a
        self.lbl_a2.configure(text=f"{a:4.2f} V")

    def _on_ph2_change(self, val: str) -> None:
        ph = float(val)
        self.engine.ch2.phase = ph
        self.lbl_ph2.configure(text=f"{ph:4.1f}°")

    def _on_toggle_fm(self) -> None:
        self.engine.fm_enabled = self.var_fm.get()

    def _on_toggle_am(self) -> None:
        self.engine.am_enabled = self.var_am.get()

    def _on_filter_type_change(self, event: Any) -> None:
        ftype = self.cb_filter.get()
        self.engine.filter.filter_type = ftype
        self.engine.filter.recompute()

    def _on_cutoff_change(self, val: str) -> None:
        fc = float(val)
        self.engine.filter.cutoff = fc
        self.engine.filter.recompute()
        self.lbl_fc.configure(text=f"{int(fc)} Hz")

    def _on_tdiv_change(self, val: str) -> None:
        self.time_div = float(val)
        self.lbl_tdiv.configure(text=f"{self.time_div:4.1f} ms")

    def _on_trig_mode_change(self, event: Any) -> None:
        mode = self.cb_trig.get()
        if mode == "Auto":
            self.engine.trigger_mode = "Auto"
        elif mode == "Rising Slope":
            self.engine.trigger_mode = "Normal"
            self.engine.trigger_slope = "Rising"
        else:
            self.engine.trigger_mode = "Normal"
            self.engine.trigger_slope = "Falling"

    def export_audio_wav(self) -> None:
        """Export synthesized waveform to 16-bit PCM WAV file."""
        path = os.path.join(os.path.dirname(__file__), "signalscope_recording.wav")
        self.engine.export_wav(path, duration_sec=2.5)
        messagebox.showinfo("Export Successful", f"Synthesized 16-bit WAV audio saved to:\n{path}")

    # --- Render Pipeline ---

    def _update_loop(self) -> None:
        """Periodic UI update and animation step."""
        if not self._loop_running:
            return

        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

        if self.running:
            self.sim_time += dt
            self.render_frame()

        self.root.after(16, self._update_loop)

    def render_frame(self) -> None:
        """Draw oscilloscope graticule, traces, spectrum, or Lissajous trajectories."""
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            return

        # 1. Calibrated CRT Oscilloscope Graticule
        self._draw_graticule(w, h)

        # 2. Render selected mode
        if self.display_mode == "Oscilloscope":
            self._draw_oscilloscope_traces(w, h)
        elif self.display_mode == "Spectrum":
            self._draw_spectrum(w, h)
        elif self.display_mode == "Lissajous":
            self._draw_lissajous(w, h)

    def _draw_graticule(self, w: int, h: int) -> None:
        """Draw calibrated 10x8 division CRT graticule grid with center axis crosshairs."""
        num_div_x = 10
        num_div_y = 8
        dx = w / num_div_x
        dy = h / num_div_y

        grid_color = "#0E1C2E"
        axis_color = "#1F3B5C"

        # Background grid lines
        for i in range(1, num_div_x):
            x = i * dx
            self.canvas.create_line(x, 0, x, h, fill=grid_color, width=1, dash=(2, 2))

        for j in range(1, num_div_y):
            y = j * dy
            self.canvas.create_line(0, y, w, y, fill=grid_color, width=1, dash=(2, 2))

        # Center Crosshair Axes
        mid_x = w / 2.0
        mid_y = h / 2.0
        self.canvas.create_line(mid_x, 0, mid_x, h, fill=axis_color, width=1)
        self.canvas.create_line(0, mid_y, w, mid_y, fill=axis_color, width=1)

        # Axis Subdivision Tick Marks (5 ticks per division)
        tick_len = 3
        for i in range(num_div_x * 5 + 1):
            x = (i / 5.0) * dx
            self.canvas.create_line(x, mid_y - tick_len, x, mid_y + tick_len, fill=axis_color, width=1)

        for j in range(num_div_y * 5 + 1):
            y = (j / 5.0) * dy
            self.canvas.create_line(mid_x - tick_len, y, mid_x + tick_len, y, fill=axis_color, width=1)

    def _draw_oscilloscope_traces(self, w: int, h: int) -> None:
        """Render dual-trace time-domain voltage waveforms."""
        mid_y = h / 2.0
        # Time displayed across screen = 10 divisions * time_div ms
        total_time_sec = (10.0 * self.time_div) * 0.001
        samples_needed = max(256, int(total_time_sec * self.engine.sample_rate))

        # Synthesize buffer
        buf1, buf2, mix = self.engine.generate_buffers(samples_needed + 200, t_start=self.sim_time)

        # Trigger synchronization
        start_idx = 0
        if self.engine.trigger_mode == "Normal":
            start_idx = self.engine.find_trigger_index(buf1, self.engine.trigger_level, self.engine.trigger_slope)

        # Scale Volts to Pixels: (8 divisions total = 4 div up, 4 div down)
        pixels_per_volt_1 = (h / 8.0) / self.volts_div_ch1
        pixels_per_volt_2 = (h / 8.0) / self.volts_div_ch2

        pts_ch1 = []
        pts_ch2 = []
        step_x = w / float(samples_needed - 1)

        for i in range(samples_needed):
            idx = start_idx + i
            if idx >= len(buf1):
                break
            x = i * step_x
            y1 = mid_y - buf1[idx] * pixels_per_volt_1
            y2 = mid_y - buf2[idx] * pixels_per_volt_2
            pts_ch1.append((x, y1))
            pts_ch2.append((x, y2))

        # Channel 2 trace (Amber phosphor)
        if len(pts_ch2) > 1:
            coords2 = [coord for pt in pts_ch2 for coord in pt]
            self.canvas.create_line(*coords2, fill="#F59E0B", width=2)

        # Channel 1 trace (Cyan phosphor)
        if len(pts_ch1) > 1:
            coords1 = [coord for pt in pts_ch1 for coord in pt]
            self.canvas.create_line(*coords1, fill="#06B6D4", width=2)

        # Update Telemetry Readouts
        metrics = self.engine.analyze_signal_metrics(buf1[:samples_needed])
        self.lbl_vpp.configure(text=f"CH1 Vpp: {metrics['vpp']:.2f} V")
        self.lbl_vrms.configure(text=f"CH1 Vrms: {metrics['vrms']:.2f} V")
        self.lbl_fpeak.configure(text=f"Est Freq: {metrics['freq']:.1f} Hz")

        # Top CRT HUD
        hud_text = f"CH1: {self.volts_div_ch1:.2f} V/div • CH2: {self.volts_div_ch2:.2f} V/div • TIME: {self.time_div:.1f} ms/div • TRIG: {self.engine.trigger_mode.upper()} {self.engine.trigger_level:.2f}V • FPS: {self.fps:.1f}"
        self.crt_hud.configure(text=hud_text)

    def _draw_spectrum(self, w: int, h: int) -> None:
        """Render FFT Frequency Spectrum Analyzer in decibels."""
        fft_size = 512
        buf1, buf2, mix = self.engine.generate_buffers(fft_size, t_start=self.sim_time)

        freqs, mags_db = self.engine.compute_spectrum(mix, window_type="Hanning")
        half_n = len(freqs)
        if half_n < 2:
            return

        # Frequency scale: 0 to Nyquist (22,050 Hz) or zoom to 5000 Hz
        max_f = 4000.0
        min_db = -90.0
        max_db = 0.0

        pts = []
        peak_f = 0.0
        peak_db = -120.0

        for i in range(half_n):
            f = freqs[i]
            if f > max_f:
                break
            db = mags_db[i]

            x = (f / max_f) * w
            norm_db = (db - min_db) / (max_db - min_db)
            y = h - norm_db * (h - 40) - 20
            pts.append((x, y))

            if db > peak_db:
                peak_db = db
                peak_f = f

        # Draw filled gradient bars/polygon
        if len(pts) > 2:
            poly_pts = [0, h]
            for p in pts:
                poly_pts.extend([p[0], p[1]])
            poly_pts.extend([w, h])
            self.canvas.create_polygon(*poly_pts, fill="#082F49", outline="")

            # Top spectral trace curve
            coords = [coord for pt in pts for coord in pt]
            self.canvas.create_line(*coords, fill="#38BDF8", width=2)

        # Peak frequency marker
        px = (peak_f / max_f) * w
        p_norm = (peak_db - min_db) / (max_db - min_db)
        py = h - p_norm * (h - 40) - 20
        self.canvas.create_oval(px - 4, py - 4, px + 4, py + 4, fill="#F59E0B", outline="#FFF")
        self.canvas.create_text(px, py - 12, text=f"{peak_f:.1f} Hz ({peak_db:.1f} dB)", fill="#F59E0B", font=("Menlo", 9, "bold"))

        # HUD
        hud_text = f"SPECTRUM ANALYZER • SPAN: 0 - 4000 Hz • RES: {freqs[1]:.1f} Hz • PEAK: {peak_f:.1f} Hz • FPS: {self.fps:.1f}"
        self.crt_hud.configure(text=hud_text)
        self.lbl_fpeak.configure(text=f"Peak Freq: {peak_f:.1f} Hz")

    def _draw_lissajous(self, w: int, h: int) -> None:
        """Render Lissajous XY parametric phase orbital trajectories (X = CH1, Y = CH2)."""
        mid_x = w / 2.0
        mid_y = h / 2.0
        scale_x = (w * 0.38) / max(0.1, self.engine.ch1.amplitude)
        scale_y = (h * 0.38) / max(0.1, self.engine.ch2.amplitude)

        # Number of samples for one complete Lissajous cycle
        samples = 600
        buf1, buf2, mix = self.engine.generate_buffers(samples, t_start=self.sim_time)

        pts = []
        for i in range(samples):
            x = mid_x + buf1[i] * scale_x
            y = mid_y - buf2[i] * scale_y
            pts.append((x, y))

        if len(pts) > 2:
            coords = [coord for pt in pts for coord in pt]
            # Phosphor cyan/purple glow
            self.canvas.create_line(*coords, fill="#8B5CF6", width=3)
            self.canvas.create_line(*coords, fill="#06B6D4", width=1.5)

        # Telemetry
        f_ratio = self.engine.ch2.frequency / max(1.0, self.engine.ch1.frequency)
        d_phase = (self.engine.ch2.phase - self.engine.ch1.phase) % 360.0
        hud_text = f"LISSAJOUS PHASE (X-Y) • FREQ RATIO: {f_ratio:.3f} • Δ PHASE: {d_phase:.1f}° • CH1: {self.engine.ch1.frequency:.1f}Hz • CH2: {self.engine.ch2.frequency:.1f}Hz"
        self.crt_hud.configure(text=hud_text)

    def on_close(self) -> None:
        """Clean shutdown handler."""
        self._loop_running = False
        self.root.destroy()


def main() -> None:
    """Application entry point."""
    root = tk.Tk()
    app = SignalScopeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
