from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from waterint.ui.core import comma_list, run_analysis, type_map_yaml, yaml_string


MODULES = {
    "density": {
        "title": "Density profile",
        "fields": [
            ("coord_range", "Coordinate range", "[-1.0, 30.0]"),
            ("coord_bins", "Coordinate bins", "310"),
            ("species", "Oxygen species", "OH-, H2O"),
            ("normalization", "Normalization", "number_density"),
            ("prefix", "Output prefix", "waterint_density"),
        ],
    },
    "oh-orientation": {
        "title": "OH orientation",
        "fields": [
            ("coord_range", "Coordinate range", "[-1.0, 8.0]"),
            ("coord_bins", "Coordinate bins", "180"),
            ("species", "Oxygen species", "OH-, H2O, H3O+"),
            ("angle_range", "Angle range", "[0.0, 180.0]"),
            ("angle_bins", "Angle bins", "180"),
            ("prefix", "Output prefix", "waterint_oh_orientation"),
        ],
    },
    "hbond": {
        "title": "H-bond topology",
        "fields": [
            ("species", "Oxygen species", "OH-, H2O, H3O+"),
            ("oo_cutoff", "O-O cutoff", "3.5"),
            ("angle_min", "D-H-A angle min", "150.0"),
            ("pbc", "PBC flags", "[true, true, false]"),
            ("prefix", "Output prefix", "waterint_hbond"),
        ],
    },
    "sfg": {
        "title": "SFG spectrum",
        "fields": [
            ("dt_ps", "dt_ps", "0.005"),
            ("lag_ps", "lag_ps", "0.995"),
            ("pbc", "PBC flags", "[true, true, false]"),
            ("window", "Window z1,z2", "[0.0, 8.0]"),
            ("nzeros", "FT zero padding", "2000"),
            ("prefix", "Output prefix", "waterint_sfg"),
        ],
    },
}


EXAMPLES = {
    "density": {
        "base_dir": "examples/density_xyz",
        "trajectory": "input/input.xyz",
        "format": "xyz",
        "system_cell": "[10.0, 10.0, 10.0]",
        "type_map": "",
        "fields": {
            "coord_range": "[0.0, 10.0]",
            "coord_bins": "20",
            "species": "O2-, OH-, H2O, H3O+",
            "normalization": "number_density",
            "prefix": "ui_density_oxygen_species",
        },
    },
    "oh-orientation": {
        "base_dir": "examples/angle_z_lammpstrj",
        "trajectory": "input/small_angle.lammpstrj",
        "format": "lammpstrj",
        "system_cell": "auto",
        "type_map": "1: H\n2: O\n3: Mg",
        "fields": {
            "coord_range": "[0.0, 5.0]",
            "coord_bins": "10",
            "species": "OH-, H2O, H3O+",
            "angle_range": "[0.0, 180.0]",
            "angle_bins": "18",
            "prefix": "ui_oh_orientation",
        },
    },
    "hbond": {
        "base_dir": "examples/hbond_lammpstrj",
        "trajectory": "input/small_hbond.lammpstrj",
        "format": "lammpstrj",
        "system_cell": "auto",
        "type_map": "1: H\n2: O",
        "fields": {
            "species": "H2O",
            "oo_cutoff": "3.5",
            "angle_min": "150.0",
            "pbc": "[false, false, false]",
            "prefix": "ui_hbond_topology",
        },
    },
    "sfg": {
        "base_dir": "examples/sfg_trajectory",
        "trajectory": "input/small_oh.lammpstrj",
        "format": "lammpstrj",
        "system_cell": "auto",
        "type_map": "1: H\n2: O",
        "fields": {
            "dt_ps": "0.001",
            "lag_ps": "0.003",
            "pbc": "[true, true, false]",
            "window": "",
            "nzeros": "32",
            "prefix": "ui_sfg_small_oh",
        },
    },
}


class ScrollFrame(ttk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.y_scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.x_scroll = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.content = ttk.Frame(self.canvas)
        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor=tk.NW)

        self.canvas.configure(yscrollcommand=self.y_scroll.set, xscrollcommand=self.x_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.y_scroll.grid(row=0, column=1, sticky="ns")
        self.x_scroll.grid(row=1, column=0, sticky="ew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.content.bind("<Configure>", self._content_configured)
        self.canvas.bind("<Configure>", self._canvas_configured)
        self.canvas.bind("<Enter>", lambda _event: self.canvas.bind_all("<MouseWheel>", self._mousewheel))
        self.canvas.bind("<Leave>", lambda _event: self.canvas.unbind_all("<MouseWheel>"))

    def _content_configured(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _canvas_configured(self, event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _mousewheel(self, event) -> None:
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class WaterIntDesktopApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("WaterInt")
        self.geometry("1440x860")
        self.minsize(1120, 700)
        self.field_vars: dict[str, tk.StringVar] = {}
        self.artifacts: list[dict[str, Any]] = []
        self.preview_image: tk.PhotoImage | None = None
        self.current_image_path: Path | None = None
        self._preview_resize_job: str | None = None
        self._build_vars()
        self._build_ui()
        self._render_module()

    def _build_vars(self) -> None:
        self.module_var = tk.StringVar(value="density")
        self.base_dir_var = tk.StringVar(value=str(Path.cwd()))
        self.trajectory_var = tk.StringVar(value="")
        self.format_var = tk.StringVar(value="npz")
        self.max_frames_var = tk.StringVar(value="all")
        self.system_cell_var = tk.StringVar(value="auto")
        self.type_map_var = tk.StringVar(value="1: H\n2: Mg\n3: O")
        self.unit_style_var = tk.StringVar(value="metal")
        self.out_length_var = tk.StringVar(value="A")
        self.out_mass_density_var = tk.StringVar(value="g/cm^3")
        self.out_frequency_var = tk.StringVar(value="cm^-1")
        self.status_var = tk.StringVar(value="Idle")

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.configure("Header.TLabel", font=("TkDefaultFont", 15, "bold"))
        style.configure("Section.TLabelframe.Label", font=("TkDefaultFont", 10, "bold"))

        root = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        root.pack(fill=tk.BOTH, expand=True)

        self.left = ttk.Frame(root, padding=(10, 10, 6, 10), width=330)
        self.center = ttk.Frame(root, padding=(6, 10), width=430)
        self.right = ttk.Frame(root, padding=(6, 10, 10, 10), width=680)
        root.add(self.left, weight=1)
        root.add(self.center, weight=1)
        root.add(self.right, weight=3)

        self._build_left()
        self._build_center()
        self._build_right()
        self.after(50, lambda: _set_sashes(root, [330, 760]))

    def _build_left(self) -> None:
        scroll = ScrollFrame(self.left)
        scroll.pack(fill=tk.BOTH, expand=True)
        content = scroll.content

        ttk.Label(content, text="WaterInt", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Label(content, text="Local molecular-interface analysis").pack(anchor=tk.W, pady=(0, 12))

        analysis = ttk.LabelFrame(content, text="Analysis", style="Section.TLabelframe", padding=8)
        analysis.pack(fill=tk.X, pady=(0, 10))
        self._labeled_combo(analysis, "Module", self.module_var, list(MODULES), self._on_module_change)
        ttk.Button(analysis, text="Load compact example", command=self._load_example).pack(fill=tk.X, pady=(2, 8))
        self._path_row(analysis, "Base directory", self.base_dir_var, self._browse_base_dir)
        self._path_row(analysis, "Trajectory", self.trajectory_var, self._browse_trajectory)
        self._labeled_combo(analysis, "Format", self.format_var, ["xyz", "lammpstrj", "npz"], self._refresh_yaml)
        self._labeled_entry(analysis, "Max frames", self.max_frames_var)
        self._labeled_entry(analysis, "System cell", self.system_cell_var)
        self._labeled_text(analysis, "Type map", self.type_map_var, height=4)

        units = ttk.LabelFrame(content, text="Units", style="Section.TLabelframe", padding=8)
        units.pack(fill=tk.X, pady=(0, 10))
        self._labeled_combo(units, "Style", self.unit_style_var, ["metal", "real", "si", "cgs", "electron", "micro", "nano"], self._refresh_yaml)
        self._labeled_combo(units, "Length output", self.out_length_var, ["A", "nm", "m", "cm", "um", "bohr"], self._refresh_yaml)
        self._labeled_combo(units, "Mass density", self.out_mass_density_var, ["g/cm^3", "kg/m^3", "amu/A^3"], self._refresh_yaml)
        self._labeled_combo(units, "Frequency", self.out_frequency_var, ["cm^-1", "THz"], self._refresh_yaml)

        self.module_frame = ttk.LabelFrame(content, text="Module parameters", style="Section.TLabelframe", padding=8)
        self.module_frame.pack(fill=tk.BOTH, expand=True)

    def _build_center(self) -> None:
        header = ttk.Frame(self.center)
        header.pack(fill=tk.X)
        self.module_title = ttk.Label(header, text="", style="Header.TLabel")
        self.module_title.pack(side=tk.LEFT, anchor=tk.W)
        ttk.Button(header, text="Regenerate config", command=self._refresh_yaml).pack(side=tk.RIGHT)

        self.yaml_text = tk.Text(self.center, wrap=tk.NONE, undo=True, font=("Menlo", 12))
        y_scroll = ttk.Scrollbar(self.center, orient=tk.VERTICAL, command=self.yaml_text.yview)
        x_scroll = ttk.Scrollbar(self.center, orient=tk.HORIZONTAL, command=self.yaml_text.xview)
        self.yaml_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.yaml_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=(10, 0))
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=(10, 0))
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)

    def _build_right(self) -> None:
        actions = ttk.Frame(self.right)
        actions.pack(fill=tk.X)
        ttk.Button(actions, text="Run analysis", command=self._run_clicked).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(actions, text="Open output folder", command=self._open_output_folder).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(self.right, textvariable=self.status_var).pack(anchor=tk.W, pady=(8, 12))

        results_pane = ttk.PanedWindow(self.right, orient=tk.VERTICAL)
        results_pane.pack(fill=tk.BOTH, expand=True)

        artifact_frame = ttk.LabelFrame(results_pane, text="Artifacts", style="Section.TLabelframe", padding=8)
        preview_frame = ttk.LabelFrame(results_pane, text="Preview", style="Section.TLabelframe", padding=8)
        results_pane.add(artifact_frame, weight=1)
        results_pane.add(preview_frame, weight=4)
        self.after(70, lambda: _set_sashes(results_pane, [180]))

        artifact_scroll_y = ttk.Scrollbar(artifact_frame, orient=tk.VERTICAL)
        artifact_scroll_x = ttk.Scrollbar(artifact_frame, orient=tk.HORIZONTAL)
        self.artifact_list = tk.Listbox(
            artifact_frame,
            height=7,
            yscrollcommand=artifact_scroll_y.set,
            xscrollcommand=artifact_scroll_x.set,
        )
        artifact_scroll_y.configure(command=self.artifact_list.yview)
        artifact_scroll_x.configure(command=self.artifact_list.xview)
        self.artifact_list.grid(row=0, column=0, sticky="nsew")
        artifact_scroll_y.grid(row=0, column=1, sticky="ns")
        artifact_scroll_x.grid(row=1, column=0, sticky="ew")
        artifact_frame.columnconfigure(0, weight=1)
        artifact_frame.rowconfigure(0, weight=1)
        self.artifact_list.bind("<<ListboxSelect>>", self._artifact_selected)
        ttk.Button(artifact_frame, text="Open selected artifact", command=self._open_selected_artifact).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        self.preview_canvas = tk.Canvas(preview_frame, background="#ffffff", highlightthickness=0)
        preview_y = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.preview_canvas.yview)
        preview_x = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self.preview_canvas.xview)
        self.preview_canvas.configure(yscrollcommand=preview_y.set, xscrollcommand=preview_x.set)
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        preview_y.grid(row=0, column=1, sticky="ns")
        preview_x.grid(row=1, column=0, sticky="ew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.preview_canvas.bind("<Configure>", self._preview_resized)
        self._show_preview_text("PNG previews appear here.")

    def _labeled_entry(self, parent: ttk.Frame, label: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=label).pack(anchor=tk.W)
        entry = ttk.Entry(parent, textvariable=var)
        entry.pack(fill=tk.X, pady=(2, 8))
        var.trace_add("write", lambda *_: self._refresh_yaml())

    def _labeled_combo(self, parent: ttk.Frame, label: str, var: tk.StringVar, values: list[str], callback) -> None:
        ttk.Label(parent, text=label).pack(anchor=tk.W)
        combo = ttk.Combobox(parent, textvariable=var, values=values, state="readonly")
        combo.pack(fill=tk.X, pady=(2, 8))
        combo.bind("<<ComboboxSelected>>", lambda _event: callback())

    def _labeled_text(self, parent: ttk.Frame, label: str, var: tk.StringVar, *, height: int) -> None:
        ttk.Label(parent, text=label).pack(anchor=tk.W)
        text = tk.Text(parent, height=height, wrap=tk.NONE)
        text.insert("1.0", var.get())
        text.pack(fill=tk.X, pady=(2, 8))

        def sync(_event=None) -> None:
            var.set(text.get("1.0", tk.END).strip())
            self._refresh_yaml()

        text.bind("<KeyRelease>", sync)
        var.trace_add("write", lambda *_: _replace_text(text, var.get()))

    def _path_row(self, parent: ttk.Frame, label: str, var: tk.StringVar, browse_command) -> None:
        ttk.Label(parent, text=label).pack(anchor=tk.W)
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(2, 8))
        ttk.Entry(row, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse", command=browse_command).pack(side=tk.LEFT, padx=(6, 0))
        var.trace_add("write", lambda *_: self._refresh_yaml())

    def _on_module_change(self) -> None:
        self._render_module()

    def _render_module(self) -> None:
        for child in self.module_frame.winfo_children():
            child.destroy()
        info = MODULES[self.module_var.get()]
        self.module_title.configure(text=info["title"])
        self.field_vars = {}
        for key, label, value in info["fields"]:
            var = tk.StringVar(value=value)
            self.field_vars[key] = var
            self._labeled_entry(self.module_frame, label, var)
        self._refresh_yaml()

    def _load_example(self) -> None:
        preset = EXAMPLES[self.module_var.get()]
        base = Path.cwd() / preset["base_dir"]
        self.base_dir_var.set(str(base))
        self.trajectory_var.set(preset["trajectory"])
        self.format_var.set(preset["format"])
        self.system_cell_var.set(preset["system_cell"])
        self.type_map_var.set(preset["type_map"])
        self.max_frames_var.set("all")
        for key, value in preset["fields"].items():
            if key in self.field_vars:
                self.field_vars[key].set(value)
        self._refresh_yaml()

    def _browse_base_dir(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.base_dir_var.get() or str(Path.cwd()))
        if selected:
            self.base_dir_var.set(selected)

    def _browse_trajectory(self) -> None:
        initial = self.base_dir_var.get() or str(Path.cwd())
        selected = filedialog.askopenfilename(initialdir=initial)
        if selected:
            path = Path(selected)
            base = Path(self.base_dir_var.get()).expanduser()
            try:
                self.trajectory_var.set(str(path.relative_to(base)))
            except ValueError:
                self.trajectory_var.set(str(path))
            suffix = path.suffix.lower()
            if suffix == ".xyz":
                self.format_var.set("xyz")
            elif suffix == ".npz":
                self.format_var.set("npz")
            elif suffix in {".lammpstrj", ".dump"}:
                self.format_var.set("lammpstrj")

    def _refresh_yaml(self) -> None:
        if not hasattr(self, "yaml_text"):
            return
        current = self.yaml_text.get("1.0", tk.END)
        generated = self._build_yaml() + "\n"
        if current != generated:
            self.yaml_text.delete("1.0", tk.END)
            self.yaml_text.insert("1.0", generated)

    def _build_yaml(self) -> str:
        module = self.module_var.get()
        output_dir = f"waterint_ui_output/{module}"
        lines: list[str] = []
        lines.append("input:")
        lines.append(f"  trajectory: {yaml_string(self.trajectory_var.get() or 'path/to/trajectory.npz')}")
        lines.append(f"  format: {self.format_var.get()}")
        type_map = type_map_yaml(self.type_map_var.get()) if self.type_map_var.get().strip() else []
        if type_map:
            lines.append("  type_map:")
            lines.extend(type_map)
        lines.append(f"  max_frames: {self.max_frames_var.get() or 'all'}")
        lines.append("  stride: 1")
        lines.append("")
        lines.append("system:")
        lines.append(f"  cell: {self.system_cell_var.get() or 'auto'}")
        lines.append("")
        lines.append("units:")
        lines.append(f"  style: {self.unit_style_var.get()}")
        lines.append("  output:")
        lines.append(f"    length: {self.out_length_var.get()}")
        lines.append(f"    mass_density: {self.out_mass_density_var.get()}")
        lines.append("    number_density: 1/A^3")
        lines.append("    time: ps")
        lines.append(f"    frequency: {self.out_frequency_var.get()}")
        lines.append("")

        if module != "sfg":
            lines.append("selection:")
            if module == "density":
                lines.append("  mode: oxygen_species")
            lines.append(f"  oxygen_species: [{', '.join(comma_list(self._field('species')))}]")
            lines.append("  oxygen_symbol: O")
            lines.append("  hydrogen_symbol: H")
            lines.append("  oh_cutoff: 1.25")
            lines.append("")

        if module in {"density", "oh-orientation"}:
            lines.append("coordinate:")
            lines.append("  mode: relative_to_slab")
            lines.append("  axis: z")
            lines.append("  reference:")
            lines.append("    type: slab_surface")
            lines.append("    species: [Mg]")
            lines.append("    surface: max")
            lines.append(f"  range: {self._field('coord_range')}")
            lines.append(f"  bins: {self._field('coord_bins')}")
            lines.append("")

        if module == "density":
            lines.append("normalization:")
            lines.append(f"  type: {self._field('normalization')}")
            if self._field("normalization") == "mass_density":
                lines.append("  masses_amu:")
                lines.append("    O2-: 15.999")
                lines.append("    OH-: 17.007")
                lines.append("    H2O: 18.015")
                lines.append("    H3O+: 19.023")
        elif module == "oh-orientation":
            lines.append("angle:")
            lines.append("  vector_mode: oh_bond")
            lines.append(f"  range: {self._field('angle_range')}")
            lines.append(f"  bins: {self._field('angle_bins')}")
            lines.append("  axis_sign: 1")
            lines.append("")
            lines.append("normalization:")
            lines.append("  type: counts_per_frame")
        elif module == "hbond":
            lines.append("hbond:")
            lines.append(f"  oo_cutoff: {self._field('oo_cutoff')}")
            lines.append(f"  dha_angle_min: {self._field('angle_min')}")
            lines.append("  h_acceptor_cutoff: null")
            lines.append("  max_acceptors_per_hydrogen: true")
            lines.append(f"  pbc: {self._field('pbc') or '[true, true, false]'}")
            lines.append("  classes_by_species:")
            lines.append("    OH-: [DAAA, DAA, DA, AAA, AA, A, other]")
            lines.append("    H2O: [DDAA, DDA, DAA, DA, AA, A, other]")
            lines.append("    H3O+: [DDDA, DDD, DDA, DD, DA, D, other]")
        elif module == "sfg":
            lines.append("sfg:")
            lines.append("  mode: trajectory")
            lines.append("  hydrogen_symbol: H")
            lines.append("  oxygen_symbol: O")
            lines.append(f"  dt_ps: {self._field('dt_ps')}")
            lines.append(f"  lag_ps: {self._field('lag_ps')}")
            lines.append(f"  pbc: {self._field('pbc') or '[true, true, false]'}")
            lines.append("  z_ref0: 0.0")
            window = comma_list(self._field("window"))
            if len(window) >= 2:
                lines.append("  window:")
                lines.append("    mode: 1")
                lines.append(f"    z1: {window[0]}")
                lines.append(f"    z2: {window[1]}")
                lines.append("    ramp: 0.5")
                lines.append("    flip: true")
            lines.append("  mu_mode: full")
            lines.append("  symmetrize: true")
            lines.append(f"  nzeros: {self._field('nzeros') or '2000'}")

        lines.append("")
        lines.append("output:")
        lines.append(f"  directory: {output_dir}")
        lines.append(f"  prefix: {self._field('prefix') or f'waterint_{module}'}")
        lines.append("  plot: true")
        return "\n".join(lines)

    def _field(self, key: str) -> str:
        var = self.field_vars.get(key)
        return var.get().strip() if var else ""

    def _run_clicked(self) -> None:
        module = self.module_var.get()
        base_dir = Path(self.base_dir_var.get()).expanduser()
        config_yaml = self.yaml_text.get("1.0", tk.END)
        self.status_var.set("Running analysis...")
        self.artifact_list.delete(0, tk.END)
        self.artifacts = []
        self.current_image_path = None
        self._show_preview_text("Running...")
        thread = threading.Thread(target=self._run_worker, args=(module, config_yaml, base_dir), daemon=True)
        thread.start()

    def _run_worker(self, module: str, config_yaml: str, base_dir: Path) -> None:
        try:
            result = run_analysis(module, config_yaml, base_dir)
        except Exception as exc:
            error = str(exc)
            self.after(0, lambda message=error: self._run_failed(message))
            return
        self.after(0, lambda: self._run_finished(result.artifacts))

    def _run_failed(self, message: str) -> None:
        self.status_var.set("Analysis failed")
        self.current_image_path = None
        self._show_preview_text(message)
        messagebox.showerror("WaterInt analysis failed", message)

    def _run_finished(self, artifacts: list[dict[str, Any]]) -> None:
        self.artifacts = artifacts
        self.artifact_list.delete(0, tk.END)
        for item in artifacts:
            self.artifact_list.insert(tk.END, f"{item['label']} - {item['name']}")
        self.status_var.set(f"Analysis complete: {len(artifacts)} artifacts")
        first_image = next((idx for idx, item in enumerate(artifacts) if item["kind"] == "image"), None)
        if first_image is not None:
            self.artifact_list.selection_set(first_image)
            self._show_artifact(self.artifacts[first_image])
        else:
            self.current_image_path = None
            self._show_preview_text("No PNG artifact was generated.")

    def _artifact_selected(self, _event=None) -> None:
        selected = self.artifact_list.curselection()
        if selected:
            self._show_artifact(self.artifacts[selected[0]])

    def _show_artifact(self, artifact: dict[str, Any]) -> None:
        if artifact["kind"] != "image":
            self.preview_image = None
            self.current_image_path = None
            self._show_preview_text(artifact["path"])
            return
        self.current_image_path = Path(artifact["path"])
        self._draw_current_image()

    def _draw_current_image(self) -> None:
        if self.current_image_path is None:
            return
        try:
            image = tk.PhotoImage(file=str(self.current_image_path))
        except tk.TclError:
            self.preview_image = None
            self._show_preview_text(f"Cannot preview image:\n{self.current_image_path}")
            return
        canvas_w = max(240, self.preview_canvas.winfo_width() - 24)
        canvas_h = max(240, self.preview_canvas.winfo_height() - 24)
        factor = max(1, int(max(image.width() / canvas_w, image.height() / canvas_h)))
        if factor > 1:
            image = image.subsample(factor, factor)
        self.preview_image = image
        self.preview_canvas.delete("all")
        x = max(12, (self.preview_canvas.winfo_width() - image.width()) // 2)
        y = max(12, (self.preview_canvas.winfo_height() - image.height()) // 2)
        self.preview_canvas.create_image(x, y, anchor=tk.NW, image=image)
        self.preview_canvas.configure(scrollregion=(0, 0, max(image.width() + 24, self.preview_canvas.winfo_width()), max(image.height() + 24, self.preview_canvas.winfo_height())))

    def _show_preview_text(self, message: str) -> None:
        self.preview_image = None
        self.preview_canvas.delete("all")
        width = max(300, self.preview_canvas.winfo_width())
        height = max(240, self.preview_canvas.winfo_height())
        self.preview_canvas.create_text(
            width // 2,
            height // 2,
            text=message,
            width=max(260, width - 48),
            fill="#667085",
            anchor=tk.CENTER,
        )
        self.preview_canvas.configure(scrollregion=(0, 0, width, height))

    def _preview_resized(self, _event=None) -> None:
        if self._preview_resize_job is not None:
            self.after_cancel(self._preview_resize_job)
        self._preview_resize_job = self.after(120, self._redraw_preview_after_resize)

    def _redraw_preview_after_resize(self) -> None:
        self._preview_resize_job = None
        if self.current_image_path is not None:
            self._draw_current_image()

    def _open_selected_artifact(self) -> None:
        selected = self.artifact_list.curselection()
        if not selected:
            return
        _open_path(Path(self.artifacts[selected[0]]["path"]))

    def _open_output_folder(self) -> None:
        output_dir = Path(self.base_dir_var.get()).expanduser() / "waterint_ui_output"
        _open_path(output_dir)


def _replace_text(widget: tk.Text, value: str) -> None:
    current = widget.get("1.0", tk.END).strip()
    if current != value:
        widget.delete("1.0", tk.END)
        widget.insert("1.0", value)


def _set_sashes(paned: ttk.PanedWindow, positions: list[int]) -> None:
    for index, position in enumerate(positions):
        try:
            paned.sashpos(index, position)
        except tk.TclError:
            pass


def _open_path(path: Path) -> None:
    path = path.expanduser().resolve()
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    elif os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def run_desktop_ui() -> None:
    app = WaterIntDesktopApp()
    app.mainloop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="waterint ui")
    parser.parse_args(argv)
    run_desktop_ui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
