from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from waterint.io.common import TrajectoryFrame


BOHR_TO_ANGSTROM = 0.529177210903
AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660
CM1_PER_THz = 33.35640951981521


@dataclass(frozen=True)
class UnitStyle:
    name: str
    length_unit: str
    length_to_angstrom: float
    time_unit: str
    time_to_ps: float
    mass_unit: str


@dataclass(frozen=True)
class UnitSystem:
    style: UnitStyle
    output_length_unit: str
    output_length_to_angstrom: float
    output_number_density_unit: str
    output_number_density_length_to_angstrom: float
    output_mass_density_unit: str
    output_time_unit: str
    output_time_to_ps: float
    output_frequency_unit: str

    def input_length(self, value: float) -> float:
        return float(value) * self.style.length_to_angstrom

    def input_length_array(self, values):
        return np.asarray(values, dtype=float) * self.style.length_to_angstrom

    def input_time_ps(self, value: float) -> float:
        return float(value) * self.style.time_to_ps

    def output_length(self, values):
        return np.asarray(values, dtype=float) / self.output_length_to_angstrom

    def output_number_density(self, values):
        return np.asarray(values, dtype=float) * self.output_number_density_length_to_angstrom**3

    def output_mass_density(self, values):
        data = np.asarray(values, dtype=float)
        unit = canonical_mass_density_unit(self.output_mass_density_unit)
        if unit == "g/cm^3":
            return data
        if unit == "kg/m^3":
            return data * 1000.0
        if unit == "amu/A^3":
            return data / AMU_PER_A3_TO_G_PER_CM3
        raise ValueError(f"Unsupported mass-density output unit: {self.output_mass_density_unit}")

    def output_time(self, values):
        return np.asarray(values, dtype=float) / self.output_time_to_ps

    def output_frequency(self, values):
        data = np.asarray(values, dtype=float)
        unit = canonical_frequency_unit(self.output_frequency_unit)
        if unit == "cm^-1":
            return data
        if unit == "THz":
            return data / CM1_PER_THz
        raise ValueError(f"Unsupported frequency output unit: {self.output_frequency_unit}")

    @property
    def length_label(self) -> str:
        return self.output_length_unit

    @property
    def number_density_label(self) -> str:
        return self.output_number_density_unit

    @property
    def mass_density_label(self) -> str:
        return canonical_mass_density_unit(self.output_mass_density_unit)

    @property
    def time_label(self) -> str:
        return self.output_time_unit

    @property
    def frequency_label(self) -> str:
        return canonical_frequency_unit(self.output_frequency_unit)


UNIT_STYLES: dict[str, UnitStyle] = {
    # `waterint` is kept as a legacy alias for the original package convention,
    # which matches the supported length/time dimensions of LAMMPS `metal`.
    "waterint": UnitStyle("waterint", "A", 1.0, "ps", 1.0, "amu"),
    "real": UnitStyle("real", "A", 1.0, "fs", 0.001, "g/mol"),
    "metal": UnitStyle("metal", "A", 1.0, "ps", 1.0, "g/mol"),
    "si": UnitStyle("si", "m", 1.0e10, "s", 1.0e12, "kg"),
    "cgs": UnitStyle("cgs", "cm", 1.0e8, "s", 1.0e12, "g"),
    "electron": UnitStyle("electron", "bohr", BOHR_TO_ANGSTROM, "fs", 0.001, "amu"),
    "micro": UnitStyle("micro", "um", 1.0e4, "us", 1.0e6, "pg"),
    "nano": UnitStyle("nano", "nm", 10.0, "ns", 1000.0, "ag"),
}


LENGTH_TO_ANGSTROM = {
    "a": 1.0,
    "angstrom": 1.0,
    "angstroms": 1.0,
    "nm": 10.0,
    "nanometer": 10.0,
    "nanometers": 10.0,
    "m": 1.0e10,
    "meter": 1.0e10,
    "meters": 1.0e10,
    "cm": 1.0e8,
    "centimeter": 1.0e8,
    "centimeters": 1.0e8,
    "um": 1.0e4,
    "micron": 1.0e4,
    "microns": 1.0e4,
    "bohr": BOHR_TO_ANGSTROM,
}

TIME_TO_PS = {
    "ps": 1.0,
    "picosecond": 1.0,
    "picoseconds": 1.0,
    "fs": 0.001,
    "femtosecond": 0.001,
    "femtoseconds": 0.001,
    "ns": 1000.0,
    "nanosecond": 1000.0,
    "nanoseconds": 1000.0,
    "us": 1.0e6,
    "microsecond": 1.0e6,
    "microseconds": 1.0e6,
    "s": 1.0e12,
    "sec": 1.0e12,
    "second": 1.0e12,
    "seconds": 1.0e12,
}


def unit_system_from_config(config: dict[str, Any]) -> UnitSystem:
    units_cfg = config.get("units", {})
    if units_cfg is None:
        units_cfg = {}
    if not isinstance(units_cfg, dict):
        raise ValueError("units must be a mapping.")

    style_name = str(units_cfg.get("style", "metal")).lower()
    if style_name == "lj":
        style = _lj_style(units_cfg)
    else:
        try:
            style = UNIT_STYLES[style_name]
        except KeyError as exc:
            supported = ", ".join(sorted([*UNIT_STYLES, "lj"]))
            raise ValueError(f"units.style must be one of: {supported}.") from exc

    output_cfg = units_cfg.get("output", {})
    if output_cfg is None:
        output_cfg = {}
    if not isinstance(output_cfg, dict):
        raise ValueError("units.output must be a mapping.")

    length_unit = str(output_cfg.get("length", style.length_unit))
    time_unit = str(output_cfg.get("time", style.time_unit))
    number_density_unit = str(output_cfg.get("number_density", f"1/{length_unit}^3"))
    mass_density_unit = str(output_cfg.get("mass_density", "g/cm^3"))
    frequency_unit = str(output_cfg.get("frequency", "cm^-1"))

    return UnitSystem(
        style=style,
        output_length_unit=canonical_length_unit(length_unit),
        output_length_to_angstrom=length_to_angstrom(length_unit),
        output_number_density_unit=canonical_number_density_unit(number_density_unit),
        output_number_density_length_to_angstrom=length_to_angstrom(number_density_length_unit(number_density_unit)),
        output_mass_density_unit=canonical_mass_density_unit(mass_density_unit),
        output_time_unit=canonical_time_unit(time_unit),
        output_time_to_ps=time_to_ps(time_unit),
        output_frequency_unit=canonical_frequency_unit(frequency_unit),
    )


def convert_frame_to_internal_units(frame: TrajectoryFrame, units: UnitSystem) -> TrajectoryFrame:
    scale = units.style.length_to_angstrom
    if scale == 1.0:
        return frame
    cell = None if frame.cell is None else tuple(float(value) * scale for value in frame.cell)
    return TrajectoryFrame(
        index=frame.index,
        comment=frame.comment,
        symbols=frame.symbols,
        positions=np.asarray(frame.positions, dtype=float) * scale,
        cell=cell,
        step=frame.step,
        types=frame.types,
    )


def length_to_angstrom(unit: str) -> float:
    key = _unit_key(unit)
    try:
        return LENGTH_TO_ANGSTROM[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported length unit: {unit}") from exc


def time_to_ps(unit: str) -> float:
    key = _unit_key(unit)
    try:
        return TIME_TO_PS[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported time unit: {unit}") from exc


def canonical_length_unit(unit: str) -> str:
    key = _unit_key(unit)
    if key in {"a", "angstrom", "angstroms"}:
        return "A"
    if key in {"nm", "nanometer", "nanometers"}:
        return "nm"
    if key in {"um", "micron", "microns"}:
        return "um"
    if key in {"cm", "centimeter", "centimeters"}:
        return "cm"
    if key in {"m", "meter", "meters"}:
        return "m"
    if key == "bohr":
        return "bohr"
    raise ValueError(f"Unsupported length unit: {unit}")


def canonical_time_unit(unit: str) -> str:
    key = _unit_key(unit)
    if key in {"fs", "femtosecond", "femtoseconds"}:
        return "fs"
    if key in {"ps", "picosecond", "picoseconds"}:
        return "ps"
    if key in {"ns", "nanosecond", "nanoseconds"}:
        return "ns"
    if key in {"us", "microsecond", "microseconds"}:
        return "us"
    if key in {"s", "sec", "second", "seconds"}:
        return "s"
    raise ValueError(f"Unsupported time unit: {unit}")


def canonical_number_density_unit(unit: str) -> str:
    text = str(unit).strip().replace(" ", "")
    aliases = {
        "1/A^3": "1/A^3",
        "1/angstrom^3": "1/A^3",
        "1/angstroms^3": "1/A^3",
        "A^-3": "1/A^3",
        "angstrom^-3": "1/A^3",
        "1/nm^3": "1/nm^3",
        "nm^-3": "1/nm^3",
        "1/m^3": "1/m^3",
        "m^-3": "1/m^3",
        "1/cm^3": "1/cm^3",
        "cm^-3": "1/cm^3",
        "1/um^3": "1/um^3",
        "um^-3": "1/um^3",
        "1/bohr^3": "1/bohr^3",
        "bohr^-3": "1/bohr^3",
    }
    key = text.lower().replace("å", "a")
    if key in aliases:
        return aliases[key]
    if key.startswith("1/") and key.endswith("^3"):
        length_unit = key[2:-2]
        return f"1/{canonical_length_unit(length_unit)}^3"
    raise ValueError(f"Unsupported number-density output unit: {unit}")


def number_density_length_unit(unit: str) -> str:
    canonical = canonical_number_density_unit(unit)
    return canonical[2:-2]


def canonical_mass_density_unit(unit: str) -> str:
    key = str(unit).strip().replace(" ", "").lower()
    aliases = {
        "g/cm^3": "g/cm^3",
        "g/cm3": "g/cm^3",
        "g_cm3": "g/cm^3",
        "kg/m^3": "kg/m^3",
        "kg/m3": "kg/m^3",
        "kg_m3": "kg/m^3",
        "amu/a^3": "amu/A^3",
        "amu/a3": "amu/A^3",
        "amu/angstrom^3": "amu/A^3",
    }
    if key in aliases:
        return aliases[key]
    raise ValueError(f"Unsupported mass-density output unit: {unit}")


def canonical_frequency_unit(unit: str) -> str:
    key = str(unit).strip().replace(" ", "").lower()
    aliases = {
        "cm^-1": "cm^-1",
        "1/cm": "cm^-1",
        "cm-1": "cm^-1",
        "wavenumber": "cm^-1",
        "thz": "THz",
    }
    if key in aliases:
        return aliases[key]
    raise ValueError(f"Unsupported frequency output unit: {unit}")


def _lj_style(units_cfg: dict[str, Any]) -> UnitStyle:
    lj_cfg = units_cfg.get("lj", {})
    if not isinstance(lj_cfg, dict):
        raise ValueError("units.lj must be a mapping when units.style is lj.")
    sigma = lj_cfg.get("sigma_A")
    tau = lj_cfg.get("tau_ps")
    if sigma is None or tau is None:
        raise ValueError("units.style: lj requires units.lj.sigma_A and units.lj.tau_ps.")
    sigma = float(sigma)
    tau = float(tau)
    if sigma <= 0 or tau <= 0:
        raise ValueError("units.lj.sigma_A and units.lj.tau_ps must be positive.")
    return UnitStyle("lj", "sigma", sigma, "tau", tau, "m")


def _unit_key(unit: str) -> str:
    return str(unit).strip().replace(" ", "").replace("Angstrom", "angstrom").replace("Å", "A").lower()
