"""Unit normalization for Type 2 physics problems."""

from __future__ import annotations

import re


UNIT_ALIASES: dict[str, tuple[float, str]] = {
    "ohm": (1.0, "Ω"),
    "ohms": (1.0, "Ω"),
    "Ω": (1.0, "Ω"),
    "ω": (1.0, "Ω"),
    "kohm": (1e3, "Ω"),
    "kω": (1e3, "Ω"),
    "kΩ": (1e3, "Ω"),
    "v": (1.0, "V"),
    "V": (1.0, "V"),
    "volt": (1.0, "V"),
    "volts": (1.0, "V"),
    "mv": (1e-3, "V"),
    "mV": (1e-3, "V"),
    "kv": (1e3, "V"),
    "kV": (1e3, "V"),
    "a": (1.0, "A"),
    "A": (1.0, "A"),
    "amp": (1.0, "A"),
    "amps": (1.0, "A"),
    "ma": (1e-3, "A"),
    "mA": (1e-3, "A"),
    "w": (1.0, "W"),
    "W": (1.0, "W"),
    "watt": (1.0, "W"),
    "watts": (1.0, "W"),
    "h": (1.0, "H"),
    "H": (1.0, "H"),
    "henry": (1.0, "H"),
    "henries": (1.0, "H"),
    "mh": (1e-3, "H"),
    "mH": (1e-3, "H"),
    "f": (1.0, "F"),
    "F": (1.0, "F"),
    "farad": (1.0, "F"),
    "farads": (1.0, "F"),
    "uf": (1e-6, "F"),
    "uF": (1e-6, "F"),
    "μF": (1e-6, "F"),
    "µF": (1e-6, "F"),
    "nf": (1e-9, "F"),
    "nF": (1e-9, "F"),
    "pf": (1e-12, "F"),
    "pF": (1e-12, "F"),
    "hz": (1.0, "Hz"),
    "Hz": (1.0, "Hz"),
    "hertz": (1.0, "Hz"),
    "c": (1.0, "C"),
    "C": (1.0, "C"),
    "coulomb": (1.0, "C"),
    "coulombs": (1.0, "C"),
    "nc": (1e-9, "C"),
    "nC": (1e-9, "C"),
    "uc": (1e-6, "C"),
    "uC": (1e-6, "C"),
    "μC": (1e-6, "C"),
    "µC": (1e-6, "C"),
    "m": (1.0, "m"),
    "cm": (1e-2, "m"),
    "meter": (1.0, "m"),
    "meters": (1.0, "m"),
    "n/c": (1.0, "N/C"),
    "N/C": (1.0, "N/C"),
    "n": (1.0, "N"),
    "N": (1.0, "N"),
    "j": (1.0, "J"),
    "J": (1.0, "J"),
}


def normalize_unit(unit: str) -> tuple[float, str]:
    cleaned = clean_unit(unit)
    if cleaned in UNIT_ALIASES:
        return UNIT_ALIASES[cleaned]
    lower = cleaned.lower()
    if lower in UNIT_ALIASES:
        return UNIT_ALIASES[lower]
    return 1.0, cleaned


def convert_value(value: float, unit: str) -> tuple[float, str]:
    factor, normalized = normalize_unit(unit)
    return value * factor, normalized


def clean_unit(unit: str) -> str:
    text = unit.strip()
    text = text.replace("Ω", "Ω").replace("μ", "μ").replace("µ", "µ")
    text = re.sub(r"[.,;:)]+$", "", text)
    return text
