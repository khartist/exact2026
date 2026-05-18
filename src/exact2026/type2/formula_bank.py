"""Small trusted formula bank for Type 2 MVP solving."""

from __future__ import annotations

from .schemas import Formula


FORMULAS: tuple[Formula, ...] = (
    Formula("ohm_voltage", "ohm_law", "V = I * R", "I * R", ("I", "R"), ("V", "U"), "V", ("voltage", "potential", "ohm"), "Ohm law: V = I * R"),
    Formula("ohm_current", "ohm_law", "I = V / R", "V / R", ("V", "R"), ("I",), "A", ("current", "ohm"), "Ohm law current: I = V/R"),
    Formula("ohm_resistance", "ohm_law", "R = V / I", "V / I", ("V", "I"), ("R", "Z"), "Ω", ("resistance", "impedance", "ohm"), "Ohm law resistance: R = V/I"),
    Formula("power_vi", "power", "P = V * I", "V * I", ("V", "I"), ("P",), "W", ("power", "watt"), "Power: P = V * I"),
    Formula("power_i2r", "power", "P = I**2 * R", "I**2 * R", ("I", "R"), ("P",), "W", ("power", "dissipated", "real"), "Power: P = I^2 * R"),
    Formula("power_v2r", "power", "P = V**2 / R", "V**2 / R", ("V", "R"), ("P",), "W", ("power", "dissipated"), "Power: P = V^2/R"),
    Formula("series_resistance", "series_resistance", "Req = R1 + R2 + ...", "R1 + R2", ("R1", "R2"), ("Req", "R"), "Ω", ("series", "equivalent", "resistance"), "Series resistance: Req = R1 + R2 + ..."),
    Formula("parallel_two_resistance", "parallel_resistance", "Req = (R1*R2)/(R1+R2)", "(R1*R2)/(R1+R2)", ("R1", "R2"), ("Req", "R"), "Ω", ("parallel", "equivalent", "resistance"), "Parallel resistance: Req = (R1*R2)/(R1+R2)"),
    Formula("inductive_reactance", "inductive_reactance", "X_L = 2*pi*f*L", "2*pi*f*L", ("f", "L"), ("X_L", "XL"), "Ω", ("inductive", "reactance", "inductor"), "Inductive reactance: X_L = 2πfL"),
    Formula("capacitive_reactance", "capacitive_reactance", "X_C = 1/(2*pi*f*C)", "1/(2*pi*f*C)", ("f", "C"), ("X_C", "XC"), "Ω", ("capacitive", "reactance", "capacitor"), "Capacitive reactance: X_C = 1/(2πfC)"),
    Formula(
        "series_rlc_impedance",
        "series_rlc_impedance",
        "Z = sqrt(R**2 + (X_L - X_C)**2)",
        "sqrt(R**2 + (X_L - X_C)**2)",
        ("R", "f", "L", "C"),
        ("Z",),
        "Ω",
        ("rlc", "impedance", "series"),
        "Series RLC impedance: Z = sqrt(R^2 + (X_L - X_C)^2)",
        (("X_L", "2*pi*f*L"), ("X_C", "1/(2*pi*f*C)")),
    ),
    Formula("ac_current", "ac_current", "I = V / Z", "V / Z", ("V", "Z"), ("I",), "A", ("current", "impedance", "ac"), "AC current: I = V/Z"),
    Formula("ac_real_power_vzr", "ac_real_power", "P = (V/Z)**2 * R", "(V/Z)**2 * R", ("V", "Z", "R"), ("P",), "W", ("power", "real", "dissipated", "impedance"), "AC real power: P = (V/Z)^2 * R", (("I", "V/Z"),)),
    Formula("coulomb_force", "coulomb_force", "F = k*q1*q2/r**2", "k*q1*q2/r**2", ("k", "q1", "q2", "r"), ("F",), "N", ("force", "coulomb", "charge"), "Coulomb force: F = k*q1*q2/r^2"),
    Formula(
        "net_coulomb_force_right_triangle",
        "net_coulomb_force",
        "F = sqrt(F13**2 + F23**2)",
        "sqrt(F13**2 + F23**2)",
        ("k", "q1", "q2", "q3", "AC", "BC"),
        ("F",),
        "N",
        ("force", "test", "charge", "coulomb", "magnitude"),
        "Net Coulomb force at right angle: F = sqrt(F13^2 + F23^2)",
        (
            ("F13", "k*Abs(q1*q3)/AC**2"),
            ("F23", "k*Abs(q2*q3)/BC**2"),
        ),
    ),
    Formula("electric_field_force", "electric_field", "E = F/q", "F/q", ("F", "q"), ("E",), "N/C", ("electric", "field"), "Electric field: E = F/q"),
    Formula("electric_field_charge", "electric_field", "E = k*q/r**2", "k*q/r**2", ("k", "q", "r"), ("E",), "N/C", ("electric", "field", "charge"), "Electric field: E = k*q/r^2"),
    Formula("electric_potential", "electric_potential", "V = k*q/r", "k*q/r", ("k", "q", "r"), ("V", "U"), "V", ("electric", "potential", "voltage"), "Electric potential: V = k*q/r"),
    Formula("capacitance", "capacitance", "C = Q/V", "Q/V", ("Q", "V"), ("C",), "F", ("capacitance", "charge", "voltage"), "Capacitance: C = Q/V"),
    Formula("capacitor_energy", "capacitor_energy", "E = 0.5*C*V**2", "0.5*C*V**2", ("C", "V"), ("E",), "J", ("energy", "capacitor", "stored"), "Capacitor energy: E = 0.5*C*V^2"),
)


def get_formula(formula_id: str) -> Formula | None:
    return next((formula for formula in FORMULAS if formula.id == formula_id), None)
