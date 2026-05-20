"""Trusted physics knowledge bank for Type 2 solving."""

from __future__ import annotations

from .schemas import Formula, LawStatement


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
    Formula(
        "net_coulomb_force_equilateral",
        "coulomb_force",
        "F = sqrt(3)*k*Abs(q1*q3)/r**2",
        "sqrt(3)*k*Abs(q1*q3)/r**2",
        ("k", "q1", "q3", "r"),
        ("F",),
        "N",
        ("force", "coulomb", "equilateral", "triangle", "vector", "net"),
        "Net force on a vertex charge in an equilateral triangle with equal source distances: F = sqrt(3) k |q1 q3| / r^2",
    ),
    Formula(
        "net_coulomb_force_perpendicular_bisector",
        "coulomb_force",
        "F = sqrt(Fx**2 + Fy**2)",
        "sqrt(Fx**2 + Fy**2)",
        ("k", "q1", "q2", "q", "AB", "h"),
        ("F",),
        "N",
        ("force", "coulomb", "perpendicular", "bisector", "net", "magnitude"),
        "Net Coulomb force at a point on the perpendicular bisector: combine x and y components.",
        (
            ("a", "AB/2"),
            ("r", "sqrt(a**2 + h**2)"),
            ("F1", "k*Abs(q1*q)/r**2"),
            ("F2", "k*Abs(q2*q)/r**2"),
            ("Fx", "(F1 + F2)*a/r"),
            ("Fy", "(F1 - F2)*h/r"),
        ),
    ),
    Formula("electric_field_force", "electric_field", "E = F/q", "F/q", ("F", "q"), ("E",), "N/C", ("electric", "field"), "Electric field: E = F/q"),
    Formula("electric_field_charge", "electric_field", "E = k*q/r**2", "k*q/r**2", ("k", "q", "r"), ("E",), "N/C", ("electric", "field", "charge"), "Electric field: E = k*q/r^2"),
    Formula("electric_potential", "electric_potential", "V = k*q/r", "k*q/r", ("k", "q", "r"), ("V", "U"), "V", ("electric", "potential", "voltage"), "Electric potential: V = k*q/r"),
    Formula("capacitance", "capacitance", "C = Q/V", "Q/V", ("Q", "V"), ("C",), "F", ("capacitance", "charge", "voltage"), "Capacitance: C = Q/V"),
    Formula("capacitor_energy", "capacitor_energy", "E = 0.5*C*V**2", "0.5*C*V**2", ("C", "V"), ("E",), "J", ("energy", "capacitor", "stored"), "Capacitor energy: E = 0.5*C*V^2"),
)


LAW_STATEMENTS: tuple[LawStatement, ...] = (
    LawStatement(
        id="passive_sign_convention",
        topic="circuits",
        statement="Passive sign convention: current enters the positive terminal of an element.",
        keywords=("passive", "sign", "current", "terminal", "positive", "voltage"),
        aliases=("PSC",),
        related_formula_ids=("ohm_voltage", "power_vi"),
    ),
    LawStatement(
        id="voltage_drop_current_direction",
        topic="circuits",
        statement="Voltage drop is in the direction of conventional current flow through a passive element.",
        keywords=("voltage", "drop", "current", "direction", "resistor"),
        related_formula_ids=("ohm_voltage",),
    ),
    LawStatement(
        id="series_current_same",
        topic="series_resistance",
        statement="In a series circuit, the same current flows through every element.",
        keywords=("series", "same", "current", "resistor", "circuit"),
        related_formula_ids=("series_resistance", "ohm_current", "ohm_voltage"),
    ),
    LawStatement(
        id="series_voltage_adds",
        topic="series_resistance",
        statement="In a series circuit, source voltage equals the sum of element voltage drops.",
        keywords=("series", "voltage", "sum", "drop", "kirchhoff"),
        related_formula_ids=("series_resistance", "ohm_voltage"),
    ),
    LawStatement(
        id="parallel_voltage_same",
        topic="parallel_resistance",
        statement="In a parallel circuit, each branch has the same voltage across it.",
        keywords=("parallel", "same", "voltage", "branch", "circuit"),
        related_formula_ids=("parallel_two_resistance", "ohm_current"),
    ),
    LawStatement(
        id="parallel_currents_add",
        topic="parallel_resistance",
        statement="In a parallel circuit, total current equals the sum of branch currents.",
        keywords=("parallel", "current", "sum", "branch", "kirchhoff"),
        related_formula_ids=("parallel_two_resistance", "ohm_current"),
    ),
    LawStatement(
        id="kirchhoff_current_law",
        topic="circuits",
        statement="Kirchhoff current law: total current entering a node equals total current leaving it.",
        keywords=("kirchhoff", "kcl", "node", "current", "entering", "leaving"),
        aliases=("KCL",),
    ),
    LawStatement(
        id="kirchhoff_voltage_law",
        topic="circuits",
        statement="Kirchhoff voltage law: signed voltage changes around a closed loop sum to zero.",
        keywords=("kirchhoff", "kvl", "loop", "voltage", "sum"),
        aliases=("KVL",),
    ),
    LawStatement(
        id="coulomb_superposition",
        topic="coulomb_force",
        statement="For multiple charges, electric forces add by vector superposition.",
        keywords=("coulomb", "force", "superposition", "vector", "charges"),
        related_formula_ids=("coulomb_force", "net_coulomb_force_right_triangle"),
    ),
    LawStatement(
        id="electric_field_superposition",
        topic="electric_field",
        statement="Electric fields from multiple source charges add by vector superposition.",
        keywords=("electric", "field", "superposition", "vector", "charges"),
        related_formula_ids=("electric_field_charge",),
    ),
)


KnowledgeItem = Formula | LawStatement


def get_formula(formula_id: str) -> Formula | None:
    return next((formula for formula in FORMULAS if formula.id == formula_id), None)


def get_law(law_id: str) -> LawStatement | None:
    return next((law for law in LAW_STATEMENTS if law.id == law_id), None)
