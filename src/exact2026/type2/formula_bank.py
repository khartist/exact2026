"""Trusted physics knowledge bank for Type 2 solving."""

from __future__ import annotations

from typing import Iterable

from .schemas import (
    Category,
    Diagram,
    Formula,
    LawStatement,
    PhysicalConstant,
    SymbolDefinition,
    WorkedExample,
)

FORMULAS: tuple[Formula, ...] = (
    Formula(
        "ohm_voltage",
        "ohm_law",
        "V = I * R",
        "I * R",
        ("I", "R"),
        ("V", "U"),
        "V",
        ("voltage", "potential", "ohm"),
        "Ohm law: V = I * R",
    ),
    Formula(
        "ohm_current",
        "ohm_law",
        "I = V / R",
        "V / R",
        ("V", "R"),
        ("I",),
        "A",
        ("current", "ohm"),
        "Ohm law current: I = V/R",
    ),
    Formula(
        "ohm_resistance",
        "ohm_law",
        "R = V / I",
        "V / I",
        ("V", "I"),
        ("R", "Z"),
        "Ω",
        ("resistance", "impedance", "ohm"),
        "Ohm law resistance: R = V/I",
    ),
    Formula(
        "power_vi",
        "power",
        "P = V * I",
        "V * I",
        ("V", "I"),
        ("P",),
        "W",
        ("power", "watt"),
        "Power: P = V * I",
    ),
    Formula(
        "power_i2r",
        "power",
        "P = I**2 * R",
        "I**2 * R",
        ("I", "R"),
        ("P",),
        "W",
        ("power", "dissipated", "real"),
        "Power: P = I^2 * R",
    ),
    Formula(
        "power_v2r",
        "power",
        "P = V**2 / R",
        "V**2 / R",
        ("V", "R"),
        ("P",),
        "W",
        ("power", "dissipated"),
        "Power: P = V^2/R",
    ),
    Formula(
        "series_resistance",
        "series_resistance",
        "Req = R1 + R2 + ...",
        "R1 + R2",
        ("R1", "R2"),
        ("Req", "R"),
        "Ω",
        ("series", "equivalent", "resistance"),
        "Series resistance: Req = R1 + R2 + ...",
    ),
    Formula(
        "parallel_two_resistance",
        "parallel_resistance",
        "Req = (R1*R2)/(R1+R2)",
        "(R1*R2)/(R1+R2)",
        ("R1", "R2"),
        ("Req", "R"),
        "Ω",
        ("parallel", "equivalent", "resistance"),
        "Parallel resistance: Req = (R1*R2)/(R1+R2)",
    ),
    Formula(
        "inductive_reactance",
        "inductive_reactance",
        "X_L = 2*pi*f*L",
        "2*pi*f*L",
        ("f", "L"),
        ("X_L", "XL"),
        "Ω",
        ("inductive", "reactance", "inductor"),
        "Inductive reactance: X_L = 2πfL",
    ),
    Formula(
        "capacitive_reactance",
        "capacitive_reactance",
        "X_C = 1/(2*pi*f*C)",
        "1/(2*pi*f*C)",
        ("f", "C"),
        ("X_C", "XC"),
        "Ω",
        ("capacitive", "reactance", "capacitor"),
        "Capacitive reactance: X_C = 1/(2πfC)",
    ),
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
    Formula(
        "ac_current",
        "ac_current",
        "I = V / Z",
        "V / Z",
        ("V", "Z"),
        ("I",),
        "A",
        ("current", "impedance", "ac"),
        "AC current: I = V/Z",
    ),
    Formula(
        "ac_real_power_vzr",
        "ac_real_power",
        "P = (V/Z)**2 * R",
        "(V/Z)**2 * R",
        ("V", "Z", "R"),
        ("P",),
        "W",
        ("power", "real", "dissipated", "impedance"),
        "AC real power: P = (V/Z)^2 * R",
        (("I", "V/Z"),),
    ),
    Formula(
        "coulomb_force",
        "coulomb_force",
        "F = k*q1*q2/r**2",
        "k*q1*q2/r**2",
        ("k", "q1", "q2", "r"),
        ("F",),
        "N",
        ("force", "coulomb", "charge"),
        "Coulomb force: F = k*q1*q2/r^2",
    ),
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
    Formula(
        "electric_field_force",
        "electric_field",
        "E = F/q",
        "F/q",
        ("F", "q"),
        ("E",),
        "N/C",
        ("electric", "field"),
        "Electric field: E = F/q",
    ),
    Formula(
        "electric_field_charge",
        "electric_field",
        "E = k*q/r**2",
        "k*q/r**2",
        ("k", "q", "r"),
        ("E",),
        "N/C",
        ("electric", "field", "charge"),
        "Electric field: E = k*q/r^2",
    ),
    Formula(
        "electric_potential",
        "electric_potential",
        "V = k*q/r",
        "k*q/r",
        ("k", "q", "r"),
        ("V", "U"),
        "V",
        ("electric", "potential", "voltage"),
        "Electric potential: V = k*q/r",
    ),
    Formula(
        "capacitance",
        "capacitance",
        "C = Q/V",
        "Q/V",
        ("Q", "V"),
        ("C",),
        "F",
        ("capacitance", "charge", "voltage"),
        "Capacitance: C = Q/V",
    ),
    Formula(
        "capacitor_energy",
        "capacitor_energy",
        "E = 0.5*C*V**2",
        "0.5*C*V**2",
        ("C", "V"),
        ("E",),
        "J",
        ("energy", "capacitor", "stored"),
        "Capacitor energy: E = 0.5*C*V^2",
    ),
    Formula(
        "capacitor_charge",
        "capacitors",
        "Q = C * V",
        "C * V",
        ("C", "V"),
        ("Q",),
        "C",
        ("capacitor", "charge", "voltage"),
        "Capacitor charge: Q = C * V",
    ),
    Formula(
        "capacitor_voltage",
        "capacitors",
        "V = Q / C",
        "Q / C",
        ("Q", "C"),
        ("V", "U"),
        "V",
        ("capacitor", "voltage", "charge"),
        "Capacitor voltage: V = Q / C",
    ),
    Formula(
        "capacitor_energy_qv",
        "capacitor_energy",
        "E = 0.5*Q*V",
        "0.5*Q*V",
        ("Q", "V"),
        ("E",),
        "J",
        ("energy", "capacitor", "charge", "voltage"),
        "Capacitor energy: E = 0.5 * Q * V",
    ),
    Formula(
        "capacitor_energy_q2c",
        "capacitor_energy",
        "E = Q**2 / (2*C)",
        "Q**2 / (2*C)",
        ("Q", "C"),
        ("E",),
        "J",
        ("energy", "capacitor", "charge"),
        "Capacitor energy: E = Q^2 / (2C)",
    ),
    Formula(
        "parallel_capacitance",
        "capacitors",
        "Ceq = C1 + C2 + ...",
        "C1 + C2",
        ("C1", "C2"),
        ("Ceq", "C"),
        "F",
        ("parallel", "capacitor", "equivalent"),
        "Parallel capacitors: Ceq = C1 + C2 + ...",
    ),
    Formula(
        "series_capacitance",
        "capacitors",
        "1/Ceq = 1/C1 + 1/C2 + ...",
        "1/(1/C1 + 1/C2)",
        ("C1", "C2"),
        ("Ceq", "C"),
        "F",
        ("series", "capacitor", "equivalent"),
        "Series capacitors: 1/Ceq = 1/C1 + 1/C2 + ...",
    ),
    Formula(
        "rc_time_constant",
        "rc_circuit",
        "tau = R * C",
        "R * C",
        ("R", "C"),
        ("tau", "t"),
        "s",
        ("rc", "time", "constant", "capacitor", "resistor"),
        "RC time constant: tau = R * C",
    ),
    Formula(
        "voltage_divider",
        "voltage_divider",
        "Vout = Vin * R2 / (R1 + R2)",
        "Vin * R2 / (R1 + R2)",
        ("Vin", "R1", "R2"),
        ("Vout", "V2", "U2"),
        "V",
        ("voltage", "divider", "resistor"),
        "Voltage divider: Vout = Vin * R2 / (R1 + R2)",
    ),
    Formula(
        "current_divider",
        "current_divider",
        "I1 = Itotal * R2 / (R1 + R2)",
        "Itotal * R2 / (R1 + R2)",
        ("Itotal", "R1", "R2"),
        ("I1", "I2"),
        "A",
        ("current", "divider", "parallel", "resistor"),
        "Current divider: I1 = Itotal * R2 / (R1 + R2)",
    ),
    Formula(
        "resistivity",
        "materials",
        "R = rho * L / A",
        "rho * L / A",
        ("rho", "L", "A"),
        ("R",),
        "Ω",
        ("resistivity", "resistance", "wire"),
        "Resistance of a wire: R = rho * L / A",
    ),
    Formula(
        "electric_force_field",
        "electric_field",
        "F = q * E",
        "q * E",
        ("q", "E"),
        ("F",),
        "N",
        ("force", "electric", "field"),
        "Electric force: F = q * E",
    ),
    Formula(
        "electric_potential_energy",
        "electrostatics",
        "U = k*q1*q2/r",
        "k*q1*q2/r",
        ("k", "q1", "q2", "r"),
        ("U", "E"),
        "J",
        ("potential", "energy", "charge", "coulomb"),
        "Electric potential energy: U = k*q1*q2/r",
    ),
    Formula(
        "current_definition",
        "current",
        "I = Q / t",
        "Q / t",
        ("Q", "t"),
        ("I",),
        "A",
        ("current", "charge", "rate", "flow"),
        "Electric current is charge transferred per unit time: I = Q/t",
    ),
    Formula(
        "potential_difference_definition",
        "potential_difference",
        "V = W / Q",
        "W / Q",
        ("W", "Q"),
        ("V", "U"),
        "V",
        ("potential", "difference", "voltage", "work", "energy", "charge"),
        "Potential difference is energy transferred per unit charge: V = W/Q",
    ),
    Formula(
        "internal_resistance_terminal_voltage",
        "internal_resistance",
        "V = emf - I*r_int",
        "emf - I*r_int",
        ("emf", "I", "r_int"),
        ("V", "U"),
        "V",
        ("internal", "resistance", "terminal", "emf", "lost", "volts"),
        "Terminal voltage of a source delivering current: V = emf - I*r_int",
    ),
    Formula(
        "internal_resistance_current",
        "internal_resistance",
        "I = emf/(R + r_int)",
        "emf/(R + r_int)",
        ("emf", "R", "r_int"),
        ("I",),
        "A",
        ("internal", "resistance", "current", "emf", "cell"),
        "Current in a simple circuit with source internal resistance: I = emf/(R + r_int)",
    ),
    Formula(
        "electric_field_uniform",
        "electric_field",
        "E = V / d",
        "V / d",
        ("V", "d"),
        ("E",),
        "V/m",
        ("uniform", "electric", "field", "parallel", "plates", "potential", "difference"),
        "Uniform electric field magnitude between parallel plates: E = V/d",
    ),
    Formula(
        "electric_potential_definition",
        "electric_potential",
        "V = U / q",
        "U / q",
        ("U", "q"),
        ("V",),
        "V",
        ("electric", "potential", "energy", "charge"),
        "Electric potential is electric potential energy per unit charge: V = U/q",
    ),
    Formula(
        "electric_potential_energy_change",
        "electric_potential_energy",
        "delta_U = q*delta_V",
        "q*delta_V",
        ("q", "delta_V"),
        ("delta_U", "U"),
        "J",
        ("potential", "energy", "change", "voltage", "work"),
        "Change in electric potential energy: delta_U = q*delta_V",
    ),
    Formula(
        "parallel_plate_capacitance",
        "capacitance",
        "C = eps0*A/d",
        "eps0*A/d",
        ("eps0", "A", "d"),
        ("C",),
        "F",
        ("parallel", "plate", "capacitance", "area", "separation", "permittivity"),
        "Capacitance of a vacuum or air-gap parallel-plate capacitor: C = eps0*A/d",
    ),
    Formula(
        "dielectric_capacitance",
        "capacitance",
        "C = eps_r*eps0*A/d",
        "eps_r*eps0*A/d",
        ("eps_r", "eps0", "A", "d"),
        ("C",),
        "F",
        ("dielectric", "relative", "permittivity", "capacitor", "parallel", "plate"),
        "Parallel-plate capacitance with dielectric: C = eps_r*eps0*A/d",
    ),
    Formula(
        "rc_charge_voltage",
        "rc_circuit",
        "V_C = V0*(1 - exp(-t/(R*C)))",
        "V0*(1 - exp(-t/(R*C)))",
        ("V0", "t", "R", "C"),
        ("V_C", "V"),
        "V",
        ("rc", "charging", "capacitor", "voltage", "exponential"),
        "Capacitor voltage while charging from an ideal source: V_C = V0*(1 - exp(-t/(R*C)))",
    ),
    Formula(
        "rc_charge_current",
        "rc_circuit",
        "I = (V0/R)*exp(-t/(R*C))",
        "(V0/R)*exp(-t/(R*C))",
        ("V0", "R", "t", "C"),
        ("I",),
        "A",
        ("rc", "charging", "capacitor", "current", "exponential"),
        "Charging current magnitude in an RC circuit: I = (V0/R)*exp(-t/(R*C))",
    ),
    Formula(
        "rc_discharge_voltage",
        "rc_circuit",
        "V_C = V0*exp(-t/(R*C))",
        "V0*exp(-t/(R*C))",
        ("V0", "t", "R", "C"),
        ("V_C", "V"),
        "V",
        ("rc", "discharging", "capacitor", "voltage", "exponential"),
        "Capacitor voltage while discharging through a resistor: V_C = V0*exp(-t/(R*C))",
    ),
    Formula(
        "rc_discharge_charge",
        "rc_circuit",
        "Q = Q0*exp(-t/(R*C))",
        "Q0*exp(-t/(R*C))",
        ("Q0", "t", "R", "C"),
        ("Q",),
        "C",
        ("rc", "discharging", "capacitor", "charge", "exponential"),
        "Charge on a discharging capacitor: Q = Q0*exp(-t/(R*C))",
    ),
    Formula(
        "rc_half_time",
        "rc_circuit",
        "t_half = R*C*log(2)",
        "R*C*log(2)",
        ("R", "C"),
        ("t_half",),
        "s",
        ("rc", "half", "time", "half-life", "discharge", "capacitor"),
        "Half-time for a first-order RC discharge: t_half = R*C*log(2)",
    ),
    Formula(
        "ac_rms_voltage",
        "alternating_current",
        "V_rms = V0/sqrt(2)",
        "V0/sqrt(2)",
        ("V0",),
        ("V_rms", "V"),
        "V",
        ("rms", "voltage", "peak", "sinusoidal", "ac"),
        "RMS voltage for a sinusoidal waveform: V_rms = V0/sqrt(2)",
    ),
    Formula(
        "ac_rms_current",
        "alternating_current",
        "I_rms = I0/sqrt(2)",
        "I0/sqrt(2)",
        ("I0",),
        ("I_rms", "I"),
        "A",
        ("rms", "current", "peak", "sinusoidal", "ac"),
        "RMS current for a sinusoidal waveform: I_rms = I0/sqrt(2)",
    ),
    Formula(
        "angular_frequency",
        "alternating_current",
        "omega = 2*pi*f",
        "2*pi*f",
        ("f",),
        ("omega",),
        "rad/s",
        ("angular", "frequency", "omega", "period", "ac"),
        "Angular frequency for a sinusoidal signal: omega = 2*pi*f",
    ),
    Formula(
        "frequency_period",
        "alternating_current",
        "f = 1/T",
        "1/T",
        ("T",),
        ("f",),
        "Hz",
        ("frequency", "period", "ac"),
        "Frequency is the reciprocal of period: f = 1/T",
    ),
    Formula(
        "diode_series_resistor_current",
        "semiconductor",
        "I = (V - V_D)/R",
        "(V - V_D)/R",
        ("V", "V_D", "R"),
        ("I",),
        "A",
        ("diode", "resistor", "current", "forward", "bias", "silicon"),
        "Approximate current for a forward-biased diode with a series resistor: I = (V - V_D)/R",
    ),
    Formula(
        "amplifier_voltage_gain",
        "semiconductor",
        "A_v = Vout/Vin",
        "Vout/Vin",
        ("Vout", "Vin"),
        ("A_v",),
        "dimensionless",
        ("amplifier", "gain", "voltage", "output", "input"),
        "Voltage gain of an amplifier: A_v = Vout/Vin",
    ),
    Formula(
        "transistor_current_gain_bjt",
        "semiconductor",
        "I_C = beta*I_B",
        "beta*I_B",
        ("beta", "I_B"),
        ("I_C",),
        "A",
        ("transistor", "bjt", "current", "gain", "collector", "base"),
        "Active-region BJT current estimate: I_C = beta*I_B",
    ),
    Formula(
        "transistor_collector_load_voltage",
        "semiconductor",
        "Vout = V - I_C*R_C",
        "V - I_C*R_C",
        ("V", "I_C", "R_C"),
        ("Vout", "V_CE"),
        "V",
        ("transistor", "collector", "load", "voltage", "common", "emitter"),
        "Collector output voltage with a load resistor: Vout = V - I_C*R_C",
    ),
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
        topic="circuits",
        statement="In a series circuit, source voltage equals the sum of element voltage drops.",
        keywords=("series", "voltage", "sum", "drop", "kirchhoff"),
        related_formula_ids=("series_resistance", "ohm_voltage"),
    ),
    LawStatement(
        id="parallel_voltage_same",
        topic="circuits",
        statement="In a parallel circuit, each branch has the same voltage across it.",
        keywords=("parallel", "same", "voltage", "branch", "circuit"),
        related_formula_ids=("parallel_two_resistance", "ohm_current"),
    ),
    LawStatement(
        id="parallel_currents_add",
        topic="circuits",
        statement="In a parallel circuit, total current equals the sum of branch currents.",
        keywords=("parallel", "current", "sum", "branch", "kirchhoff"),
        related_formula_ids=("parallel_two_resistance", "ohm_current"),
    ),
    LawStatement(
        id="series_capacitor_charge_same",
        topic="capacitors",
        statement="In a series connection of capacitors, each capacitor carries the same magnitude of charge.",
        keywords=("series", "capacitor", "same", "charge"),
        related_formula_ids=("series_capacitance", "capacitor_charge"),
    ),
    LawStatement(
        id="parallel_capacitor_voltage_same",
        topic="capacitors",
        statement="In a parallel connection of capacitors, each capacitor has the same voltage across it.",
        keywords=("parallel", "capacitor", "same", "voltage"),
        related_formula_ids=("parallel_capacitance", "capacitor_voltage"),
    ),
    LawStatement(
        id="parallel_capacitance_adds",
        topic="capacitors",
        statement="Capacitances in parallel add directly.",
        keywords=("parallel", "capacitance", "add", "equivalent"),
        related_formula_ids=("parallel_capacitance",),
    ),
    LawStatement(
        id="series_capacitance_reciprocal",
        topic="capacitors",
        statement="Capacitances in series add as reciprocals.",
        keywords=("series", "capacitance", "reciprocal", "equivalent"),
        related_formula_ids=("series_capacitance",),
    ),
    LawStatement(
        id="capacitor_energy_equivalences",
        topic="capacitor_energy",
        statement="Capacitor energy can be written as 1/2 C V^2, 1/2 QV, or Q^2/(2C).",
        keywords=("capacitor", "energy", "charge", "voltage"),
        related_formula_ids=("capacitor_energy", "capacitor_energy_qv", "capacitor_energy_q2c"),
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
        id="electric_force_on_charge",
        topic="electrostatics",
        statement="A charge in an electric field experiences a force F = qE.",
        keywords=("electric", "force", "field", "charge"),
        related_formula_ids=("electric_force_field",),
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
    LawStatement(
        id="air_as_vacuum",
        topic="electrostatics",
        statement="For standard school electrostatics problems, air is usually treated as vacuum unless a dielectric is stated.",
        keywords=("air", "vacuum", "dielectric", "electrostatics", "approximation"),
        related_formula_ids=("coulomb_force", "electric_field_charge"),
    ),
    LawStatement(
        id="equipotential_no_work",
        topic="electrostatics",
        statement="No work is done in moving a charge along an equipotential surface.",
        keywords=("equipotential", "work", "potential", "surface"),
        related_formula_ids=("electric_potential_energy_change",),
    ),
    LawStatement(
        id="gauss_symmetry_rule",
        topic="electrostatics",
        statement="Gauss law is most useful when the charge distribution has spherical, cylindrical, or planar symmetry.",
        keywords=("gauss", "law", "symmetry", "electric", "field", "flux"),
        related_formula_ids=("electric_field_charge", "electric_field_uniform"),
    ),
    LawStatement(
        id="ohmic_conditions",
        topic="circuits",
        statement="Ohm law applies to an ohmic conductor only when physical conditions such as temperature remain constant.",
        keywords=("ohm", "ohmic", "temperature", "constant", "conditions"),
        related_formula_ids=("ohm_voltage", "ohm_current", "ohm_resistance"),
    ),
    LawStatement(
        id="capacitor_current_leads",
        topic="alternating_current",
        statement="In a pure capacitor, current leads voltage by 90 degrees in sinusoidal steady state.",
        keywords=("capacitor", "current", "leads", "voltage", "phase", "ac"),
        related_formula_ids=("capacitive_reactance",),
    ),
    LawStatement(
        id="inductor_current_lags",
        topic="alternating_current",
        statement="In a pure inductor, current lags voltage by 90 degrees in sinusoidal steady state.",
        keywords=("inductor", "current", "lags", "voltage", "phase", "ac"),
        related_formula_ids=("inductive_reactance",),
    ),
    LawStatement(
        id="diode_one_way_rule",
        topic="semiconductor",
        statement="A diode conducts readily when forward-biased and blocks current strongly when reverse-biased until breakdown.",
        keywords=("diode", "forward", "reverse", "bias", "rectification"),
        related_formula_ids=("diode_series_resistor_current",),
    ),
    LawStatement(
        id="transistor_regions",
        topic="semiconductor",
        statement="A transistor used as a switch is driven between cutoff and saturation; as an amplifier it is biased in an approximately linear active region.",
        keywords=("transistor", "switch", "amplifier", "cutoff", "saturation", "active"),
        related_formula_ids=("transistor_current_gain_bjt", "transistor_collector_load_voltage"),
    ),
)


SYMBOLS: tuple[SymbolDefinition, ...] = (
    SymbolDefinition("A", "area or cross-sectional area", "m^2", ("area",)),
    SymbolDefinition("A_v", "voltage gain", "dimensionless", ("gain",)),
    SymbolDefinition("C", "capacitance", "F", ("capacitance", "capacitor")),
    SymbolDefinition("Ceq", "equivalent capacitance", "F", ("C_eq",)),
    SymbolDefinition("d", "separation distance", "m", ("distance", "spacing")),
    SymbolDefinition("E", "electric field strength", "N/C or V/m", ("field",)),
    SymbolDefinition("emf", "electromotive force or open-circuit source voltage", "V", ("epsilon", "ε")),
    SymbolDefinition("eps0", "vacuum permittivity", "F/m", ("epsilon0", "ε0")),
    SymbolDefinition("eps_r", "relative permittivity", "dimensionless", ("epsilon_r", "εr", "dielectric constant")),
    SymbolDefinition("f", "frequency", "Hz", ("frequency",)),
    SymbolDefinition("F", "force", "N", ("force",)),
    SymbolDefinition("I", "electric current", "A", ("current",)),
    SymbolDefinition("I_B", "base current in a BJT", "A", ("base current",)),
    SymbolDefinition("I_C", "collector current in a BJT", "A", ("collector current",)),
    SymbolDefinition("k", "Coulomb constant", "N m^2/C^2", ("coulomb constant",)),
    SymbolDefinition("L", "inductance", "H", ("inductance",)),
    SymbolDefinition("omega", "angular frequency", "rad/s", ("ω",)),
    SymbolDefinition("P", "power", "W", ("power",)),
    SymbolDefinition("Q", "charge", "C", ("charge",)),
    SymbolDefinition("q", "test charge or moving charge", "C", ("charge",)),
    SymbolDefinition("R", "resistance", "Ω", ("resistance", "resistor")),
    SymbolDefinition("Req", "equivalent resistance", "Ω", ("R_eq",)),
    SymbolDefinition("r", "radial distance or separation", "m", ("distance", "separation")),
    SymbolDefinition("r_int", "internal resistance of a source", "Ω", ("internal resistance",)),
    SymbolDefinition("rho", "resistivity", "Ω m", ("ρ",)),
    SymbolDefinition("t", "time", "s", ("time",)),
    SymbolDefinition("T", "period", "s", ("period",)),
    SymbolDefinition("tau", "RC time constant", "s", ("τ", "time constant")),
    SymbolDefinition("U", "energy or electric potential energy", "J", ("energy", "work")),
    SymbolDefinition("V", "potential difference or voltage", "V", ("voltage", "potential")),
    SymbolDefinition("V0", "initial or peak voltage depending on context", "V", ("initial voltage", "peak voltage")),
    SymbolDefinition("V_C", "capacitor voltage", "V", ("capacitor voltage",)),
    SymbolDefinition("V_D", "diode forward voltage drop", "V", ("diode drop",)),
    SymbolDefinition("V_rms", "root-mean-square voltage", "V", ("rms voltage",)),
    SymbolDefinition("X_C", "capacitive reactance", "Ω", ("capacitive reactance", "XC")),
    SymbolDefinition("X_L", "inductive reactance", "Ω", ("inductive reactance", "XL")),
    SymbolDefinition("Z", "impedance magnitude", "Ω", ("impedance",)),
    SymbolDefinition("beta", "BJT current gain", "dimensionless", ("β", "h_FE")),
    SymbolDefinition("delta_U", "change in electric potential energy", "J", ("ΔU",)),
    SymbolDefinition("delta_V", "potential difference or change in potential", "V", ("ΔV",)),
)


CONSTANTS: tuple[PhysicalConstant, ...] = (
    PhysicalConstant("eps0", "8.8541878188e-12", "F/m", "Vacuum permittivity.", ("epsilon0", "ε0")),
    PhysicalConstant("k", "8.9875517923e9", "N m^2/C^2", "Coulomb constant, equal to 1/(4*pi*eps0)."),
    PhysicalConstant("e", "1.602176634e-19", "C", "Elementary charge magnitude."),
    PhysicalConstant("ln2", "0.69314718056", "dimensionless", "Natural logarithm of 2, useful for RC half-time.", ("ln 2",)),
)


WORKED_EXAMPLES: tuple[WorkedExample, ...] = (
    WorkedExample(
        "example_coulomb_force",
        "electrostatics",
        "Force between two small charges",
        "Two charges are separated in air. Find the electrostatic force.",
        ("q1 and q2 in coulombs", "r in metres", "k = 8.99e9 N m^2/C^2"),
        ("Use F = k*abs(q1*q2)/r^2 for magnitude.", "Use charge signs to decide attraction or repulsion."),
        ("Force magnitude in newtons.",),
        ("coulomb_force",),
        ("coulomb_superposition", "air_as_vacuum"),
    ),
    WorkedExample(
        "example_uniform_field",
        "electrostatics",
        "Uniform electric field between plates",
        "Parallel plates have a potential difference and separation. Find field and force on a charge.",
        ("V in volts", "d in metres", "q in coulombs if force is needed"),
        ("Use E = V/d.", "If force is requested, use F = qE."),
        ("Field in V/m or N/C.",),
        ("electric_field_uniform", "electric_force_field"),
    ),
    WorkedExample(
        "example_capacitor_charge_sharing",
        "capacitors",
        "Disconnected charged capacitor connected to an uncharged capacitor",
        "A charged capacitor is disconnected from a source and connected in parallel to another capacitor.",
        ("Initial C and V", "Second capacitance", "Initial charge on uncharged capacitor is zero"),
        ("Conserve total charge.", "Find parallel equivalent capacitance.", "Use final energy E = Q^2/(2*Ceq) if asked."),
        ("Final energy or final voltage after charge sharing.",),
        ("capacitor_charge", "parallel_capacitance", "capacitor_energy_q2c"),
        ("parallel_capacitor_voltage_same", "parallel_capacitance_adds"),
    ),
    WorkedExample(
        "example_internal_resistance",
        "circuits",
        "Cell with internal resistance",
        "A source with emf and internal resistance supplies an external resistor.",
        ("emf in volts", "r_int in ohms", "R in ohms"),
        ("Use I = emf/(R+r_int).", "Use V = emf - I*r_int for terminal voltage."),
        ("Current in amperes and terminal voltage in volts.",),
        ("internal_resistance_current", "internal_resistance_terminal_voltage"),
    ),
    WorkedExample(
        "example_rc_charging",
        "rc_circuit",
        "Charging or discharging a capacitor",
        "An RC circuit is observed after a time t.",
        ("R in ohms", "C in farads", "initial/final voltage", "t in seconds"),
        ("Find tau = R*C.", "Choose charging or discharging exponential.", "Substitute t/(R*C)."),
        ("Capacitor voltage, charge, or current at time t.",),
        ("rc_time_constant", "rc_charge_voltage", "rc_discharge_voltage"),
    ),
    WorkedExample(
        "example_ac_reactance",
        "alternating_current",
        "Reactance of a capacitor or inductor",
        "A pure capacitor or inductor is connected to a sinusoidal AC source.",
        ("f in hertz", "C in farads or L in henries", "V_rms if current is requested"),
        ("Compute X_C or X_L.", "Use I = V/Z or I = V/X for a pure component."),
        ("Reactance in ohms and current in amperes.",),
        ("capacitive_reactance", "inductive_reactance", "ac_current"),
    ),
    WorkedExample(
        "example_diode_series_resistor",
        "semiconductor",
        "Forward-biased diode with a series resistor",
        "A silicon diode and resistor are connected to a DC source.",
        ("Supply voltage V", "diode drop V_D, often about 0.7 V", "series resistance R"),
        ("Use I = (V - V_D)/R when forward-biased and V > V_D."),
        ("Current in amperes.",),
        ("diode_series_resistor_current",),
        ("diode_one_way_rule",),
    ),
    WorkedExample(
        "example_transistor_gain",
        "semiconductor",
        "BJT gain estimate",
        "A BJT has base current and current gain with a collector load resistor.",
        ("beta", "I_B", "R_C", "supply V"),
        ("Use I_C = beta*I_B.", "Use Vout = V - I_C*R_C for the collector node estimate."),
        ("Collector current and output voltage.",),
        ("transistor_current_gain_bjt", "transistor_collector_load_voltage"),
        ("transistor_regions",),
    ),
)


DIAGRAMS: tuple[Diagram, ...] = (
    Diagram(
        "diagram_resistor_topologies",
        "Series and parallel resistor topologies",
        "flowchart LR\n  A[Source] --> R1[R1] --> R2[R2] --> B[Return]\n  C[Source] --> J[Junction]\n  J --> P1[R1]\n  J --> P2[R2]\n  P1 --> K[Return]\n  P2 --> K",
        ("series_resistance", "parallel_two_resistance"),
        ("series_current_same", "parallel_voltage_same"),
    ),
    Diagram(
        "diagram_capacitor_topologies",
        "Series and parallel capacitor topologies",
        "flowchart LR\n  A[Supply] --> C1[C1] --> C2[C2] --> B[Return]\n  S[Supply] --> J[Node]\n  J --> P1[C1]\n  J --> P2[C2]\n  P1 --> K[Return]\n  P2 --> K",
        ("series_capacitance", "parallel_capacitance"),
        ("series_capacitor_charge_same", "parallel_capacitor_voltage_same"),
    ),
    Diagram(
        "diagram_rc_switching",
        "RC charge and discharge paths",
        "flowchart LR\n  V[Supply V0] --> SW{Switch}\n  SW -->|charge| R[R]\n  R --> C[C]\n  SW -->|discharge| RD[R]\n  RD --> C\n  C --> G[Return]",
        ("rc_time_constant", "rc_charge_voltage", "rc_discharge_voltage"),
    ),
    Diagram(
        "diagram_diode_rectifier",
        "Diode rectifier topology",
        "flowchart LR\n  AC[AC source] --> D[Diode]\n  D --> L[Load]\n  L --> R[Return]",
        ("diode_series_resistor_current",),
        ("diode_one_way_rule",),
    ),
    Diagram(
        "diagram_transistor_switch",
        "Simple transistor switch topology",
        "flowchart LR\n  IN[Input] --> B[Control]\n  VCC[Supply] --> LOAD[Load]\n  LOAD --> T[Transistor]\n  T --> G[Ground]",
        ("transistor_current_gain_bjt", "transistor_collector_load_voltage"),
        ("transistor_regions",),
    ),
)


CATEGORIES: tuple[Category, ...] = (
    Category(
        "electrostatics",
        "Electrostatics",
        "Forces, fields, potentials, energy, and common school electrostatic approximations.",
        ("coulomb_force", "electric_field_force", "electric_field_charge", "electric_field_uniform", "electric_potential", "electric_potential_energy", "electric_potential_energy_change"),
        ("air_as_vacuum", "coulomb_superposition", "electric_field_superposition", "equipotential_no_work", "gauss_symmetry_rule"),
        ("example_coulomb_force", "example_uniform_field"),
        ("E", "F", "q", "Q", "k", "r", "V", "U", "delta_U", "delta_V"),
        ("k", "eps0", "e"),
    ),
    Category(
        "capacitors",
        "Capacitors",
        "Capacitance, capacitor energy, series/parallel combinations, dielectrics, and charge sharing.",
        ("capacitance", "capacitor_charge", "capacitor_voltage", "capacitor_energy", "capacitor_energy_qv", "capacitor_energy_q2c", "parallel_capacitance", "series_capacitance", "parallel_plate_capacitance", "dielectric_capacitance"),
        ("series_capacitor_charge_same", "parallel_capacitor_voltage_same", "capacitor_energy_equivalences"),
        ("example_capacitor_charge_sharing",),
        ("C", "Ceq", "Q", "V", "A", "d", "eps0", "eps_r"),
        ("eps0",),
        ("diagram_capacitor_topologies",),
    ),
    Category(
        "circuits",
        "DC circuits",
        "Ohm law, power, resistor networks, dividers, Kirchhoff laws, and internal resistance.",
        ("ohm_voltage", "ohm_current", "ohm_resistance", "power_vi", "power_i2r", "power_v2r", "series_resistance", "parallel_two_resistance", "voltage_divider", "current_divider", "resistivity", "internal_resistance_current", "internal_resistance_terminal_voltage"),
        ("kirchhoff_current_law", "kirchhoff_voltage_law", "series_current_same", "parallel_voltage_same", "ohmic_conditions"),
        ("example_internal_resistance",),
        ("V", "I", "R", "Req", "P", "emf", "r_int", "rho"),
        (),
        ("diagram_resistor_topologies",),
    ),
    Category(
        "rc_circuit",
        "DC circuits with capacitors",
        "First-order RC time constants and charging/discharging exponentials.",
        ("rc_time_constant", "rc_charge_voltage", "rc_charge_current", "rc_discharge_voltage", "rc_discharge_charge", "rc_half_time"),
        (),
        ("example_rc_charging",),
        ("R", "C", "tau", "t", "V0", "V_C", "Q"),
        ("ln2",),
        ("diagram_rc_switching",),
    ),
    Category(
        "alternating_current",
        "Simple AC",
        "Sinusoidal RMS values, reactance, impedance, phase rules, and resistive AC power.",
        ("inductive_reactance", "capacitive_reactance", "series_rlc_impedance", "ac_current", "ac_real_power_vzr", "ac_rms_voltage", "ac_rms_current", "angular_frequency", "frequency_period"),
        ("capacitor_current_leads", "inductor_current_lags"),
        ("example_ac_reactance",),
        ("f", "T", "omega", "V_rms", "X_C", "X_L", "Z", "I"),
    ),
    Category(
        "semiconductor",
        "Semiconductor circuit basics",
        "Approximate school-level diode and transistor circuit relations.",
        ("diode_series_resistor_current", "amplifier_voltage_gain", "transistor_current_gain_bjt", "transistor_collector_load_voltage"),
        ("diode_one_way_rule", "transistor_regions"),
        ("example_diode_series_resistor", "example_transistor_gain"),
        ("V_D", "I", "A_v", "beta", "I_B", "I_C"),
        (),
        ("diagram_diode_rectifier", "diagram_transistor_switch"),
    ),
)


KnowledgeItem = Formula | LawStatement | SymbolDefinition | PhysicalConstant | WorkedExample | Category | Diagram

FORMULA_INDEX = {item.id: item for item in FORMULAS}
LAW_INDEX = {item.id: item for item in LAW_STATEMENTS}
SYMBOL_INDEX = {item.symbol: item for item in SYMBOLS}
CONSTANT_INDEX = {item.symbol: item for item in CONSTANTS}
EXAMPLE_INDEX = {item.id: item for item in WORKED_EXAMPLES}
CATEGORY_INDEX = {item.id: item for item in CATEGORIES}
DIAGRAM_INDEX = {item.id: item for item in DIAGRAMS}


def get_formula(formula_id: str) -> Formula | None:
    return FORMULA_INDEX.get(formula_id)


def get_law(law_id: str) -> LawStatement | None:
    return LAW_INDEX.get(law_id)


def get_symbol(symbol: str) -> SymbolDefinition | None:
    return SYMBOL_INDEX.get(symbol)


def get_constant(symbol: str) -> PhysicalConstant | None:
    return CONSTANT_INDEX.get(symbol)


def get_example(example_id: str) -> WorkedExample | None:
    return EXAMPLE_INDEX.get(example_id)


def get_category(category_id: str) -> Category | None:
    return CATEGORY_INDEX.get(category_id)


def get_diagram(diagram_id: str) -> Diagram | None:
    return DIAGRAM_INDEX.get(diagram_id)


def _ensure_unique_ids(items: Iterable[object], attr: str, label: str) -> None:
    seen: set[str] = set()
    for item in items:
        value = getattr(item, attr)
        if value in seen:
            raise ValueError(f"Duplicate {label} id detected: {value}")
        seen.add(value)


def _validate_references() -> None:
    formula_ids = set(FORMULA_INDEX)
    law_ids = set(LAW_INDEX)
    symbol_ids = set(SYMBOL_INDEX)
    constant_ids = set(CONSTANT_INDEX)
    example_ids = set(EXAMPLE_INDEX)
    diagram_ids = set(DIAGRAM_INDEX)

    for law in LAW_STATEMENTS:
        for formula_id in law.related_formula_ids:
            if formula_id not in formula_ids:
                raise ValueError(f"Law {law.id!r} references unknown formula id {formula_id!r}")
        for diagram_id in law.diagram_ids:
            if diagram_id not in diagram_ids:
                raise ValueError(f"Law {law.id!r} references unknown diagram id {diagram_id!r}")

    for example in WORKED_EXAMPLES:
        for formula_id in example.related_formula_ids:
            if formula_id not in formula_ids:
                raise ValueError(f"Example {example.id!r} references unknown formula id {formula_id!r}")
        for law_id in example.related_law_ids:
            if law_id not in law_ids:
                raise ValueError(f"Example {example.id!r} references unknown law id {law_id!r}")

    for diagram in DIAGRAMS:
        for formula_id in diagram.related_formula_ids:
            if formula_id not in formula_ids:
                raise ValueError(f"Diagram {diagram.id!r} references unknown formula id {formula_id!r}")
        for law_id in diagram.related_law_ids:
            if law_id not in law_ids:
                raise ValueError(f"Diagram {diagram.id!r} references unknown law id {law_id!r}")

    for category in CATEGORIES:
        for formula_id in category.formula_ids:
            if formula_id not in formula_ids:
                raise ValueError(f"Category {category.id!r} references unknown formula id {formula_id!r}")
        for law_id in category.law_ids:
            if law_id not in law_ids:
                raise ValueError(f"Category {category.id!r} references unknown law id {law_id!r}")
        for example_id in category.example_ids:
            if example_id not in example_ids:
                raise ValueError(f"Category {category.id!r} references unknown example id {example_id!r}")
        for symbol_id in category.symbol_ids:
            if symbol_id not in symbol_ids:
                raise ValueError(f"Category {category.id!r} references unknown symbol id {symbol_id!r}")
        for constant_id in category.constant_ids:
            if constant_id not in constant_ids:
                raise ValueError(f"Category {category.id!r} references unknown constant id {constant_id!r}")
        for diagram_id in category.diagram_ids:
            if diagram_id not in diagram_ids:
                raise ValueError(f"Category {category.id!r} references unknown diagram id {diagram_id!r}")


def _validate() -> None:
    _ensure_unique_ids(FORMULAS, "id", "formula")
    _ensure_unique_ids(LAW_STATEMENTS, "id", "law")
    _ensure_unique_ids(SYMBOLS, "symbol", "symbol")
    _ensure_unique_ids(CONSTANTS, "symbol", "constant")
    _ensure_unique_ids(WORKED_EXAMPLES, "id", "example")
    _ensure_unique_ids(CATEGORIES, "id", "category")
    _ensure_unique_ids(DIAGRAMS, "id", "diagram")
    _validate_references()


_validate()
