# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Cost functions."""

from __future__ import annotations

from typing import List, Tuple

from qiskit.circuit import QuantumCircuit
from qiskit.providers.backend import Backend
from qiskit.qasm import pi

import numpy as np


def cost_func_scaled_cr(
    circ: QuantumCircuit,
    layouts: List[List[int]],
    backend: Backend,
) -> List[Tuple[List[int], float]]:
    """
    A custom cost function that includes T1 and T2 computed during idle periods,
    for an already transpiled and scheduled circuit circ

    Parameters:
        circ (QuantumCircuit): scheduled circuit of interest
        backend (IBMQBackend): An IBM Quantum backend instance

    Returns:
        float: error
    """
    out = []
    props = backend.properties()
    inst_sched_map = backend.defaults().instruction_schedule_map
    dt = backend.configuration().dt
    num_qubits = backend.configuration().num_qubits
    t1s = [props.qubit_property(qq, "T1")[0] for qq in range(num_qubits)]
    t2s = [props.qubit_property(qq, "T2")[0] for qq in range(num_qubits)]

    error = 0.0
    fid = 1.0
    touched = set()
    for layout in layouts:
        for item in circ.data:
            if item[0].num_qubits == 2:
                q0 = circ.find_bit(item[1][0]).index
                q1 = circ.find_bit(item[1][1]).index
                if inst_sched_map.has("cx", qubits=[q0, q1]):
                    basis_2q_error = props.gate_error("cx", [q0, q1])
                elif inst_sched_map.has("ecr", qubits=[q0, q1]):
                    basis_2q_error = props.gate_error("ecr", [q0, q1])
                elif inst_sched_map.has("ecr", qubits=[q1, q0]):
                    basis_2q_error = props.gate_error("ecr", [q1, q0])
                else:
                    print(
                        f"{backend.name} missing 2Q gate between qubit pair ({q0}, {q1})"
                    )

            if item[0].name == "cx":
                fid *= 1 - basis_2q_error
                touched.add(q0)
                touched.add(q1)

            # if it is a scaled pulse derived from cx
            elif item[0].name == "rzx":
                cr_error = np.abs(float(item[0].params[0]) / (pi / 2)) * basis_2q_error

                # assumes control qubit is actually control for cr
                echo_error = props.gate_error("x", q0)

                fid *= 1 - max(cr_error, 2 * echo_error)
            elif item[0].name == "secr":
                cr_error = (
                    np.abs((float(item[0].params[0])) / (pi / 2)) * basis_2q_error
                )

                # assumes control qubit is actually control for cr
                echo_error = props.gate_error("x", q0)

                fid *= 1 - max(cr_error, echo_error)

            elif item[0].name in ["sx", "x"]:
                q0 = circ.find_bit(item[1][0]).index
                fid *= 1 - props.gate_error(item[0].name, q0)
                touched.add(q0)

            # if it is a dynamical decoupling pulse
            elif item[0].name in ["xp", "xm", "y", "yp", "ym"]:
                q0 = circ.find_bit(item[1][0]).index
                fid *= 1 - props.gate_error("x", q0)
                touched.add(q0)

            elif item[0].name == "measure":
                q0 = circ.find_bit(item[1][0]).index
                fid *= 1 - props.readout_error(q0)
                touched.add(q0)

            elif item[0].name == "delay":
                q0 = circ.find_bit(item[1][0]).index
                # Ignore delays that occur before gates
                # This assumes you are in ground state and errors
                # do not occur.
                if q0 in touched:
                    time = item[0].duration * dt
                    fid *= 1 - idle_error(time, t1s[q0], t2s[q0])

        error = 1 - fid
        out.append((layout, error))
    return out


def idle_error(
    time: float,
    t1: float,
    t2: float,
) -> float:
    """Compute the approx. idle error from T1 and T2
    Parameters:
        time (float): Delay time in sec
        t1 (float): T1 time in sec
        t2, (float): T2 time in sec
    Returns:
        float: Idle error
    """
    t2 = min(t1, t2)
    rate1 = 1 / t1
    rate2 = 1 / t2
    p_reset = 1 - np.exp(-time * rate1)
    p_z = (1 - p_reset) * (1 - np.exp(-time * (rate2 - rate1))) / 2
    return p_z + p_reset


def cost_func_ecr(
    circ: QuantumCircuit,
    layouts: List[List[int]],
    backend: Backend,
) -> List[Tuple[List[int], float]]:
    """
    A custom cost function that includes ECR gates in either direction

    Parameters:
        circ (QuantumCircuit): circuit of interest
        layouts (list of lists): List of specified layouts
        backend (IBMQBackend): An IBM Quantum backend instance

    Returns:
        list: Tuples of layout and cost
    """
    out = []
    inst_sched_map = backend.defaults().instruction_schedule_map
    props = backend.properties()
    for layout in layouts:
        error = 0.0
        fid = 1.0
        touched = set()
        for item in circ.data:
            if item[0].name == "ecr":
                q0 = layout[circ.find_bit(item[1][0]).index]
                q1 = layout[circ.find_bit(item[1][1]).index]
                if inst_sched_map.has("ecr", [q0, q1]):
                    fid *= 1 - props.gate_error("ecr", [q0, q1])
                elif inst_sched_map.has("ecr", [q1, q0]):
                    fid *= 1 - props.gate_error("ecr", [q1, q0])
                else:
                    print(
                        f"{backend.name} does not support {item[0].name} \
                        for qubit pair ({q0}, {q1})"
                    )
                touched.add(q0)
                touched.add(q1)

            elif item[0].name in ["sx", "x"]:
                q0 = layout[circ.find_bit(item[1][0]).index]
                fid *= 1 - props.gate_error(item[0].name, q0)
                touched.add(q0)

            elif item[0].name == "measure":
                q0 = layout[circ.find_bit(item[1][0]).index]
                fid *= 1 - props.readout_error(q0)
                touched.add(q0)

        error = 1 - fid
        out.append((layout, error))
    return out
