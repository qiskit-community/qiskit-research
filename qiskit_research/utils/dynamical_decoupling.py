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

from typing import Optional, Union
from qiskit.qasm import pi
import numpy

from qiskit import pulse
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.circuit.library import XGate, YGate
from qiskit.pulse import DriveChannel
from qiskit.transpiler import InstructionDurations, PassManager
from qiskit.transpiler.passes import ALAPSchedule, ASAPSchedule, DynamicalDecoupling

def add_dd_calibrations(qc: QuantumCircuit, backend, dd_str: str, sched_method=None):
    if sched_method is None:
        sched_method = 'alap'

    qc_dd = add_dd_sequence(qc, backend, dd_str, sched_method)
    add_dd_pulse_calibrations(qc_dd, backend)

    return qc_dd

def get_dd_sequence(dd_str: str): # -> List(Gate):
    r"""
    We will define standard dynamical decoupling sequences via the
    strings:

        'X2': X-X
        'X2pm': Xp-Xm
        'XY4': X-Y-X-Y
        'XY4pm': Xp-Yp-Xm-Ym
        'XY8': X-Y-X-Y-Y-X-Y-X
        'XY8pm': Xp-Yp-Xm-Ym-Ym-Xm-Yp-Xp
    """

    if dd_str == 'X2':
        return [XGate(), XGate()]
    elif dd_str == 'X2pm':
        return [XpGate(), XmGate()]
    elif dd_str == 'XY4':
        return [XGate(), YGate(), XGate(), YGate()]
    elif dd_str == 'XY4pm':
        return [XpGate(), YpGate(), XmGate(), YmGate()]
    elif dd_str == 'XY8':
        return [XGate(), YGate(), XGate(), YGate(), YGate(), XGate(), YGate(), XGate()]
    elif dd_str == 'XY8pm':
        return [XpGate(), YpGate(), XmGate(), YmGate(), YmGate(), XmGate(), YpGate(), XpGate()]
    else:
        return []

def get_timing(backend) -> InstructionDurations:
    inst_durs = []
    inst_sched_map = backend.defaults().instruction_schedule_map
    num_qubits = backend.configuration().num_qubits

    # single qubit gates
    for qubit in range(num_qubits):
        for inst_str in inst_sched_map.qubit_instructions(qubits=[qubit]):
            inst = inst_sched_map.get(inst_str, qubits=[qubit])
            inst_durs.append((inst_str, qubit, inst.duration))

            # create DD pulses from CR echo 'x' pulse
            if inst_str == 'x':
                inst_durs.append(('xp', qubit, inst.duration))
                inst_durs.append(('xm', qubit, inst.duration))
                inst_durs.append(('y', qubit, inst.duration))
                inst_durs.append(('yp', qubit, inst.duration))
                inst_durs.append(('ym', qubit, inst.duration))

    # two qubit gates
    for qc in range(num_qubits):
        for qt in range(num_qubits):
            for inst_str in inst_sched_map.qubit_instructions(qubits=[qc, qt]):
                inst = inst_sched_map.get(inst_str, qubits=[qc, qt])
                inst_durs.append((inst_str, [qc, qt], inst.duration))

    return InstructionDurations(inst_durs)

def add_dd_sequence(qc: QuantumCircuit, backend, dd_str: str, sched_method: str) -> QuantumCircuit:
    durations = get_timing(backend)
    sequence = get_dd_sequence(dd_str)

    if sched_method == 'asap':
        pm = PassManager([ASAPSchedule(durations),
            DynamicalDecoupling(durations, sequence)])
    elif sched_method == 'alap':
        pm = PassManager([ALAPSchedule(durations),
            DynamicalDecoupling(durations, sequence)])

    return pm.run(qc)

def add_dd_pulse_calibrations(qc: QuantumCircuit, backend):
    inst_sched_map = backend.defaults().instruction_schedule_map
    num_qubits = backend.configuration().num_qubits

    for qubit in range(num_qubits):
        with pulse.build('xp gate for qubit '+str(qubit)) as sched:
            # def of XpGate() in terms of XGate()
            x_sched = inst_sched_map.get('x', qubits=[qubit])
            pulse.call(x_sched)

            # for each DD sequence with a XpGate() in it
            for circ in [qc]:
                circ.add_calibration('xp', [qubit], sched)

        with pulse.build('xm gate for qubit '+str(qubit)) as sched:
            # def of XmGate() in terms of XGate() and amplitude inversion
            x_sched = inst_sched_map.get('x', qubits=[qubit])
            x_pulse = x_sched.instructions[0][1].pulse
            x_pulse._amp = -x_pulse.amp # bad form
            pulse.play(x_pulse, DriveChannel(qubit))

            # for each DD sequence with a XmGate() in it
            for circ in [qc]:
                circ.add_calibration('xm', [qubit], sched)

        with pulse.build('y gate for qubit '+str(qubit)) as sched:
            # def of YGate() in terms of XGate() and phase_offset
            with pulse.phase_offset(pi/2, DriveChannel(qubit)):
                x_sched = inst_sched_map.get('x', qubits=[qubit])
                pulse.call(x_sched)

            # for each DD sequence with a YGate() in it
            for circ in [qc]:
                circ.add_calibration('y', [qubit], sched)

        with pulse.build('yp gate for qubit '+str(qubit)) as sched:
            # def of YpGate() in terms of XGate() and phase_offset
            with pulse.phase_offset(pi/2, DriveChannel(qubit)):
                x_sched = inst_sched_map.get('x', qubits=[qubit])
                pulse.call(x_sched)

            # for each DD sequence with a YpGate() in it
            for circ in [qc]:
                circ.add_calibration('yp', [qubit], sched)

        with pulse.build('ym gate for qubit '+str(qubit)) as sched:
            # def of YGate() in terms of XGate() and phase_offset
            with pulse.phase_offset(-pi/2, DriveChannel(qubit)):
                x_sched = inst_sched_map.get('x', qubits=[qubit])
                x_pulse = x_sched.instructions[0][1].pulse
                x_pulse._amp = -x_pulse.amp # bad form
                pulse.play(x_pulse, DriveChannel(qubit))

            # for each DD sequence with a YmGate() in it
            for circ in [qc]:
                circ.add_calibration('ym', [qubit], sched)

class XpGate(Gate):
    r"""The single-qubit Pauli-X gate (:math:`\sigma_x`), implemented
    via RX(\pi).
    """

    def __init__(self, label: Optional[str] = None):
        """Create new Xp gate."""
        super().__init__("xp", 1, [], label=label)

    def _define(self):
        """
        gate xp a { u3(pi,0,pi) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u3 import U3Gate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U3Gate(pi, 0, pi), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        r"""Return inverted Xp gate (Xm)."""
        return XmGate()  # self-inverse

    def __array__(self, dtype=None):
        """Return a numpy.array for the Xp gate."""
        return numpy.array([[0, 1], [1, 0]], dtype=dtype)

class XmGate(Gate):
    r"""The single-qubit Pauli-X gate (:math:`\sigma_x`), implemented
    via RX(-\pi).
    """

    def __init__(self, label: Optional[str] = None):
        """Create new Xm gate."""
        super().__init__("xm", 1, [], label=label)

    def _define(self):
        """
        gate xm a { u3(pi,0,pi) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u3 import U3Gate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U3Gate(pi, 0, pi), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        r"""Return inverted Xm gate (Xp)."""
        return XpGate()  # self-inverse

    def __array__(self, dtype=None):
        """Return a numpy.array for the X gate."""
        return numpy.array([[0, 1], [1, 0]], dtype=dtype)

class YpGate(Gate):
    r"""The single-qubit Pauli-Y gate (:math:`\sigma_y`), implemented
    via RY(\pi).
    """

    def __init__(self, label: Optional[str] = None):
        """Create new Yp gate."""
        super().__init__("yp", 1, [], label=label)

    def _define(self):
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u3 import U3Gate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U3Gate(pi, pi / 2, pi / 2), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        r"""Return inverted Yp gate (:math:`Y{\dagger} = Y`)"""
        return YmGate()  # self-inverse

    def __array__(self, dtype=None):
        """Return a numpy.array for the Yp gate."""
        return numpy.array([[0, -1j], [1j, 0]], dtype=dtype)

class YmGate(Gate):
    r"""The single-qubit Pauli-Y gate (:math:`\sigma_y`), implemented
    via RY(-\pi).
    """

    def __init__(self, label: Optional[str] = None):
        """Create new Ym gate."""
        super().__init__("ym", 1, [], label=label)

    def _define(self):
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u3 import U3Gate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U3Gate(pi, pi / 2, pi / 2), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        r"""Return inverted Ym gate (:math:`Y{\dagger} = Y`)"""
        return YpGate()  # self-inverse

    def __array__(self, dtype=None):
        """Return a numpy.array for the Ym gate."""
        return numpy.array([[0, -1j], [1j, 0]], dtype=dtype)
