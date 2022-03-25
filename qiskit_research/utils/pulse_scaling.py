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

from typing import Iterable, List, Optional, Union

from .passes import BindParameters, CombineRuns, ECRCalibrationBuilder, RZXtoEchoedCR


from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.qasm import pi
from qiskit.providers.backend import Backend
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.passes import TemplateOptimization
from qiskit.transpiler.passes import CXCancellation, Optimize1qGatesDecomposition
from qiskit.transpiler.passes.calibration.rzx_templates import rzx_templates

BASIS_GATES = ['sx', 'rz', 'rzx', 'cx']

def scale_cr_pulses(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    backend: Backend,
    param_bind: Optional[dict] = None,
) -> Union[QuantumCircuit, List[QuantumCircuit]]:
    """
    Scale circuits using Pulse scaling technique from
    http://arxiv.org/abs/2012.11660
    """
    templates = rzx_templates()
    pass_manager = PassManager(
        list(cr_scaling_passes(backend, templates, param_bind=param_bind))
    )
    return pass_manager.run(circuits)

def cr_scaling_passes(
    backend: Backend,
    templates: List[QuantumCircuit],
    param_bind: Optional[dict] = None,
) -> Iterable[BasePass]:
    """Yields transpilation passes for CR pulse scaling."""

    inst_sched_map = backend.defaults().instruction_schedule_map
    yield TemplateOptimization(**templates)
    yield CombineRuns(['rzx'])
    # pauli twirl here
    yield RZXtoEchoedCR(inst_sched_map)
    yield Optimize1qGatesDecomposition(BASIS_GATES)
    yield CXCancellation()
    yield CombineRuns(['rz'])
    if param_bind is not None:
        yield BindParameters(param_bind)
        yield Optimize1qGatesDecomposition(BASIS_GATES)
        yield CXCancellation()
        yield ECRCalibrationBuilder(backend)
