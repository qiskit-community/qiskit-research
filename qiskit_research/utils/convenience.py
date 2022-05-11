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

from typing import Any, Iterable, List, Optional, Union

from qiskit.circuit import QuantumCircuit
from qiskit.providers.backend import Backend
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.passes import (
    ALAPSchedule,
)
from qiskit.transpiler.passes.calibration.rzx_templates import rzx_templates
from qiskit_research.utils import (
    PauliTwirl,
    dynamical_decoupling_passes,
    cr_scaling_passes,
)
from qiskit_research.utils.gates import (
    add_pulse_calibrations,
)


def add_dynamical_decoupling(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    backend: Backend,
    dd_str: str,
    scheduler: BasePass = ALAPSchedule,
    add_pulse_cals: bool = False,
) -> Union[QuantumCircuit, List[QuantumCircuit]]:
    """Add dynamical decoupling sequences and calibrations to circuits.

    Adds dynamical decoupling sequences and the calibrations necessary
    to run them on an IBM backend.
    """
    pass_manager = PassManager(
        list(dynamical_decoupling_passes(backend, dd_str, scheduler))
    )
    circuits_dd = pass_manager.run(circuits)
    if add_pulse_cals:
        add_pulse_calibrations(circuits_dd, backend)
    return circuits_dd


def add_pauli_twirls(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    num_twirled_circuits: int = 1,
    gates_to_twirl: Optional[Iterable[str]] = None,
    seed: Any = None,
) -> Union[List[QuantumCircuit], List[List[QuantumCircuit]]]:
    """Add Pauli twirls to circuits.

    Args:
        circuits: Circuit or list of circuits to be twirled.
        num_twirled_circuits: Number of twirled circuits to return for each input circuit.
        gates_to_twirl: Names of gates to twirl. The default behavior is to twirl all
            supported gates.
        seed: Seed for the pseudorandom number generator.

    Returns:
        If the input is a single circuit, then a list of circuits is returned.
        If the input is a list of circuit, then a list of lists of circuits is returned.
    """
    pass_manager = PassManager(PauliTwirl(gates_to_twirl=gates_to_twirl, seed=seed))
    if isinstance(circuits, QuantumCircuit):
        return [pass_manager.run(circuits) for _ in range(num_twirled_circuits)]
    return [
        [pass_manager.run(circuit) for _ in range(num_twirled_circuits)]
        for circuit in circuits
    ]


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
    inst_sched_map = backend.defaults().instruction_schedule_map
    channel_map = backend.configuration().qubit_channel_mapping

    pass_manager = PassManager(
        list(
            cr_scaling_passes(
                inst_sched_map, channel_map, templates, param_bind=param_bind
            )
        )
    )
    return pass_manager.run(circuits)
