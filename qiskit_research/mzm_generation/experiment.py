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

import copy
import functools
import itertools
import math
from collections import namedtuple
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers import Backend
from qiskit_experiments.framework import BaseAnalysis, BaseExperiment, ExperimentData
from qiskit_nature.circuit.library import FermionicGaussianState
from qiskit_research.mzm_generation.utils import (
    kitaev_hamiltonian,
    measure_interaction_op,
)
from qiskit_research.utils.dynamical_decoupling import add_dynamical_decoupling

# TODO make this a JSON serializable dataclass when Aer supports it
# See https://github.com/Qiskit/qiskit-aer/issues/1435
CircuitParameters = namedtuple(
    "CircuitParameters",
    [
        "tunneling",
        "superconducting",
        "chemical_potential",
        "occupied_orbitals",
        "permutation",
        "measurement_label",
    ],
)


class KitaevHamiltonianExperiment(BaseExperiment):
    """Prepare and measure eigenstates of the Kitaev Hamiltonian."""

    def __init__(
        self,
        experiment_id: str,
        backend: Backend,
        readout_calibration_date: str,
        qubits: Sequence[int],
        tunneling_values: float,
        superconducting_values: float,
        chemical_potential_values: float,
        occupied_orbitals_list: Sequence[Tuple[int, ...]],
        dynamical_decoupling_sequence: Optional[str] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.readout_calibration_date = readout_calibration_date
        # TODO qubits should be set in parent class
        # see https://github.com/Qiskit/qiskit-experiments/issues/627
        self.qubits = qubits
        self.n_modes = len(qubits)
        self.tunneling_values = tunneling_values
        self.superconducting_values = superconducting_values
        self.chemical_potential_values = chemical_potential_values
        self.occupied_orbitals_list = occupied_orbitals_list
        self.dynamical_decoupling_sequence = dynamical_decoupling_sequence
        super().__init__(qubits=qubits, backend=backend)

    def _additional_metadata(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "readout_calibration_date": self.readout_calibration_date,
            "qubits": self.qubits,
            "tunneling_values": self.tunneling_values,
            "superconducting_values": self.superconducting_values,
            "chemical_potential_values": self.chemical_potential_values,
            "occupied_orbitals_list": self.occupied_orbitals_list,
            "dynamical_decoupling_sequence": self.dynamical_decoupling_sequence,
        }

    def circuits(self) -> List[QuantumCircuit]:
        return list(self._circuits())

    def _circuits(self) -> Iterable[QuantumCircuit]:
        for circuit_params in self.circuit_parameters():
            yield self.generate_circuit(circuit_params)

    def generate_circuit(self, circuit_params: CircuitParameters) -> QuantumCircuit:
        base_circuit = self._base_circuit(
            circuit_params.tunneling,
            circuit_params.superconducting,
            circuit_params.chemical_potential,
            circuit_params.occupied_orbitals,
            circuit_params.permutation,
        )
        circuit = measure_interaction_op(base_circuit, circuit_params.measurement_label)
        circuit.metadata = {"params": circuit_params}
        return circuit

    @functools.lru_cache
    def _base_circuit(
        self,
        tunneling: float,
        superconducting: float,
        chemical_potential: float,
        occupied_orbitals: Tuple[int, ...],
        permutation: Tuple[int, ...],
    ) -> QuantumCircuit:
        hamiltonian = kitaev_hamiltonian(
            self.n_modes,
            tunneling=tunneling,
            superconducting=superconducting,
            chemical_potential=chemical_potential,
        )
        transformation_matrix, _, _ = hamiltonian.diagonalizing_bogoliubov_transform()
        perm = np.array(permutation)
        full_permutation = np.concatenate([perm, perm + self.n_modes])
        for i in range(self.n_modes):
            transformation_matrix[i, :] = transformation_matrix[i, full_permutation]
        return FermionicGaussianState(transformation_matrix, occupied_orbitals)

    def circuit_parameters(self) -> Iterable[CircuitParameters]:
        for (
            tunneling,
            superconducting,
            chemical_potential,
            occupied_orbitals,
        ) in itertools.product(
            self.tunneling_values,
            self.superconducting_values,
            self.chemical_potential_values,
            self.occupied_orbitals_list,
        ):
            for permutation, label in self.measurement_labels():
                yield CircuitParameters(
                    tunneling=tunneling,
                    superconducting=superconducting,
                    chemical_potential=chemical_potential,
                    occupied_orbitals=occupied_orbitals,
                    permutation=permutation,
                    measurement_label=label,
                )

    def permutations(self) -> Iterable[Tuple[int, ...]]:
        """Fermionic mode permutations used to measure the full correlation matrix."""
        permutation = list(range(self.n_modes))
        for _ in range(math.ceil(self.n_modes / 2)):
            yield tuple(permutation)
            for i in range(0, self.n_modes - 1, 2):
                a, b = permutation[i], permutation[i + 1]
                permutation[i], permutation[i + 1] = b, a
            for i in range(1, self.n_modes - 1, 2):
                a, b = permutation[i], permutation[i + 1]
                permutation[i], permutation[i + 1] = b, a

    def measurement_labels(self) -> Iterable[Tuple[Tuple[int, ...], str]]:
        yield tuple(range(self.n_modes)), "number"
        for permutation in self.permutations():
            yield permutation, "tunneling_plus_even"
            yield permutation, "tunneling_plus_odd"
            yield permutation, "tunneling_minus_even"
            yield permutation, "tunneling_minus_odd"
            yield permutation, "superconducting_plus_even"
            yield permutation, "superconducting_plus_odd"
            yield permutation, "superconducting_minus_even"
            yield permutation, "superconducting_minus_odd"

    # HACK override run because Qiskit Experiments does not support custom transpilation
    # See https://github.com/Qiskit/qiskit-experiments/issues/669
    def run(
        self,
        backend: Optional[Backend] = None,
        analysis: Optional[Union[BaseAnalysis, None]] = "default",
        timeout: Optional[float] = None,
        circuits_per_job: Optional[int] = None,
        **run_options,
    ) -> ExperimentData:
        if backend is not None or analysis != "default":
            # Make a copy to update analysis or backend if one is provided at runtime
            experiment = self.copy()
            if backend:
                experiment._set_backend(backend)
            if isinstance(analysis, BaseAnalysis):
                experiment.analysis = analysis
        else:
            experiment = self

        if experiment.backend is None:
            raise RuntimeError("Cannot run experiment, no backend has been set.")

        # Initialize result container
        experiment_data = experiment._initialize_experiment_data()

        # Run options
        run_opts = copy.copy(experiment.run_options)
        run_opts.update_options(**run_options)
        run_opts = run_opts.__dict__

        # Generate and transpile circuits
        transpile_opts = copy.copy(experiment.transpile_options.__dict__)
        transpile_opts["initial_layout"] = list(experiment.physical_qubits)
        circuits = self._transpile(
            experiment.circuits(), experiment.backend, **transpile_opts
        )
        experiment._postprocess_transpiled_circuits(circuits, **run_options)

        # Run jobs
        jobs = experiment._run_jobs(circuits, circuits_per_job, **run_opts)
        experiment_data.add_jobs(jobs, timeout=timeout)
        experiment._add_job_metadata(experiment_data.metadata, jobs, **run_opts)

        # Optionally run analysis
        if analysis and experiment.analysis:
            return experiment.analysis.run(experiment_data)
        else:
            return experiment_data

    def _transpile(
        self, circuits: List[QuantumCircuit], backend: Backend, **transpile_options
    ) -> List[QuantumCircuit]:
        transpiled = transpile(circuits, backend, **transpile_options)

        if self.dynamical_decoupling_sequence:
            transpiled = add_dynamical_decoupling(
                transpiled, backend, self.dynamical_decoupling_sequence
            )

        return transpiled
