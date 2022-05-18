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

"""Majorana zero modes generation experiment."""

from __future__ import annotations

import dataclasses
import functools
import itertools
from collections import namedtuple
from typing import Iterable, Optional, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate
from qiskit.providers import Provider
from qiskit_experiments.framework import BaseExperiment
from qiskit_nature.circuit.library import FermionicGaussianState
from qiskit_research.mzm_generation.utils import (
    get_backend,
    kitaev_hamiltonian,
    measure_interaction_op,
    measurement_labels,
    transpile_circuit,
)

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
        "dynamical_decoupling_sequence",
        "pauli_twirl_index",
    ],
)


@dataclasses.dataclass
class KitaevHamiltonianExperimentParameters:
    """Parameters for a Kitaev Hamiltonian Experiment."""

    timestamp: str
    backend_name: str
    qubits: list[int]
    n_modes: int
    tunneling_values: list[float]
    superconducting_values: list[Union[float, complex]]
    chemical_potential_values: list[float]
    occupied_orbitals_list: list[tuple[int, ...]]
    dynamical_decoupling_sequences: Optional[list[str]] = None
    pulse_scaling: bool = False
    num_twirled_circuits: int = 0
    seed: Optional[int] = None
    basedir: Optional[str] = None

    @property
    def filename(self) -> str:
        """Filename for data from this experiment."""
        return f"{self.timestamp}_{self.backend_name}_n{self.n_modes}"

    @classmethod
    def __json_decode__(cls, value: dict) -> "KitaevHamiltonianExperimentParameters":
        """JSON deserialization."""
        value["occupied_orbitals_list"] = [
            tuple(occupied_orbitals)
            for occupied_orbitals in value["occupied_orbitals_list"]
        ]
        return cls(**value)


class KitaevHamiltonianExperiment(BaseExperiment):
    """Prepare and measure eigenstates of the Kitaev Hamiltonian."""

    def __init__(
        self,
        params: KitaevHamiltonianExperimentParameters,
        provider: Optional[Provider] = None,
    ) -> None:
        self.params = params
        self.rng = np.random.default_rng(params.seed)
        backend = get_backend(params.backend_name, provider, seed_simulator=params.seed)
        super().__init__(qubits=params.qubits, backend=backend)

    def _metadata(self) -> dict:
        metadata = super()._metadata()
        additional_metadata = {"params": self.params}
        metadata.update(additional_metadata)
        return metadata

    def circuits(self) -> list[QuantumCircuit]:
        return list(self._circuits())

    def _circuits(self) -> Iterable[QuantumCircuit]:
        dd_sequences = self.params.dynamical_decoupling_sequences or [None]
        for (
            tunneling,
            superconducting,
            chemical_potential,
            occupied_orbitals,
        ) in itertools.product(
            self.params.tunneling_values,
            self.params.superconducting_values,
            self.params.chemical_potential_values,
            self.params.occupied_orbitals_list,
        ):
            for permutation, label in measurement_labels(self.params.n_modes):
                base_circuit = self._base_circuit(
                    tunneling,
                    superconducting,
                    chemical_potential,
                    occupied_orbitals,
                    permutation,
                )
                # if the circuit is real-valued, the correlation matrix
                # has zero imaginary part so the "minus" circuits are
                # not needed
                if "_minus_" in label and _all_real_rz_gates(base_circuit, atol=1e-6):
                    continue
                for dd_sequence in dd_sequences:
                    for pauli_twirl_index in range(
                        max(1, self.params.num_twirled_circuits)
                    ):
                        params = CircuitParameters(
                            tunneling=tunneling,
                            superconducting=superconducting,
                            chemical_potential=chemical_potential,
                            occupied_orbitals=occupied_orbitals,
                            permutation=permutation,
                            measurement_label=label,
                            dynamical_decoupling_sequence=dd_sequence,
                            pauli_twirl_index=pauli_twirl_index
                            if self.params.num_twirled_circuits
                            else None,
                        )
                        circuit = measure_interaction_op(base_circuit, label)
                        circuit.metadata = {"params": params}
                        yield circuit

    @functools.lru_cache(maxsize=1024)
    def _base_circuit(
        self,
        tunneling: float,
        superconducting: float,
        chemical_potential: float,
        occupied_orbitals: tuple[int, ...],
        permutation: tuple[int, ...],
    ) -> QuantumCircuit:
        hamiltonian = kitaev_hamiltonian(
            self.params.n_modes,
            tunneling=tunneling,
            superconducting=superconducting,
            chemical_potential=chemical_potential,
        )
        transformation_matrix, _, _ = hamiltonian.diagonalizing_bogoliubov_transform()
        perm = np.array(permutation)
        full_permutation = np.concatenate([perm, perm + self.params.n_modes])
        for i in range(self.params.n_modes):
            transformation_matrix[i, :] = transformation_matrix[i, full_permutation]
        return FermionicGaussianState(transformation_matrix, occupied_orbitals)

    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        """Return a list of experiment circuits, transpiled."""
        return [
            transpile_circuit(
                circuit,
                self.backend,
                initial_layout=list(self.physical_qubits),
                dynamical_decoupling_sequence=circuit.metadata[
                    "params"
                ].dynamical_decoupling_sequence,
                pulse_scaling=self.params.pulse_scaling,
                pauli_twirling=bool(self.params.num_twirled_circuits),
                seed=self.rng,
            )
            for circuit in self.circuits()
        ]


def _all_real_rz_gates(circuit: QuantumCircuit, rtol=1e-5, atol=1e-8) -> bool:
    """Check if all RZ gates in the circuit are real-valued up to global phase."""
    for gate, _, _ in circuit:
        if isinstance(gate, RZGate):
            (theta,) = gate.params
            if not np.isclose(
                theta % np.pi, 0.0, rtol=rtol, atol=atol
            ) and not np.isclose(theta % np.pi, np.pi, atol=1e-8):
                return False
    return True
