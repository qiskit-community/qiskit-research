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

"""Analysis for Majorana zero mode experiment."""

from __future__ import annotations

import itertools
import os
from collections import defaultdict
from typing import Iterable, Optional, Sequence, Union

import mthree
import numpy as np
from matplotlib.figure import Figure
from mthree.classes import QuasiDistribution
from qiskit.result import Counts
from qiskit_experiments.framework import (
    AnalysisResultData,
    BaseAnalysis,
    ExperimentData,
)
from qiskit_research.mzm_generation.experiment import (
    CircuitParameters,
    KitaevHamiltonianExperimentParameters,
)
from qiskit_research.mzm_generation.utils import (
    _CovarianceDict,
    compute_correlation_matrix,
    compute_parity,
    counts_to_quasis,
    diagonalizing_bogoliubov_transform,
    edge_correlation_op,
    expectation_from_correlation_matrix,
    fidelity_witness,
    kitaev_hamiltonian,
    measurement_labels,
    number_op,
    orbital_combinations,
    post_select_quasis,
    purify_idempotent_matrix,
    site_correlation_op,
)


class KitaevHamiltonianAnalysis(BaseAnalysis):
    "Analyze Kitaev Hamiltonian experimental data."

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        experiment_params = experiment_data.metadata["params"]

        # put data into dictionary for easier handling
        data = {}
        for result in experiment_data.data():
            (
                tunneling,
                superconducting,
                chemical_potential,
                occupied_orbitals,
                permutation,
                measurement_label,
                dynamical_decoupling_sequence,
            ) = result["metadata"]["params"][:7]
            if len(result["metadata"]["params"]) == 8:
                pauli_twirl_index = result["metadata"]["params"][7]
            else:
                pauli_twirl_index = None
            circuit_params = CircuitParameters(
                tunneling=tunneling,
                superconducting=superconducting
                if isinstance(superconducting, (float, complex))
                else complex(*superconducting),
                chemical_potential=chemical_potential,
                occupied_orbitals=tuple(occupied_orbitals),
                permutation=tuple(permutation),
                measurement_label=measurement_label,
                dynamical_decoupling_sequence=dynamical_decoupling_sequence,
                pauli_twirl_index=pauli_twirl_index,
            )
            data[circuit_params] = result

        # load readout calibration
        mit = mthree.M3Mitigation()
        mit.cals_from_file(
            os.path.join(
                experiment_params.basedir or "",
                "data",
                "readout_calibration",
                experiment_params.backend_name,
                f"{experiment_params.timestamp}.json",
            )
        )

        # get results
        results = list(self._compute_analysis_results(experiment_params, data, mit))
        return results, []

    def _compute_analysis_results(
        self,
        params: KitaevHamiltonianExperimentParameters,
        data: dict[CircuitParameters, dict],
        mit: mthree.M3Mitigation,
    ) -> Iterable[AnalysisResultData]:
        # fix tunneling and superconducting
        tunneling = -1.0
        superconducting = 1.0

        # get simulation results
        yield from self._compute_simulation_results(tunneling, superconducting, params)

        # create data storage objects
        corr_matrices: dict[
            tuple[
                float, Union[float, complex], float, tuple[int, ...], Optional[str], str
            ],
            tuple[np.ndarray, _CovarianceDict],
        ] = {}
        quasi_dists: dict[
            tuple[
                float, Union[float, complex], float, tuple[int, ...], Optional[str], str
            ],
            dict[tuple[tuple[int, ...], str], QuasiDistribution],
        ] = {}
        ps_removed_masses = {}

        # calculate results
        dd_sequences = params.dynamical_decoupling_sequences or [None]
        for chemical_potential in params.chemical_potential_values:
            # diagonalize
            (
                transformation_matrix,
                _,
                _,
            ) = diagonalizing_bogoliubov_transform(
                params.n_modes,
                tunneling=tunneling,
                superconducting=superconducting,
                chemical_potential=chemical_potential,
            )
            # compute parity
            W1 = transformation_matrix[:, : params.n_modes]
            W2 = transformation_matrix[:, params.n_modes :]
            full_transformation_matrix = np.block([[W1, W2], [W2.conj(), W1.conj()]])
            hamiltonian_parity = np.sign(
                np.real(np.linalg.det(full_transformation_matrix))
            )
            # compute quasis and correlation matrices
            for occupied_orbitals in params.occupied_orbitals_list:
                exact_parity = (-1) ** len(occupied_orbitals) * hamiltonian_parity
                for dd_sequence in dd_sequences:
                    quasis_raw = (
                        {}
                    )  # dict[tuple[tuple[int, ...], str], QuasiDistribution]
                    quasis_mem = (
                        {}
                    )  # dict[tuple[tuple[int, ...], str], QuasiDistribution]
                    quasis_ps = (
                        {}
                    )  # dict[tuple[tuple[int, ...], str], QuasiDistribution]
                    ps_removed_mass = {}  # dict[tuple[tuple[int, ...], str], float]
                    for permutation, label in measurement_labels(params.n_modes):
                        circuit_params = CircuitParameters(
                            tunneling,
                            superconducting,
                            chemical_potential,
                            occupied_orbitals,
                            permutation,
                            label,
                            dynamical_decoupling_sequence=dd_sequence,
                            pauli_twirl_index=0
                            if params.num_twirled_circuits
                            else None,
                        )
                        if circuit_params in data:
                            counts = Counts({})
                            for pauli_twirl_index in range(
                                max(1, params.num_twirled_circuits)
                            ):
                                circuit_params = CircuitParameters(
                                    tunneling,
                                    superconducting,
                                    chemical_potential,
                                    occupied_orbitals,
                                    permutation,
                                    label,
                                    dynamical_decoupling_sequence=dd_sequence,
                                    pauli_twirl_index=pauli_twirl_index
                                    if params.num_twirled_circuits
                                    else None,
                                )
                                these_counts = data[circuit_params]["counts"]
                                for bitstring, count in these_counts.items():
                                    if bitstring in counts:
                                        counts[bitstring] += count
                                    else:
                                        counts[bitstring] = count
                            # raw quasis
                            quasis_raw[permutation, label] = counts_to_quasis(counts)
                            # measurement error mitigation
                            quasis_mem[permutation, label] = mit.apply_correction(
                                counts,
                                params.qubits,
                                return_mitigation_overhead=True,
                            )
                            # post-selection
                            new_quasis, removed_mass = post_select_quasis(
                                quasis_mem[permutation, label],
                                lambda bitstring: (-1)
                                ** sum(1 for b in bitstring if b == "1")
                                == exact_parity,  # pylint: disable=cell-var-from-loop
                            )
                            quasis_ps[permutation, label] = new_quasis
                            ps_removed_mass[permutation, label] = removed_mass
                    # save data
                    quasi_dists[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                        "raw",
                    ] = quasis_raw
                    quasi_dists[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                        "mem",
                    ] = quasis_mem
                    quasi_dists[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                        "ps",
                    ] = quasis_ps
                    ps_removed_masses[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                    ] = ps_removed_mass
                    # compute correlation matrices
                    corr_matrices[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                        "raw",
                    ] = compute_correlation_matrix(quasis_raw)
                    corr_matrices[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                        "mem",
                    ] = compute_correlation_matrix(quasis_mem)
                    corr_mat_ps, cov_ps = compute_correlation_matrix(quasis_ps)
                    corr_matrices[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                        "ps",
                    ] = (corr_mat_ps, cov_ps)
                    corr_matrices[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                        "pur",
                    ] = (purify_idempotent_matrix(corr_mat_ps), cov_ps)

        yield AnalysisResultData("ps_removed_masses", ps_removed_masses)
        yield from self._compute_fidelity_witness(
            ["raw", "mem", "ps", "pur"],
            corr_matrices,
            params.n_modes,
            tunneling,
            superconducting,
            params.chemical_potential_values,
            params.occupied_orbitals_list,
            dd_sequences,
        )
        yield from self._compute_energy(
            ["raw", "mem", "ps", "pur"],
            corr_matrices,
            params.n_modes,
            tunneling,
            superconducting,
            params.chemical_potential_values,
            params.occupied_orbitals_list,
            dd_sequences,
        )
        yield from self._compute_edge_correlation(
            ["raw", "mem", "ps", "pur"],
            corr_matrices,
            params.n_modes,
            tunneling,
            superconducting,
            params.chemical_potential_values,
            params.occupied_orbitals_list,
            dd_sequences,
        )
        yield from self._compute_number(
            ["raw", "mem", "ps", "pur"],
            corr_matrices,
            params.n_modes,
            tunneling,
            superconducting,
            params.chemical_potential_values,
            params.occupied_orbitals_list,
            dd_sequences,
        )
        yield from self._compute_site_correlation(
            ["raw", "mem", "ps", "pur"],
            corr_matrices,
            params.n_modes,
            tunneling,
            superconducting,
            params.chemical_potential_values,
            params.occupied_orbitals_list,
            dd_sequences,
        )
        yield from self._compute_parity(
            ["raw", "mem", "ps"],
            quasi_dists,
            tunneling,
            superconducting,
            params.chemical_potential_values,
            params.occupied_orbitals_list,
            dd_sequences,
        )

    def _compute_simulation_results(
        self,
        tunneling: float,
        superconducting: Union[float, complex],
        params: KitaevHamiltonianExperimentParameters,
    ) -> Iterable[AnalysisResultData]:
        # set chemical potential values to the experiment range but with fixed resolution
        start = params.chemical_potential_values[0]
        # avoid discontinuities at 0
        if start == 0:
            start = 1e-8
        chemical_potential_values = np.linspace(
            start,
            params.chemical_potential_values[-1],
            num=50,
        )

        # construct operators
        edge_correlation = edge_correlation_op(params.n_modes)

        # create data storage objects
        energy_exact = defaultdict(list)  # dict[tuple[int, ...], list[float]]
        edge_correlation_exact = defaultdict(list)  # dict[tuple[int, ...], list[float]]
        parity_exact = defaultdict(list)  # dict[tuple[int, ...], list[float]]
        number_exact = defaultdict(list)  # dict[tuple[int, ...], list[float]]

        for chemical_potential in chemical_potential_values:
            # diagonalize Hamiltonian
            (
                transformation_matrix,
                orbital_energies,
                constant,
            ) = diagonalizing_bogoliubov_transform(
                params.n_modes,
                tunneling=tunneling,
                superconducting=superconducting,
                chemical_potential=chemical_potential,
            )
            # compute parity
            W1 = transformation_matrix[:, : params.n_modes]
            W2 = transformation_matrix[:, params.n_modes :]
            full_transformation_matrix = np.block([[W1, W2], [W2.conj(), W1.conj()]])
            hamiltonian_parity = np.sign(
                np.real(np.linalg.det(full_transformation_matrix))
            )
            # compute results
            for occupied_orbitals in params.occupied_orbitals_list:
                # compute exact correlation matrix
                occupation = np.zeros(params.n_modes)
                occupation[list(occupied_orbitals)] = 1.0
                corr_diag = np.diag(np.concatenate([occupation, 1 - occupation]))
                corr_exact = (
                    full_transformation_matrix.T.conj()
                    @ corr_diag
                    @ full_transformation_matrix
                )
                # exact values
                exact_energy = (
                    np.sum(orbital_energies[list(occupied_orbitals)]) + constant
                )
                exact_edge_correlation, _ = np.real(
                    expectation_from_correlation_matrix(edge_correlation, corr_exact)
                )
                exact_parity = (-1) ** len(occupied_orbitals) * hamiltonian_parity
                exact_number = np.real(np.sum(np.diag(corr_exact)[: params.n_modes]))
                # add computed values to data storage objects
                energy_exact[occupied_orbitals].append(exact_energy)
                edge_correlation_exact[occupied_orbitals].append(exact_edge_correlation)
                parity_exact[occupied_orbitals].append(exact_parity)
                number_exact[occupied_orbitals].append(exact_number)

        def zip_dict(d):
            return {k: (np.array(v), chemical_potential_values) for k, v in d.items()}

        yield AnalysisResultData("energy_exact", zip_dict(energy_exact))
        yield AnalysisResultData(
            "edge_correlation_exact", zip_dict(edge_correlation_exact)
        )
        yield AnalysisResultData("parity_exact", zip_dict(parity_exact))
        yield AnalysisResultData("number_exact", zip_dict(number_exact))

        # BdG
        occupied_orbitals_set = set(params.occupied_orbitals_list)
        combs = list(orbital_combinations(params.n_modes))
        threshold = -1
        for i in range(0, len(combs), 2):
            if (
                combs[i] not in occupied_orbitals_set
                or combs[i + 1] not in occupied_orbitals_set
            ):
                break
            threshold += 1
        if threshold >= 0:
            bdg_energy = np.zeros((2 * threshold, len(chemical_potential_values)))
            low = np.array(energy_exact[()])
            high = np.array(energy_exact[tuple(range(params.n_modes))])
            for i in range(threshold):
                particle = np.array(energy_exact[combs[2 * i + 2]])
                hole = np.array(energy_exact[combs[2 * i + 3]])
                bdg_energy[i] = particle - low
                bdg_energy[threshold + i] = hole - high
            yield AnalysisResultData(
                "bdg_energy_exact", (bdg_energy, chemical_potential_values)
            )

        # site correlation
        # construct operators
        site_correlation_ops = [
            site_correlation_op(i) for i in range(1, 2 * params.n_modes)
        ]

        # create data storage objects
        site_correlation_exact = defaultdict(list)  # dict[tuple[int, ...], list[float]]

        for chemical_potential in params.chemical_potential_values:
            # diagonalize Hamiltonian
            (
                transformation_matrix,
                orbital_energies,
                constant,
            ) = diagonalizing_bogoliubov_transform(
                params.n_modes,
                tunneling=tunneling,
                superconducting=superconducting,
                chemical_potential=chemical_potential,
            )
            W1 = transformation_matrix[:, : params.n_modes]
            W2 = transformation_matrix[:, params.n_modes :]
            full_transformation_matrix = np.block([[W1, W2], [W2.conj(), W1.conj()]])
            # compute results
            for occupied_orbitals in params.occupied_orbitals_list:
                # compute exact correlation matrix
                occupation = np.zeros(params.n_modes)
                occupation[list(occupied_orbitals)] = 1.0
                corr_diag = np.diag(np.concatenate([occupation, 1 - occupation]))
                corr_exact = (
                    full_transformation_matrix.T.conj()
                    @ corr_diag
                    @ full_transformation_matrix
                )
                for site_correlation in site_correlation_ops:
                    exact_site_correlation, _ = np.real(
                        expectation_from_correlation_matrix(
                            site_correlation, corr_exact
                        )
                    )
                    site_correlation_exact[
                        chemical_potential, occupied_orbitals
                    ].append(exact_site_correlation)

        yield AnalysisResultData(
            "site_correlation_exact",
            {k: np.array(v) for k, v in site_correlation_exact.items()},
        )

    def _compute_fidelity_witness(
        self,
        labels: list[str],
        corr: dict[
            tuple[
                float, Union[float, complex], float, tuple[int, ...], Optional[str], str
            ],
            tuple[np.ndarray, _CovarianceDict],
        ],
        n_modes: int,
        tunneling: float,
        superconducting: Union[float, complex],
        chemical_potential_values: Sequence[float],
        occupied_orbitals_list: Sequence[tuple[int, ...]],
        dynamical_decoupling_sequences: list[Optional[str]],
    ) -> Iterable[AnalysisResultData]:
        data: dict[
            str,
            dict[str, dict[tuple[int, ...], list[tuple[float, float]]]],
        ] = {
            dd_sequence: {label: defaultdict(list) for label in labels}
            for dd_sequence in dynamical_decoupling_sequences
        }
        for chemical_potential in chemical_potential_values:
            # diagonalize Hamiltonian
            (
                transformation_matrix,
                _,
                _,
            ) = diagonalizing_bogoliubov_transform(
                n_modes,
                tunneling=tunneling,
                superconducting=superconducting,
                chemical_potential=chemical_potential,
            )
            W1 = transformation_matrix[:, :n_modes]
            W2 = transformation_matrix[:, n_modes:]
            full_transformation_matrix = np.block([[W1, W2], [W2.conj(), W1.conj()]])
            for occupied_orbitals in occupied_orbitals_list:
                # compute exact correlation matrix
                occupation = np.zeros(n_modes)
                occupation[list(occupied_orbitals)] = 1.0
                corr_diag = np.diag(np.concatenate([occupation, 1 - occupation]))
                corr_exact = (
                    full_transformation_matrix.T.conj()
                    @ corr_diag
                    @ full_transformation_matrix
                )
                for dd_sequence, label in itertools.product(
                    dynamical_decoupling_sequences, labels
                ):
                    corr_mat, cov = corr[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                        label,
                    ]
                    fidelity_wit, stddev = fidelity_witness(corr_mat, corr_exact, cov)
                    data[dd_sequence][label][occupied_orbitals].append(
                        (fidelity_wit, stddev)
                    )

        def zip_dict(d):
            return {k: tuple(np.array(a) for a in zip(*v)) for k, v in d.items()}

        data_zipped = {
            k1: {k2: zip_dict(v2) for k2, v2 in v1.items()} for k1, v1 in data.items()
        }
        yield AnalysisResultData("fidelity_witness", data_zipped)

        data_avg: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
            dd_sequence: {} for dd_sequence in dynamical_decoupling_sequences
        }
        for dd_sequence, label in itertools.product(
            dynamical_decoupling_sequences, labels
        ):
            fidelity_witness_avg = np.zeros(len(chemical_potential_values))
            fidelity_witness_stddev = np.zeros(len(chemical_potential_values))
            for occupied_orbitals in occupied_orbitals_list:
                values, stddevs = data_zipped[dd_sequence][label][occupied_orbitals]
                fidelity_witness_avg += np.array(values)
                fidelity_witness_stddev += np.array(stddevs) ** 2
            fidelity_witness_avg /= len(occupied_orbitals_list)
            fidelity_witness_stddev = np.sqrt(fidelity_witness_stddev) / len(
                occupied_orbitals_list
            )
            data_avg[dd_sequence][label] = (
                fidelity_witness_avg,
                fidelity_witness_stddev,
            )
        yield AnalysisResultData("fidelity_witness_avg", data_avg)

    def _compute_energy(
        self,
        labels: list[str],
        corr: dict[
            tuple[
                float, Union[float, complex], float, tuple[int, ...], Optional[str], str
            ],
            tuple[np.ndarray, _CovarianceDict],
        ],
        n_modes: int,
        tunneling: float,
        superconducting: Union[float, complex],
        chemical_potential_values: Sequence[float],
        occupied_orbitals_list: Sequence[tuple[int, ...]],
        dynamical_decoupling_sequences: list[Optional[str]],
    ) -> Iterable[AnalysisResultData]:
        energy_exact = defaultdict(list)  # dict[tuple[int, ...], list[float]]
        data: dict[
            str,
            dict[str, dict[tuple[int, ...], list[tuple[float, float]]]],
        ] = {
            dd_sequence: {label: defaultdict(list) for label in labels}
            for dd_sequence in dynamical_decoupling_sequences
        }
        for chemical_potential in chemical_potential_values:
            # create Hamiltonian
            hamiltonian_quad = kitaev_hamiltonian(
                n_modes,
                tunneling=tunneling,
                superconducting=superconducting,
                chemical_potential=chemical_potential,
            )
            # diagonalize
            (
                _,
                orbital_energies,
                constant,
            ) = diagonalizing_bogoliubov_transform(
                n_modes,
                tunneling=tunneling,
                superconducting=superconducting,
                chemical_potential=chemical_potential,
            )
            for occupied_orbitals in occupied_orbitals_list:
                exact_energy = (
                    np.sum(orbital_energies[list(occupied_orbitals)]) + constant
                )
                energy_exact[occupied_orbitals].append(exact_energy)

                for dd_sequence, label in itertools.product(
                    dynamical_decoupling_sequences, labels
                ):
                    corr_mat, cov = corr[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                        label,
                    ]
                    energy, stddevs = np.real(
                        expectation_from_correlation_matrix(
                            hamiltonian_quad, corr_mat, cov
                        )
                    )
                    data[dd_sequence][label][occupied_orbitals].append(
                        (energy, stddevs)
                    )

        def zip_dict(d):
            return {k: tuple(np.array(a) for a in zip(*v)) for k, v in d.items()}

        data_zipped = {
            k1: {k2: zip_dict(v2) for k2, v2 in v1.items()} for k1, v1 in data.items()
        }
        yield AnalysisResultData("energy", data_zipped)

        # error
        data_error: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
            dd_sequence: {} for dd_sequence in dynamical_decoupling_sequences
        }
        for dd_sequence, label in itertools.product(
            dynamical_decoupling_sequences, labels
        ):
            error = np.zeros(len(chemical_potential_values))
            error_stddev = np.zeros(len(chemical_potential_values))
            for occupied_orbitals in occupied_orbitals_list:
                exact = np.array(energy_exact[occupied_orbitals])
                values, stddevs = data_zipped[dd_sequence][label][occupied_orbitals]
                values = np.array(values)
                error += np.abs(values - exact)
                error_stddev += np.array(stddevs) ** 2
            error /= len(occupied_orbitals_list)
            error_stddev = np.sqrt(error_stddev) / len(occupied_orbitals_list)
            data_error[dd_sequence][label] = error, error_stddev
        yield AnalysisResultData("energy_error", data_error)

        # BdG
        occupied_orbitals_set = set(occupied_orbitals_list)
        combs = list(orbital_combinations(n_modes))
        threshold = -1
        for i in range(0, len(combs), 2):
            if (
                combs[i] not in occupied_orbitals_set
                or combs[i + 1] not in occupied_orbitals_set
            ):
                break
            threshold += 1
        if threshold >= 0:
            data_bdg: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
                dd_sequence: {} for dd_sequence in dynamical_decoupling_sequences
            }
            data_bdg_error: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
                dd_sequence: {} for dd_sequence in dynamical_decoupling_sequences
            }
            for dd_sequence, label in itertools.product(
                dynamical_decoupling_sequences, labels
            ):
                bdg_energy = np.zeros((2 * threshold, len(chemical_potential_values)))
                bdg_stddev = np.zeros((2 * threshold, len(chemical_potential_values)))
                low, low_stddev = data_zipped[dd_sequence][label][()]
                high, high_stddev = data_zipped[dd_sequence][label][
                    tuple(range(n_modes))
                ]
                error = np.zeros(len(chemical_potential_values))
                error_stddev = np.zeros(len(chemical_potential_values))
                low_exact = energy_exact[()]
                high_exact = energy_exact[tuple(range(n_modes))]
                for i in range(threshold):
                    # data
                    particle, particle_stddev = data_zipped[dd_sequence][label][
                        combs[2 * i + 2]
                    ]
                    hole, hole_stddev = data_zipped[dd_sequence][label][
                        combs[2 * i + 3]
                    ]
                    # exact values
                    particle_exact = np.array(energy_exact[combs[2 * i + 2]])
                    hole_exact = np.array(energy_exact[combs[2 * i + 3]])
                    # energy
                    bdg_energy[i] = particle - low
                    bdg_energy[threshold + i] = hole - high
                    # stddev
                    bdg_stddev[i] = low_stddev**2 + particle_stddev**2
                    bdg_stddev[threshold + i] = high_stddev**2 + hole_stddev**2
                    # error
                    error += np.abs((particle - low) - (particle_exact - low_exact))
                    error += np.abs((hole - high) - (hole_exact - high_exact))
                    # error stddev
                    error_stddev += (
                        np.array(low_stddev) ** 2
                        + np.array(particle_stddev) ** 2
                        + np.array(high_stddev) ** 2
                        + np.array(hole_stddev) ** 2
                    )
                bdg_stddev = np.sqrt(bdg_stddev)
                error /= 2 * threshold
                error_stddev = np.sqrt(error_stddev) / (2 * threshold)
                data_bdg[dd_sequence][label] = bdg_energy, bdg_stddev
                data_bdg_error[dd_sequence][label] = error, error_stddev
            yield AnalysisResultData("bdg_energy", data_bdg)
            yield AnalysisResultData("bdg_energy_error", data_bdg_error)

    def _compute_edge_correlation(
        self,
        labels: list[str],
        corr: dict[
            tuple[
                float, Union[float, complex], float, tuple[int, ...], Optional[str], str
            ],
            tuple[np.ndarray, _CovarianceDict],
        ],
        n_modes: int,
        tunneling: float,
        superconducting: Union[float, complex],
        chemical_potential_values: Iterable[float],
        occupied_orbitals_list: Iterable[tuple[int, ...]],
        dynamical_decoupling_sequences: list[Optional[str]],
    ) -> Iterable[AnalysisResultData]:
        edge_correlation = edge_correlation_op(n_modes)
        data: dict[
            str,
            dict[str, dict[tuple[int, ...], list[tuple[float, float]]]],
        ] = {
            dd_sequence: {label: defaultdict(list) for label in labels}
            for dd_sequence in dynamical_decoupling_sequences
        }
        for chemical_potential in chemical_potential_values:
            for occupied_orbitals in occupied_orbitals_list:
                for dd_sequence, label in itertools.product(
                    dynamical_decoupling_sequences, labels
                ):
                    corr_mat, cov = corr[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                        label,
                    ]
                    edge_correlation_val, stddev = np.real(
                        expectation_from_correlation_matrix(
                            edge_correlation, corr_mat, cov
                        )
                    )
                    data[dd_sequence][label][occupied_orbitals].append(
                        (edge_correlation_val, stddev)
                    )

        def zip_dict(d):
            return {k: tuple(np.array(a) for a in zip(*v)) for k, v in d.items()}

        data_zipped = {
            k1: {k2: zip_dict(v2) for k2, v2 in v1.items()} for k1, v1 in data.items()
        }
        yield AnalysisResultData("edge_correlation", data_zipped)

    def _compute_number(
        self,
        labels: list[str],
        corr: dict[
            tuple[
                float, Union[float, complex], float, tuple[int, ...], Optional[str], str
            ],
            tuple[np.ndarray, _CovarianceDict],
        ],
        n_modes: int,
        tunneling: float,
        superconducting: Union[float, complex],
        chemical_potential_values: Iterable[float],
        occupied_orbitals_list: Iterable[tuple[int, ...]],
        dynamical_decoupling_sequences: list[Optional[str]],
    ) -> Iterable[AnalysisResultData]:
        number = number_op(n_modes)
        data: dict[
            str,
            dict[str, dict[tuple[int, ...], list[tuple[float, float]]]],
        ] = {
            dd_sequence: {label: defaultdict(list) for label in labels}
            for dd_sequence in dynamical_decoupling_sequences
        }
        for chemical_potential in chemical_potential_values:
            for occupied_orbitals in occupied_orbitals_list:
                for dd_sequence, label in itertools.product(
                    dynamical_decoupling_sequences, labels
                ):
                    corr_mat, cov = corr[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                        label,
                    ]
                    number_val, stddev = np.real(
                        expectation_from_correlation_matrix(number, corr_mat, cov)
                    )
                    data[dd_sequence][label][occupied_orbitals].append(
                        (number_val, stddev)
                    )

        def zip_dict(d):
            return {k: tuple(np.array(a) for a in zip(*v)) for k, v in d.items()}

        data_zipped = {
            k1: {k2: zip_dict(v2) for k2, v2 in v1.items()} for k1, v1 in data.items()
        }
        yield AnalysisResultData("number", data_zipped)

    def _compute_parity(
        self,
        labels: list[str],
        quasi_dists: dict[
            tuple[
                float, Union[float, complex], float, tuple[int, ...], Optional[str], str
            ],
            dict[tuple[tuple[int, ...], str], QuasiDistribution],
        ],
        tunneling: float,
        superconducting: Union[float, complex],
        chemical_potential_values: Iterable[float],
        occupied_orbitals_list: Iterable[tuple[int, ...]],
        dynamical_decoupling_sequences: list[Optional[str]],
    ) -> Iterable[AnalysisResultData]:
        data: dict[
            str,
            dict[str, dict[tuple[int, ...], list[tuple[float, float]]]],
        ] = {
            dd_sequence: {label: defaultdict(list) for label in labels}
            for dd_sequence in dynamical_decoupling_sequences
        }
        for chemical_potential in chemical_potential_values:
            for occupied_orbitals in occupied_orbitals_list:
                for dd_sequence, label in itertools.product(
                    dynamical_decoupling_sequences, labels
                ):
                    quasis = quasi_dists[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                        label,
                    ]
                    parity, stddev = compute_parity(quasis)
                    data[dd_sequence][label][occupied_orbitals].append((parity, stddev))

        def zip_dict(d):
            return {k: tuple(np.array(a) for a in zip(*v)) for k, v in d.items()}

        data_zipped = {
            k1: {k2: zip_dict(v2) for k2, v2 in v1.items()} for k1, v1 in data.items()
        }
        yield AnalysisResultData("parity", data_zipped)

    def _compute_site_correlation(
        self,
        labels: list[str],
        corr: dict[
            tuple[
                float, Union[float, complex], float, tuple[int, ...], Optional[str], str
            ],
            tuple[np.ndarray, _CovarianceDict],
        ],
        n_modes: int,
        tunneling: float,
        superconducting: Union[float, complex],
        chemical_potential_values: Iterable[float],
        occupied_orbitals_list: Iterable[tuple[int, ...]],
        dynamical_decoupling_sequences: list[Optional[str]],
    ) -> Iterable[AnalysisResultData]:
        site_correlation_ops = [site_correlation_op(i) for i in range(1, 2 * n_modes)]
        data: dict[
            str,
            dict[str, dict[tuple[float, tuple[int, ...]], list[tuple[float, float]]]],
        ] = {
            dd_sequence: {label: defaultdict(list) for label in labels}
            for dd_sequence in dynamical_decoupling_sequences
        }
        for chemical_potential in chemical_potential_values:
            for occupied_orbitals in occupied_orbitals_list:
                for dd_sequence, label in itertools.product(
                    dynamical_decoupling_sequences, labels
                ):
                    corr_mat, cov = corr[
                        tunneling,
                        superconducting,
                        chemical_potential,
                        occupied_orbitals,
                        dd_sequence,
                        label,
                    ]
                    for site_correlation in site_correlation_ops:
                        site_correlation_val, stddev = np.real(
                            expectation_from_correlation_matrix(
                                site_correlation, corr_mat, cov
                            )
                        )
                        data[dd_sequence][label][
                            chemical_potential, occupied_orbitals
                        ].append((site_correlation_val, stddev))

        def zip_dict(d):
            return {k: tuple(np.array(a) for a in zip(*v)) for k, v in d.items()}

        data_zipped = {
            k1: {k2: zip_dict(v2) for k2, v2 in v1.items()} for k1, v1 in data.items()
        }
        yield AnalysisResultData("site_correlation", data_zipped)
