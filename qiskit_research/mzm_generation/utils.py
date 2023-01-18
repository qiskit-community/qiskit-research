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

"""Utilities for Majorana zero modes generation experiment."""

from __future__ import annotations

import functools
import math
from collections import defaultdict
from collections.abc import Iterator
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Optional,
    Tuple,
    Union,
    cast,
)

import mapomatic
import mthree
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit.library import XXMinusYYGate, XXPlusYYGate
from qiskit.providers import Backend, Provider
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap, Layout, PassManager
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.passes import (
    ApplyLayout,
    BasisTranslator,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    Optimize1qGatesDecomposition,
    SetLayout,
    UnrollCustomDefinitions,
    VF2Layout,
)
from qiskit.transpiler.passes.scheduling import ALAPScheduleAnalysis
from qiskit_nature.circuit.library import FermionicGaussianState
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.operators.second_quantization import (
    FermionicOp,
    QuadraticHamiltonian,
)
from qiskit_research.utils import (
    CombineRuns,
    PauliTwirl,
    RZXtoEchoedCR,
    SECRCalibrationBuilder,
    XXMinusYYtoRZX,
    XXPlusYYtoRZX,
    add_pulse_calibrations,
    dynamical_decoupling_passes,
    get_backend,
)
from qiskit_research.utils.pulse_scaling import BASIS_GATES

_CovarianceDict = Dict[FrozenSet[Tuple[int, int]], float]


def orbital_combinations(
    n_modes: int, threshold: Optional[int] = None
) -> Iterable[tuple[int, ...]]:
    """Yields orbital combinations with 0 or 1 particles or holes."""
    if threshold is None:
        threshold = n_modes
    # no particles
    yield ()
    # no holes
    yield tuple(range(n_modes))
    for i in range(threshold):
        # one particle
        yield (i,)
        # one hole
        yield tuple(range(i)) + tuple(range(i + 1, n_modes))


def majorana_op(index: int, action: int) -> FermionicOp:
    """Majorana fermion operator."""
    if action == 0:
        return FermionicOp(f"-_{index}") + FermionicOp(f"+_{index}")
    return -1j * (FermionicOp(f"-_{index}") - FermionicOp(f"+_{index}"))


def site_correlation_op(site: int) -> FermionicOp:
    """Majorana site correlation operator."""
    return 1j * majorana_op(0, 0) @ majorana_op(site // 2, site % 2)


def edge_correlation_op(n_modes: int) -> FermionicOp:
    """Majorana edge correlation operator."""
    return site_correlation_op(2 * n_modes - 1)


def number_op(n_modes: int) -> FermionicOp:
    """Number operator."""
    if not n_modes:
        return FermionicOp.zero(register_length=0)
    return cast(FermionicOp, sum(FermionicOp(f"N_{i}") for i in range(n_modes)))


def expectation(operator: np.ndarray, state: np.ndarray) -> complex:
    """Expectation value of operator with state."""
    return np.vdot(state, operator @ state)


def variance(operator: np.ndarray, state: np.ndarray) -> complex:
    """Variance of operator with state."""
    return expectation(operator @ operator, state) - expectation(operator, state) ** 2


def jordan_wigner(op: FermionicOp) -> SparsePauliOp:
    """Jordan-Wigner transform."""
    return JordanWignerMapper().map(op).primitive


@functools.lru_cache
def kitaev_hamiltonian(
    n_modes: int,
    tunneling: float,
    superconducting: Union[float, complex],
    chemical_potential: float,
) -> QuadraticHamiltonian:
    """Create Kitaev model Hamiltonian."""
    eye = np.eye(n_modes)
    upper_diag = np.diag(np.ones(n_modes - 1), k=1)
    lower_diag = np.diag(np.ones(n_modes - 1), k=-1)
    hermitian_part = -tunneling * (upper_diag + lower_diag) + chemical_potential * eye
    antisymmetric_part = superconducting * (upper_diag - lower_diag)
    constant = -0.5 * chemical_potential * n_modes
    return QuadraticHamiltonian(
        hermitian_part=hermitian_part,
        antisymmetric_part=antisymmetric_part,
        constant=constant,
    )


@functools.lru_cache
def diagonalizing_bogoliubov_transform(
    n_modes: int,
    tunneling: float,
    superconducting: Union[float, complex],
    chemical_potential: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Diagonalize a Kitaev Hamiltonian."""
    return kitaev_hamiltonian(
        n_modes,
        tunneling=tunneling,
        superconducting=superconducting,
        chemical_potential=chemical_potential,
    ).diagonalizing_bogoliubov_transform()


def bdg_hamiltonian(hamiltonian: QuadraticHamiltonian) -> np.ndarray:
    """Bogoliubov-de Gennes Hamiltonian."""
    return np.block(
        [
            [
                hamiltonian.hermitian_part,
                hamiltonian.antisymmetric_part,
            ],
            [
                -hamiltonian.antisymmetric_part.conj(),
                -hamiltonian.hermitian_part.conj(),
            ],
        ]
    )


def correlation_matrix(
    transformation_matrix: np.ndarray, occupied_orbitals: Iterable[int]
) -> np.ndarray:
    """Compute correlation matrix of a fermionic Gaussian state."""
    n_modes, _ = transformation_matrix.shape
    W1 = transformation_matrix[:, :n_modes]
    W2 = transformation_matrix[:, n_modes:]
    full_transformation_matrix = np.block([[W1, W2], [W2.conj(), W1.conj()]])
    occupation = np.zeros(n_modes)
    occupation[list(occupied_orbitals)] = 1.0
    corr_diag = np.diag(np.concatenate([occupation, 1 - occupation]))
    return full_transformation_matrix.T.conj() @ corr_diag @ full_transformation_matrix


def correlation_matrix_from_state_vector(state: np.ndarray) -> np.ndarray:
    """Compute correlation matrix from state vector."""
    (N,) = state.shape
    n = N.bit_length() - 1
    corr = np.zeros((2 * n, 2 * n), dtype=complex)
    for i in range(n):
        for j in range(i, n):
            op = FermionicOp(f"+_{i} -_{j}", register_length=n)
            op_jw = jordan_wigner(op).to_matrix()
            val = expectation(op_jw, state)
            corr[i, j] = val
            corr[j, i] = val.conjugate()
            corr[i + n, j + n] = float(i == j) - val.conjugate()
            corr[j + n, i + n] = float(i == j) - val

            op = FermionicOp(f"-_{i} -_{j}", register_length=n)
            op_jw = jordan_wigner(op).to_matrix()
            val = expectation(op_jw, state)
            corr[i + n, j] = val
            corr[j + n, i] = -val
            corr[i, j + n] = -val.conjugate()
            corr[j, i + n] = val.conjugate()
    return corr


def covariance_matrix(corr: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to covariance matrix."""
    n, _ = corr.shape
    eye = np.eye(n // 2)
    # TODO figure out why mypy fails on this
    majorana_basis = np.block([[eye, eye], [1j * eye, -1j * eye]]) / np.sqrt(2)
    return np.real(
        1j * majorana_basis @ (2 * corr - np.eye(n)) @ majorana_basis.T.conj()
    )


def fidelity_witness(
    corr: np.ndarray,
    corr_target: np.ndarray,
    cov: Optional[_CovarianceDict] = None,
) -> tuple[float, float]:
    """Compute fidelity witness from correlation matrix.

    Reference: arXiv:1703.03152

    Args:
        corr: The correlation matrix for which to compute the witness.
        corr_target: The correlation matrix of the target state.
        cov: Covariances of the elements of the first given correlation matrix.

    Returns:
        - Fidelity witness
        - Standard deviation of fidelity witness
    """
    # compute fidelity witness
    dim, _ = corr.shape
    n = dim // 2
    witness = 1 - np.trace((corr_target - corr) @ (corr_target - 0.5 * np.eye(dim)))

    # compute variance
    var = 0.0
    if cov is not None:
        # off-diagonal entries
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(n):
                    for ell in range(k + 1, n):
                        var += 8 * np.real(
                            corr_target[i, j]
                            * corr_target[k, ell]
                            * cov[frozenset([(i, j), (k, ell)])]
                        )
                        var += 8 * np.real(
                            corr_target[i, j]
                            * corr_target[k, ell].conjugate()
                            * cov[frozenset([(i, j), (k, ell)])]
                        )
                        var += 8 * np.real(
                            corr_target[i, j + n]
                            * corr_target[k, ell + n]
                            * cov[frozenset([(i, j + n), (k, ell + n)])]
                        )
                        var += 8 * np.real(
                            corr_target[i, j + n]
                            * corr_target[k, ell + n].conjugate()
                            * cov[frozenset([(i, j + n), (k, ell + n)])]
                        )
        # diagonal entries
        for i in range(n):
            for j in range(i, n):
                var += (1 + (i != j)) * (
                    (1 - 2 * corr_target[i, i])
                    * (1 - 2 * corr_target[j, j])
                    * cov[frozenset([(i, i), (j, j)])]
                )

    return np.real(witness), np.sqrt(np.real(var))


def expectation_from_correlation_matrix(
    operator: Union[QuadraticHamiltonian, FermionicOp],
    corr: np.ndarray,
    cov: Optional[_CovarianceDict] = None,
) -> tuple[complex, float]:
    """Compute expectation value of operator from correlation matrix.

    Raises:
        ValueError: Operator must be quadratic in the fermionic ladder operators.
    """
    dim, _ = corr.shape
    n = dim // 2

    if isinstance(operator, QuadraticHamiltonian):
        # expectation
        exp_val = (
            np.sum(
                operator.hermitian_part * corr[:n, :n]
                + np.real(operator.antisymmetric_part * corr[:n, n:])
            )
            + operator.constant
        )
        # variance
        var = 0 + 0j
        if cov is not None:
            # off-diagonal entries
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(n):
                        for ell in range(k + 1, n):
                            var += 2 * np.real(
                                operator.hermitian_part[i, j]
                                * operator.hermitian_part[k, ell]
                                * cov[frozenset([(i, j), (k, ell)])]
                            )
                            var += 2 * np.real(
                                operator.hermitian_part[i, j]
                                * operator.hermitian_part[k, ell].conjugate()
                                * cov[frozenset([(i, j), (k, ell)])]
                            )
                            var += 2 * np.real(
                                operator.antisymmetric_part[i, j]
                                * operator.antisymmetric_part[k, ell]
                                * cov[frozenset([(i, j + n), (k, ell + n)])]
                            )
                            var += 2 * np.real(
                                operator.antisymmetric_part[i, j]
                                * operator.antisymmetric_part[k, ell].conjugate()
                                * cov[frozenset([(i, j + n), (k, ell + n)])]
                            )
            # diagonal entries
            for i in range(n):
                for j in range(i, n):
                    var += (1 + (i != j)) * (
                        operator.hermitian_part[i, i]
                        * operator.hermitian_part[j, j]
                        * cov[frozenset([(i, i), (j, j)])]
                    )
    else:  # isinstance(operator, FermionicOp)
        # expectation
        exp_val = 0.0
        # HACK FermionicOp should support iteration with public API
        # See https://github.com/Qiskit/qiskit-nature/issues/541
        for term, coeff in operator.terms():
            if not term:
                exp_val += coeff
            elif len(term) == 2:
                (action_i, i), (action_j, j) = term
                exp_val += (
                    coeff * corr[i + n * (action_i == "-"), j + n * (action_j == "+")]
                )
            else:
                raise ValueError(
                    "Operator must be quadratic in the fermionic ladder operators."
                )
        # variance
        var = 0 + 0j
        if cov is not None:
            # HACK FermionicOp should support iteration with public API
            # See https://github.com/Qiskit/qiskit-nature/issues/541
            for term_ij, coeff_ij in operator.terms():
                if not term_ij:
                    continue
                (action_i, i), (action_j, j) = term_ij
                sign_ij = 1
                if i > j:
                    i, j = j, i
                    action_i, action_j = action_j, action_i
                    sign_ij *= -1
                if action_i == "-":
                    sign_ij *= -1
                for term_kl, coeff_kl in operator.terms():
                    if not term_kl:
                        continue
                    (action_k, k), (action_l, ell) = term_kl
                    sign_kl = 1
                    if k > ell:
                        k, ell = ell, k
                        action_k, action_l = action_l, action_k
                        sign_kl = -1
                    if action_k == "-":
                        sign_kl *= -1
                    var += (
                        coeff_ij
                        * coeff_kl.conjugate()
                        * sign_ij
                        * sign_kl
                        * cov[
                            frozenset(
                                [
                                    (i, j + n * (action_i == action_j)),
                                    (k, ell + n * (action_k == action_l)),
                                ]
                            )
                        ]
                    )

    return exp_val, np.sqrt(np.real(var))


def measure_interaction_op(circuit: QuantumCircuit, label: str) -> QuantumCircuit:
    """Measure fermionic interaction with a circuit that preserves parity."""
    if label == "number":
        circuit = circuit.copy()
        circuit.measure_all()
        return circuit

    if label.startswith("tunneling_plus"):
        # this gate transforms a^\dagger_i a_j + h.c into (IZ - ZI) / 2
        gate = XXPlusYYGate(np.pi / 2, -np.pi / 2)
    elif label.startswith("tunneling_minus"):
        # this gate transforms -i * (a^\dagger_i a_j - h.c) into (IZ - ZI) / 2
        gate = XXPlusYYGate(np.pi / 2, -np.pi)
    elif label.startswith("superconducting_plus"):
        # this gate transforms a^\dagger_i a^_\dagger_j + h.c into (IZ + ZI) / 2
        gate = XXMinusYYGate(np.pi / 2, -np.pi / 2)
    else:  # label.startswith("superconducting_minus")
        # this gate transforms -i * (a^\dagger_i a^_\dagger_j - h.c) into (IZ + ZI) / 2
        gate = XXMinusYYGate(np.pi / 2, -np.pi)

    if label.endswith("even"):
        start_index = 0
    else:  # label.endswith('odd')
        start_index = 1

    circuit = circuit.copy()
    for i in range(start_index, circuit.num_qubits - 1, 2):
        # measure interaction between qubits i and i + 1
        circuit.append(gate, [i, i + 1])
    circuit.measure_all()

    return circuit


def compute_correlation_matrix(
    quasis: dict[tuple[tuple[int, ...], str], mthree.classes.QuasiDistribution]
) -> tuple[np.ndarray, _CovarianceDict]:
    """Compute correlation matrix from quasiprobabilities.

    Returns:
        - Correlation matrix
        - Dictionary containing covariances between entries of the correlation matrix
    """
    n = len(next(iter(next(iter(quasis.values())))))

    # off-diagonal entries
    tunneling_plus, tunneling_plus_cov = compute_interaction_matrix(
        quasis, "tunneling_plus"
    )
    tunneling_minus, tunneling_minus_cov = compute_interaction_matrix(
        quasis, "tunneling_minus"
    )
    superconducting_plus, superconducting_plus_cov = compute_interaction_matrix(
        quasis, "superconducting_plus"
    )
    superconducting_minus, superconducting_minus_cov = compute_interaction_matrix(
        quasis, "superconducting_minus"
    )
    tunneling_mat = 0.5 * (tunneling_plus + 1j * tunneling_minus)
    superconducting_mat = 0.5 * (superconducting_plus + 1j * superconducting_minus)
    corr = np.block(
        [
            [tunneling_mat, superconducting_mat],
            [-superconducting_mat.conj(), np.eye(n) - tunneling_mat.T],
        ],
    )

    # diagonal entries
    num_quasis = quasis[(tuple(range(n)), "number")]
    for i in range(n):
        num = "I" * (n - i - 1) + "1" + "I" * i
        expval = num_quasis.expval(num)
        corr[i, i] = expval
        corr[i + n, i + n] = 1 - expval

    # covariance
    cov: _CovarianceDict = defaultdict(float)
    # off-diagonal entries
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                for ell in range(k + 1, n):
                    cov[frozenset([(i, j), (k, ell)])] = 0.25 * (
                        tunneling_plus_cov[frozenset([(i, j), (k, ell)])]
                        + tunneling_minus_cov[frozenset([(i, j), (k, ell)])]
                    )
                    cov[frozenset([(i, j + n), (k, ell + n)])] = 0.25 * (
                        superconducting_plus_cov[frozenset([(i, j), (k, ell)])]
                        + superconducting_minus_cov[frozenset([(i, j), (k, ell)])]
                    )
    # diagonal entries
    for i in range(n):
        z0 = "I" * (n - i - 1) + "Z" + "I" * i
        for j in range(i, n):
            z1 = "I" * (n - j - 1) + "Z" + "I" * j
            cov[frozenset([(i, i), (j, j)])] = 0.25 * covariance(num_quasis, z0, z1)

    return corr, cov


def compute_interaction_matrix(
    quasis: dict[tuple[tuple[int, ...], str], mthree.classes.QuasiDistribution],
    label: str,
) -> tuple[np.ndarray, _CovarianceDict]:
    """Compute interaction matrix from quasiprobabilities.

    Returns:
        - Interaction matrix
        - Dictionary containing covariances between entries of the interaction matrix
    """
    n = len(next(iter(next(iter(quasis.values())))))
    mat = np.zeros((n, n))
    cov: _CovarianceDict = defaultdict(float)

    permutation = tuple(range(n))
    if (permutation, f"{label}_even") not in quasis and (
        permutation,
        f"{label}_odd",
    ) not in quasis:
        # the interaction was not measured, so it is assumed to be zero
        return mat, cov

    if label == "tunneling_plus":
        sign = -1
        symmetry = 1
    elif label == "tunneling_minus":
        sign = -1
        symmetry = -1
    elif label == "superconducting_plus":
        sign = 1
        symmetry = -1
    else:  # label == "superconducting_minus"
        sign = 1
        symmetry = -1

    # compute interaction matrix
    for permutation in orbital_permutations(n):
        even_quasis = quasis[permutation, f"{label}_even"]
        odd_quasis = quasis[permutation, f"{label}_odd"]
        for start_index in [0, 1]:
            quasi_dist = odd_quasis if start_index else even_quasis
            for i in range(start_index, n - 1, 2):
                z0 = "I" * (n - i - 1) + "Z" + "I" * i
                z1 = "I" * (n - i - 2) + "Z" + "I" * (i + 1)
                z0_expval = quasi_dist.expval(z0)
                z1_expval = quasi_dist.expval(z1)
                val = 0.5 * (z1_expval + sign * z0_expval)
                p, q = permutation[i], permutation[i + 1]
                mat[p, q] = val
                mat[q, p] = symmetry * val

    # compute covariance
    for permutation in orbital_permutations(n):
        even_quasis = quasis[permutation, f"{label}_even"]
        odd_quasis = quasis[permutation, f"{label}_odd"]
        for start_index in [0, 1]:
            quasi_dist = odd_quasis if start_index else even_quasis
            for i in range(start_index, n - 1, 2):
                z0 = "I" * (n - i - 1) + "Z" + "I" * i
                z1 = "I" * (n - i - 2) + "Z" + "I" * (i + 1)
                p, q = permutation[i], permutation[i + 1]
                if p > q:
                    p, q = q, p
                for j in range(start_index, n - 1, 2):
                    z2 = "I" * (n - j - 1) + "Z" + "I" * j
                    z3 = "I" * (n - j - 2) + "Z" + "I" * (j + 1)
                    r, s = permutation[j], permutation[j + 1]
                    if r > s:
                        r, s = s, r
                    cov[frozenset([(p, q), (r, s)])] = 0.25 * (
                        covariance(quasi_dist, z0, z2)
                        + sign * covariance(quasi_dist, z0, z3)
                        + sign * covariance(quasi_dist, z1, z2)
                        + covariance(quasi_dist, z1, z3)
                    )

    return mat, cov


def covariance(
    quasi_dist: mthree.classes.QuasiDistribution, op1: str, op2: str
) -> float:
    """Compute covariance of two diagonal operators from quasiprobabilities."""
    expval1 = quasi_dist.expval(op1)
    expval2 = quasi_dist.expval(op2)
    cov = 0.0
    for bitstring, quasiprob in quasi_dist.items():
        cov += (
            quasiprob
            * (evaluate_diagonal_op(op1, bitstring) - expval1)
            * (evaluate_diagonal_op(op2, bitstring) - expval2)
        )
    return cov * quasi_dist.mitigation_overhead / quasi_dist.shots


def evaluate_diagonal_op(operator: str, bitstring: str) -> int:
    """Evaluate a diagional operator on a bitstring."""
    prod = 1
    for op, bit in zip(operator, bitstring):
        if op in ("0", "1"):
            prod *= bit == op
        elif op == "Z":
            prod *= (-1) ** (bit == "1")
    return prod


def compute_parity(
    quasis: dict[tuple[tuple[int, ...], str], mthree.classes.QuasiDistribution]
) -> tuple[float, float]:
    """Compute parity from quasiprobabilities."""
    n = len(next(iter(next(iter(quasis.values())))))
    quasi_dist = quasis[(tuple(range(n)), "number")]
    return quasi_dist.expval_and_stddev()


def post_select_quasis(
    quasis: mthree.classes.QuasiDistribution, predicate: Callable[[str], bool]
) -> tuple[mthree.classes.QuasiDistribution, float]:
    """Post-select quasiprobabilities to enforce a given bitstring predicate.

    Returns:
        - Post-selected quasiprobabilities
        - total quasiprobability mass removed
    """
    new_quasis = quasis.copy()
    removed_mass = 0.0
    # postselect
    for bitstring in new_quasis:
        if not predicate(bitstring):
            removed_mass += new_quasis[bitstring]
            new_quasis[bitstring] = 0.0
    # normalize
    normalization = sum(new_quasis.values())
    for bitstring in new_quasis:
        new_quasis[bitstring] /= normalization
    return (
        mthree.classes.QuasiDistribution(
            new_quasis,
            shots=int(quasis.shots * (1 - removed_mass)),
            mitigation_overhead=quasis.mitigation_overhead,
        ),
        removed_mass,
    )


def counts_to_quasis(counts: dict[str, int]) -> mthree.classes.QuasiDistribution:
    """Convert counts to quasiprobabilities."""
    shots = sum(counts.values())
    data = {bitstring: count / shots for bitstring, count in counts.items()}
    return mthree.classes.QuasiDistribution(data, shots=shots, mitigation_overhead=1.0)


def purify_idempotent_matrix(
    mat: np.ndarray, tol: float = 1e-8, max_iter: int = 1000
) -> np.ndarray:
    """McWeeny purification of an idempotent matrix."""
    dim, _ = mat.shape
    three = 3 * np.eye(dim, dtype=mat.dtype)
    error = np.inf
    iterations = 0
    while error > tol and iterations < max_iter:
        mat = mat @ mat @ (three - 2 * mat)
        error = cast(float, np.linalg.norm(mat @ mat - mat))
        iterations += 1
    if error > tol:
        raise RuntimeError("Purification failed to converge.")
    return mat


def pick_qubit_layout(
    n_modes: int, backend_name: str, provider: Optional[Provider] = None
) -> tuple[list[int], str, float]:
    """Pick qubit layout using mapomatic."""
    if provider is None:
        return list(range(n_modes)), backend_name, 0.0
    backend = get_backend(backend_name, provider)
    tunneling = -1.0
    superconducting = 1.0
    chemical_potential = 1.0
    hamiltonian = kitaev_hamiltonian(
        n_modes=n_modes,
        tunneling=tunneling,
        superconducting=superconducting,
        chemical_potential=chemical_potential,
    )
    transformation_matrix, _, _ = hamiltonian.diagonalizing_bogoliubov_transform()
    occupied_orbitals = tuple(range(n_modes // 2))
    circuit = FermionicGaussianState(
        transformation_matrix, occupied_orbitals=occupied_orbitals
    )
    # TODO check that mapomatic returns a line
    return mapomatic.best_overall_layout(circuit.decompose(), [backend])


def orbital_permutations(n_modes: int) -> Iterable[tuple[int, ...]]:
    """Orbital permutations used to measure the full correlation matrix."""
    permutation = list(range(n_modes))
    for _ in range(math.ceil(n_modes / 2)):
        yield tuple(permutation)
        for i in range(0, n_modes - 1, 2):
            a, b = permutation[i], permutation[i + 1]
            permutation[i], permutation[i + 1] = b, a
        for i in range(1, n_modes - 1, 2):
            a, b = permutation[i], permutation[i + 1]
            permutation[i], permutation[i + 1] = b, a


def measurement_labels(n_modes: int) -> Iterable[tuple[tuple[int, ...], str]]:
    """Measurement labels for experiment circuits."""
    yield tuple(range(n_modes)), "number"
    for permutation in orbital_permutations(n_modes):
        yield permutation, "tunneling_plus_even"
        yield permutation, "tunneling_plus_odd"
        yield permutation, "tunneling_minus_even"
        yield permutation, "tunneling_minus_odd"
        yield permutation, "superconducting_plus_even"
        yield permutation, "superconducting_plus_odd"
        yield permutation, "superconducting_minus_even"
        yield permutation, "superconducting_minus_odd"


def transpile_circuit(
    circuit: QuantumCircuit,
    backend: Backend,
    initial_layout: Optional[list[int]] = None,
    dynamical_decoupling_sequence: Optional[str] = None,
    pulse_scaling: bool = False,
    pauli_twirling: bool = False,
    seed: Any = None,
) -> QuantumCircuit:
    """Transpile an experiment circuit."""
    pass_manager = PassManager(
        list(
            transpilation_passes(
                circuit,
                backend,
                initial_layout,
                dynamical_decoupling_sequence,
                pulse_scaling,
                pauli_twirling,
                seed,
            )
        )
    )
    transpiled = pass_manager.run(circuit)
    if dynamical_decoupling_sequence:
        add_pulse_calibrations(transpiled, backend)
    return transpiled


def transpilation_passes(
    circuit: QuantumCircuit,
    backend: Backend,
    initial_layout: Optional[list[int]] = None,
    dynamical_decoupling_sequence: Optional[str] = None,
    pulse_scaling: bool = False,
    pauli_twirling: bool = False,
    seed: Any = None,
) -> Iterator[BasePass]:
    """Transpilation passes for experiment circuits."""
    backend_config = backend.configuration()
    # qubit layout
    if initial_layout is None:
        yield VF2Layout(CouplingMap(backend_config.coupling_map))
    else:
        yield SetLayout(Layout.from_intlist(initial_layout, circuit.qregs[0]))
    yield FullAncillaAllocation(CouplingMap(backend_config.coupling_map))
    yield EnlargeWithAncilla()
    yield ApplyLayout()
    # gate decomposition
    if pulse_scaling:
        # decompose to rzx and scaled pulses
        inst_sched_map = backend.defaults().instruction_schedule_map
        channel_map = backend.configuration().qubit_channel_mapping

        yield XXPlusYYtoRZX()
        yield XXMinusYYtoRZX()
        yield CombineRuns(["rzx"])
        if pauli_twirling:
            yield PauliTwirl(seed=seed)
        yield RZXtoEchoedCR(inst_sched_map)
        yield Optimize1qGatesDecomposition(BASIS_GATES)
        yield CombineRuns(["rz"])
        yield SECRCalibrationBuilder(inst_sched_map, channel_map)
    else:
        # standard decomposition
        yield UnrollCustomDefinitions(
            SessionEquivalenceLibrary, ["id", "rz", "sx", "x", "cx", "reset"]
        )
        yield BasisTranslator(
            SessionEquivalenceLibrary, ["id", "rz", "sx", "x", "cx", "reset"]
        )
        if pauli_twirling:
            yield PauliTwirl(seed=seed)
        yield Optimize1qGatesDecomposition(BASIS_GATES)
    # add dynamical decoupling if needed
    if dynamical_decoupling_sequence:
        yield from dynamical_decoupling_passes(
            backend, dynamical_decoupling_sequence, ALAPScheduleAnalysis
        )
