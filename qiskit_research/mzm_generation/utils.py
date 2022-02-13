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

import functools
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    FrozenSet,
    Optional,
    Tuple,
    Union,
)

import mthree
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XYGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.operators.second_quantization import (
    FermionicOp,
    QuadraticHamiltonian,
)
from qiskit_research.mzm_generation.phased_xx_minus_yy import PhasedXXMinusYYGate

if TYPE_CHECKING:
    from qiskit_research.mzm_generation.experiment import KitaevHamiltonianExperiment


_CovarianceDict = Dict[FrozenSet[Tuple[int, int]], float]


def majorana_op(index: int, action: int) -> FermionicOp:
    if action == 0:
        return FermionicOp(f"-_{index}") + FermionicOp(f"+_{index}")
    return -1j * (FermionicOp(f"-_{index}") - FermionicOp(f"+_{index}"))


def edge_correlation_op(n_modes: int) -> FermionicOp:
    return -1j * majorana_op(0, 0) @ majorana_op(n_modes - 1, 1)


def number_op(n_modes: int) -> FermionicOp:
    return sum(FermionicOp(f"N_{i}") for i in range(n_modes))


def expectation(operator: np.ndarray, state: np.ndarray) -> complex:
    """Expectation value of operator with state."""
    return np.vdot(state, operator @ state)


def variance(operator: np.ndarray, state: np.ndarray) -> complex:
    """Variance of operator with state."""
    return expectation(operator @ operator, state) - expectation(operator, state) ** 2


def jordan_wigner(op: FermionicOp) -> SparsePauliOp:
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
    return QuadraticHamiltonian(hermitian_part, antisymmetric_part)


def bdg_hamiltonian(hamiltonian: QuadraticHamiltonian) -> np.ndarray:
    return np.block(
        [
            [
                -hamiltonian.hermitian_part.conj(),
                -hamiltonian.antisymmetric_part.conj(),
            ],
            [hamiltonian.antisymmetric_part, hamiltonian.hermitian_part],
        ]
    )


def correlation_matrix(state: np.ndarray) -> np.ndarray:
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
            corr[j, i] = val.conj()
            corr[i + n, j + n] = float(i == j) - val.conj()
            corr[j + n, i + n] = float(i == j) - val

            op = FermionicOp(f"-_{i} -_{j}", register_length=n)
            op_jw = jordan_wigner(op).to_matrix()
            val = expectation(op_jw, state)
            corr[i + n, j] = val
            corr[j + n, i] = -val
            corr[i, j + n] = -val.conj()
            corr[j, i + n] = val.conj()

    return corr


def covariance_matrix(corr: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to covariance matrix."""
    n, _ = corr.shape
    eye = np.eye(n // 2)
    majorana_basis = np.block([[eye, eye], [1j * eye, -1j * eye]]) / np.sqrt(2)
    return np.real(
        -1j * majorana_basis @ (2 * corr - np.eye(n)) @ majorana_basis.T.conj()
    )


def fidelity_witness(
    corr: np.ndarray,
    corr_target: np.ndarray,
    cov: Optional[_CovarianceDict] = None,
) -> Tuple[float, float]:
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
    witness = 1 + np.trace((corr - corr_target) @ (corr_target - 0.5 * np.eye(dim)))

    # compute variance
    var = 0.0
    if cov is not None:
        # off-diagonal entries
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(n):
                    for ell in range(k + 1, n):
                        var += 2 * np.real(
                            corr_target[i, j].conjugate()
                            * corr_target[k, ell].conjugate()
                            * cov[frozenset([(i, j), (k, ell)])]
                        )
                        var += 2 * np.real(
                            corr_target[i, j].conjugate()
                            * corr_target[k, ell]
                            * cov[frozenset([(i, j), (k, ell)])]
                        )
                        var += 2 * np.real(
                            corr_target[i, j + n].conjugate()
                            * corr_target[k, ell + n].conjugate()
                            * cov[frozenset([(i, j + n), (k, ell + n)])]
                        )
                        var += 2 * np.real(
                            corr_target[i, j + n].conjugate()
                            * corr_target[k, ell + n]
                            * cov[frozenset([(i, j + n), (k, ell + n)])]
                        )
        # diagonal entries
        for i in range(n):
            for j in range(i, n):
                var += (1 + (i != j)) * (
                    (corr_target[i, i] - float(i == j))
                    * (corr_target[j, j] - float(i == j))
                    * cov[frozenset([(i, i), (j, j)])]
                )
        # account for lower half of correlation matrices
        var *= 4

    return np.real(witness), np.sqrt(np.real(var))


def expectation_from_correlation_matrix(
    operator: Union[QuadraticHamiltonian, FermionicOp],
    corr: np.ndarray,
    cov: Optional[_CovarianceDict] = None,
) -> Tuple[complex, float]:
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
        var = 0.0
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
        for term, coeff in operator._data:
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
        var = 0.0
        if cov is not None:
            # HACK FermionicOp should support iteration with public API
            # See https://github.com/Qiskit/qiskit-nature/issues/541
            for term_ij, coeff_ij in operator._data:
                if not term_ij:
                    continue
                (action_i, i), (action_j, j) = term_ij
                if i > j:
                    i, j = j, i
                    action_i, action_j = action_j, action_i
                for term_kl, coeff_kl in operator._data:
                    if not term_kl:
                        continue
                    (action_k, k), (action_l, ell) = term_kl
                    if k > ell:
                        k, ell = ell, k
                        action_k, action_l = action_l, action_k
                    var += (
                        coeff_ij
                        * coeff_kl.conjugate()
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
        gate = XYGate(np.pi / 2, -np.pi / 2)
    elif label.startswith("tunneling_minus"):
        # this gate transforms -i * (a^\dagger_i a_j - h.c) into (IZ - ZI) / 2
        gate = XYGate(np.pi / 2, -np.pi)
    elif label.startswith("superconducting_plus"):
        # this gate transforms a^\dagger_i a^_\dagger_j + h.c into (IZ + ZI) / 2
        gate = PhasedXXMinusYYGate(np.pi / 2, -np.pi / 2)
    else:  # label.startswith("superconducting_minus")
        # this gate transforms -i * (a^\dagger_i a^_\dagger_j - h.c) into (IZ + ZI) / 2
        gate = PhasedXXMinusYYGate(np.pi / 2, -np.pi)

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
    quasis: Dict[str, Dict[str, float]], experiment: "KitaevHamiltonianExperiment"
) -> Tuple[np.ndarray, _CovarianceDict]:
    """Compute correlation matrix from quasiprobabilities.

    Returns:
        - Correlation matrix
        - Dictionary containing covariances between entries of the correlation matrix
    """
    n = experiment.n_modes

    # off-diagonal entries
    tunneling_plus, tunneling_plus_cov = compute_interaction_matrix(
        quasis, experiment, "tunneling_plus"
    )
    tunneling_minus, tunneling_minus_cov = compute_interaction_matrix(
        quasis, experiment, "tunneling_minus"
    )
    superconducting_plus, superconducting_plus_cov = compute_interaction_matrix(
        quasis, experiment, "superconducting_plus"
    )
    superconducting_minus, superconducting_minus_cov = compute_interaction_matrix(
        quasis, experiment, "superconducting_minus"
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
        # don't reverse pauli string because M3 does it internally!
        num = "I" * i + "1" + "I" * (n - i - 1)
        expval = num_quasis.expval(num)
        corr[i, i] = expval
        corr[i + n, i + n] = 1 - expval

    # covariance
    cov = defaultdict(float)  # _CovarianceDict
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
        z0 = "I" * i + "Z" + "I" * (n - i - 1)
        for j in range(i, n):
            z1 = "I" * j + "Z" + "I" * (n - j - 1)
            cov[frozenset([(i, i), (j, j)])] = 0.25 * covariance(num_quasis, z0, z1)

    return corr, cov


def compute_interaction_matrix(
    quasis: Dict[str, Dict[str, float]],
    experiment: "KitaevHamiltonianExperiment",
    label: str,
) -> Tuple[np.ndarray, _CovarianceDict]:
    """Compute interaction matrix from quasiprobabilities.

    Returns:
        - Interaction matrix
        - Dictionary containing covariances between entries of the interaction matrix
    """
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

    n = experiment.n_modes

    # compute interaction matrix
    mat = np.zeros((n, n))
    for permutation in experiment.permutations():
        even_quasis = quasis[permutation, f"{label}_even"]
        odd_quasis = quasis[permutation, f"{label}_odd"]
        for start_index in [0, 1]:
            quasi_dist = odd_quasis if start_index else even_quasis
            for i in range(start_index, n - 1, 2):
                # don't reverse pauli string because M3 does it internally!
                z0 = "I" * i + "Z" + "I" * (n - i - 1)
                z1 = "I" * (i + 1) + "Z" + "I" * (n - i - 2)
                z0_expval = quasi_dist.expval(z0)
                z1_expval = quasi_dist.expval(z1)
                val = 0.5 * (z1_expval + sign * z0_expval)
                p, q = permutation[i], permutation[i + 1]
                mat[p, q] = val
                mat[q, p] = symmetry * val

    # compute covariance
    cov = defaultdict(float)  # _CovarianceDict
    for permutation in experiment.permutations():
        even_quasis = quasis[permutation, f"{label}_even"]
        odd_quasis = quasis[permutation, f"{label}_odd"]
        for start_index in [0, 1]:
            quasi_dist = odd_quasis if start_index else even_quasis
            for i in range(start_index, n - 1, 2):
                z0 = "I" * i + "Z" + "I" * (n - i - 1)
                z1 = "I" * (i + 1) + "Z" + "I" * (n - i - 2)
                p, q = permutation[i], permutation[i + 1]
                if p > q:
                    p, q = q, p
                for j in range(start_index, n - 1, 2):
                    z2 = "I" * j + "Z" + "I" * (n - j - 1)
                    z3 = "I" * (j + 1) + "Z" + "I" * (n - j - 2)
                    r, s = permutation[j], permutation[j + 1]
                    if r > s:
                        r, s = s, r
                    cov[frozenset([(p, q), (r, s)])] = 0.25 * (
                        covariance(quasi_dist, z0, z2)
                        - covariance(quasi_dist, z0, z3)
                        - covariance(quasi_dist, z1, z2)
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


def evaluate_diagonal_op(operator: str, bitstring: str):
    """Evaluate a diagional operator on a bitstring."""
    prod = 1.0
    # reverse bitstring because Qiskit uses little endian
    for op, bit in zip(operator, reversed(bitstring)):
        if op == "0" or op == "1":
            prod *= bit == op
        elif op == "Z":
            prod *= (-1) ** (bit == "1")
    return prod


def compute_parity(quasis: Dict[str, Dict[str, float]]) -> Tuple[float, float]:
    """Compute parity from quasiprobabilities."""
    # TODO maybe use probs instead of quasis to avoid value outside [-1, 1]
    n = len(next(iter(next(iter(quasis.values())))))
    quasi_dist = quasis[(tuple(range(n)), "number")]
    return quasi_dist.expval_and_stddev()


def post_select_quasis(
    quasis: mthree.classes.QuasiDistribution, predicate: Callable[[str], bool]
) -> Tuple[mthree.classes.QuasiDistribution, float]:
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
            removed_mass += abs(new_quasis[bitstring])
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


def counts_to_quasis(counts: Dict[str, int]) -> mthree.classes.QuasiDistribution:
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
        error = np.linalg.norm(mat @ mat - mat)
        iterations += 1
    if error > tol:
        raise RuntimeError("Purification failed to converge.")
    return mat
