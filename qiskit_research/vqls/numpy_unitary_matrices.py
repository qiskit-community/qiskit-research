from types import SimpleNamespace
from typing import Optional, Union, List, Tuple
import numpy as np
import scipy.linalg as spla
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator


class UnitaryDecomposition:
    r"""Compute the unitary decomposition of a general matrix
    See:
        https://math.stackexchange.com/questions/1710247/every-matrix-can-be-written-as-a-sum-of-unitary-matrices/1710390#1710390
    """

    def __init__(
        self,
        matrix: Optional[np.ndarray] = None,
        circuits: Optional[Union[QuantumCircuit, List[QuantumCircuit]]] = None,
        coefficients: Optional[
            Union[float, complex, List[float], List[complex]]
        ] = None,
        check_decomposition: Optional[bool] = True,
        normalize_coefficients: Optional[bool] = True,
    ):
        """Unitary decomposition

        Args:
            matrix (Optional[np.ndarray], optional): input matrix to be transformed.
            circuit (Optional[Union[QuantumCircuit, List[QuantumCircuit]]], optional): quantum circuit(s) representing the matrix.
            coefficients (Optional[Union[float, complex, List[float], List[complex]]], optional): coefficients of associated with the input quantum circuits.
            check_decomposition (Optional[bool], optional): Check if the decomposition matches the input matrix. Defaults to True.
            normalize_coefficients (Optional[bool], optional): normalize the coefficients of the decomposition. Defaults to True.
        """

        self._matrix = None
        self.matrix = matrix

        self._circuits = None
        self.circuits = circuits

        self._coefficients = None
        self.coefficients = coefficients

        self._unitary_matrices = None

        self.iiter = None

        # if circuits are provided
        if self._circuits is not None:

            # case where the cioefficients are not provided
            if self._coefficients is None:
                if len(self._circuits) == 1:
                    self.coefficients = [1.0]
                else:
                    raise ValueError(
                        "Value of coefficients must be provided for multiple circuits"
                    )

            # check that we have same number of coefficients and circuits
            if len(self._circuits) != len(self._coefficients):
                raise ValueError(
                    "different number of coefficients and circuits provided as input"
                )

            # set the number of qubits and checkthe size of all circuits
            self.num_qubits = self._circuits[0].num_qubits
            for qc in self._circuits:
                if qc.num_qubits != self.num_qubits:
                    raise ValueError("All circuits must have the same number of qubits")

            self._unitary_matrices = [Operator(qc).data for qc in self._circuits]

        # if a matrix is provided
        elif self._matrix is not None:

            if self._circuits is not None:
                raise ValueError(
                    "Circuits cannot be provided if matrix is provided as input"
                )

            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Input matrix must be square!")

            if np.log2(matrix.shape[0]) % 1 != 0:
                raise ValueError("Input matrix dimension must be 2^n!")

            self.coefficients, self.unitary_matrices = self.decompose_numpy_matrix(
                check=check_decomposition, normalize_coefficients=normalize_coefficients
            )

            self.num_qubits = int(np.log2(matrix.shape[0]))
            self.circuits = self.create_circuits(self.unitary_matrices)

        self.num_circuits = len(self._circuits)

    @property
    def matrix(self) -> np.ndarray:
        """return the matrix of the decomposition."""
        if self._matrix is None:
            self._matrix = self.recompose(self.coefficients, self.unitary_matrices)
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: np.ndarray) -> None:
        """Sets the matrix"""
        self._matrix = matrix

    @property
    def circuits(self) -> List[QuantumCircuit]:
        """return the circuits of the decomposition."""
        return self._circuits

    @circuits.setter
    def circuits(self, circuits: Union[QuantumCircuit, List[QuantumCircuit]]) -> None:
        """Sets the matrix"""
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        self._circuits = circuits

    @property
    def coefficients(self) -> Union[List[float], List[complex]]:
        """return the coefficients of the decomposition."""
        return self._coefficients

    @coefficients.setter
    def coefficients(
        self, coefficients: Union[float, complex, List[float], List[complex]]
    ) -> None:
        """Sets the matrix"""
        if not isinstance(coefficients, List):
            coefficients = [coefficients]
        self._coefficients = [c for c in np.array(coefficients).astype(np.cdouble)]

    @property
    def num_qubits(self) -> int:
        """return the numner of qubits"""
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits"""
        self._num_qubits = num_qubits

    @property
    def unitary_matrices(self) -> int:
        """return the unitary matrices"""
        return self._unitary_matrices

    @unitary_matrices.setter
    def unitary_matrices(
        self, unitary_matrices: Union[np.ndarray, List[np.ndarray]]
    ) -> None:
        """Set the number of qubits"""
        if isinstance(unitary_matrices, np.ndarray):
            unitary_matrices = [unitary_matrices]
        self._unitary_matrices = unitary_matrices

    def __iter__(self):
        self.iiter = 0
        return self

    def __next__(self):
        if self.iiter < self.num_circuits:
            out = SimpleNamespace(
                coeff=self._coefficients[self.iiter], circuit=self._circuits[self.iiter]
            )
            self.iiter += 1
            return out
        raise StopIteration

    def __len__(self):
        return len(self._circuits)

    def __getitem__(self, index):
        return SimpleNamespace(
            coeff=self._coefficients[index], circuit=self._circuits[index]
        )

    @staticmethod
    def get_auxilliary_matrix(x: np.ndarray) -> np.ndarray:
        """Compute i * sqrt(I - x^2)

        Args:
            x (np.ndarray): input matrix

        Returns:
            np.ndarray: values of i * sqrt(I - x^2)
        """
        return 1.0j * spla.sqrtm(np.eye(len(x)) - x @ x)

    def decompose_numpy_matrix(
        self,
        check: Optional[bool] = False,
        normalize_coefficients: Optional[bool] = False,
    ) -> Tuple[List[float], List[np.ndarray]]:
        """Decompose a generic numpy matrix into a sum of unitary matrices

        Args:
            check (Optional[bool], optional): _description_. Defaults to False.
            normalize_coefficients (Optional[bool], optional): _description_. Defaults to False.

        Returns:
            Tuple: list of coefficients and numpy matrix of the decompostion
        """

        # Normalize
        norm = np.linalg.norm(self._matrix)
        mat = self._matrix / norm

        mat_real = np.real(mat)
        mat_imag = np.imag(mat)

        coef_real = norm * 0.5
        coef_imag = coef_real * 1j

        ## Get the matrices
        unitary_matrices, unitary_coefficients = [], []
        if not np.allclose(mat_real, 0.0):
            aux_mat = self.get_auxilliary_matrix(mat_real)
            unitary_matrices += [mat_real + aux_mat, mat_real - aux_mat]
            unitary_coefficients += [coef_real] * 2

        if not np.allclose(mat_imag, 0.0):
            aux_mat = self.get_auxilliary_matrix(mat_imag)
            unitary_matrices += [mat_imag + aux_mat, mat_imag - aux_mat]
            unitary_coefficients += [coef_imag] * 2

        if check:
            mat_recomp = self.recompose(unitary_coefficients, unitary_matrices)
            assert np.allclose(self._matrix, mat_recomp)

        if normalize_coefficients:
            unitary_coefficients = self.normalize_coefficients(unitary_coefficients)

        return unitary_coefficients, unitary_matrices

    def normalize_coefficients(self, unit_coeffs: List[float]) -> List[float]:
        """Normalize the coefficients

        Args:
            unit_coeffs (List[float]): list of coefficients

        Returns:
            List[float]: List of normalized coefficients
        """
        sum_coeff = np.array(unit_coeffs).sum()
        return [u / sum_coeff for u in unit_coeffs]

    def recompose(
        self, unit_coeffs: List[float], unit_mats: List[np.ndarray]
    ) -> np.ndarray:
        """Rebuilds the original matrix from the decomposed one.

        Args:
            unit_coeffs (List[float]): coefficients of the decomposition
            unit_mats (List[np.ndarray]): matrices of the decomposition

        Returns:
            np.ndarray: recomposed matrix
        """
        recomp = np.zeros_like(unit_mats[0])
        for c, m in zip(unit_coeffs, unit_mats):
            recomp += c * m
        return recomp

    def create_circuits(self, unit_mats: List[np.ndarray]) -> List[QuantumCircuit]:
        """Contstruct the quantum circuits.

        Args:
            unit_mats (List[np.ndarray]): list of unitary matrices of the decomposition.

        Returns:
            List[QuantumCircuit]: list of resulting quantum circuits.
        """
        circuits = []
        for m in unit_mats:
            qc = QuantumCircuit(self.num_qubits)
            qc.unitary(m, qc.qubits)
            circuits.append(qc)
        return circuits
