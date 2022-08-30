# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test VQLS """

import logging
import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
from test.python.transpiler._dummy_passes import DummyAP

from functools import partial
import numpy as np
from scipy.optimize import minimize as scipy_minimize
from ddt import data, ddt, idata, unpack
from qiskit.algorithms.linear_solvers.numpy_linear_solver import NumPyLinearSolver

from qiskit import BasicAer, QuantumCircuit
from qiskit.algorithms import VQLS, AlgorithmError
from qiskit.algorithms.optimizers import (
    CG,
    COBYLA,
    L_BFGS_B,
    P_BFGS,
    QNSPSA,
    SLSQP,
    SPSA,
    TNC,
    OptimizerResult,
)
from qiskit.circuit.library import EfficientSU2, RealAmplitudes, TwoLocal
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.opflow import (
    AerPauliExpectation,
    Gradient,
    I,
    MatrixExpectation,
    PauliExpectation,
    PauliSumOp,
    PrimitiveOp,
    TwoQubitReduction,
    X,
    Z,
)
from qiskit.quantum_info import Statevector
from qiskit.transpiler import PassManager, PassManagerConfig
from qiskit.transpiler.preset_passmanagers import level_1_pass_manager
from qiskit.utils import QuantumInstance, algorithm_globals, has_aer

from qiskit.algorithms.linear_solvers.matrices.tridiagonal_toeplitz import TridiagonalToeplitz
from qiskit.algorithms.linear_solvers.matrices.numpy_matrix import NumPyMatrix

from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes

from qiskit.algorithms.linear_solvers.matrices import UnitaryDecomposition

from qiskit.quantum_info import Operator

if has_aer():
    from qiskit import Aer

logger = "LocalLogger"


class LogPass(DummyAP):
    """A dummy analysis pass that logs when executed"""

    def __init__(self, message):
        super().__init__()
        self.message = message

    def run(self, dag):
        logging.getLogger(logger).info(self.message)


# pylint: disable=invalid-name, unused-argument
def _mock_optimizer(fun, x0, jac=None, bounds=None) -> OptimizerResult:
    """A mock of a callable that can be used as minimizer in the VQE."""
    result = OptimizerResult()
    result.x = np.zeros_like(x0)
    result.fun = fun(result.x)
    result.nit = 0
    return result


@ddt
class TestVQLS(QiskitAlgorithmsTestCase):
    """Test VQLS"""

    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed
        

        self.qasm_simulator = QuantumInstance(
            BasicAer.get_backend("qasm_simulator"),
            shots=1024,
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )

        self.statevector_simulator = QuantumInstance(
            BasicAer.get_backend("statevector_simulator"),
            shots=1,
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )

    @idata(
    [
        [
            np.array([
            [0.50, 0.25, 0.10, 0.00],
            [0.25, 0.50, 0.25, 0.10],
            [0.10, 0.25, 0.50, 0.25],
            [0.00, 0.10, 0.25, 0.50] ]),
            np.array([0.1]*4),
            RealAmplitudes(num_qubits=2, reps=3, entanglement='full'),
        ],
    ])
    @unpack
    def test_numpy_input_statevector(self, matrix, rhs, ansatz):
        """Test the VQLS on matrix input using statevector simulator."""
        
        classical_solution = NumPyLinearSolver().solve(matrix, rhs/np.linalg.norm(rhs))
        
        vqls = VQLS(
            ansatz=ansatz,
            quantum_instance=self.statevector_simulator,
        )
        res = vqls.solve(matrix, rhs)

        ref_solution = np.abs(classical_solution.state / np.linalg.norm(classical_solution.state))
        vqls_solution = np.abs(np.real(Statevector(res.state).data))
        
        with self.subTest(msg="test solution"):
            assert np.allclose(ref_solution, vqls_solution, atol=1E-1, rtol=1E-1)


    def test_circuit_input_statevector(self):
        """Test the VQLS on circuits input using statevector simulator."""

        num_qubits = 2
        ansatz = RealAmplitudes(num_qubits=num_qubits, reps=3, entanglement='full')

        rhs = QuantumCircuit(num_qubits)
        rhs.h(0)
        rhs.h(1)

        qc1 = QuantumCircuit(num_qubits)
        qc1.x(0)
        qc1.x(1)
        qc1.cnot(0,1)

        qc2 = QuantumCircuit(num_qubits)
        qc2.h(0)
        qc2.x(1)
        qc1.cnot(0,1)

        matrix = UnitaryDecomposition(
            circuits = [qc1, qc2],
            coefficients = [0.5, 0.5]
        )

        np_matrix = matrix.recompose()
        np_rhs = Operator(rhs).data @ np.array([1,0,0,0])

        classical_solution = NumPyLinearSolver().solve(np_matrix, np_rhs/np.linalg.norm(np_rhs))
        
        vqls = VQLS(
            ansatz=ansatz,
            quantum_instance=self.statevector_simulator,
        )
        res = vqls.solve(matrix, rhs)

        ref_solution = np.abs(classical_solution.state / np.linalg.norm(classical_solution.state))
        vqls_solution = np.abs(np.real(Statevector(res.state).data))
        
        with self.subTest(msg="test solution"):
            assert np.allclose(ref_solution, vqls_solution, atol=1E-1, rtol=1E-1)


if __name__ == "__main__":
    unittest.main()
