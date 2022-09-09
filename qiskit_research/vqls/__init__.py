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

"""
=======================================
Variational Quantum Linear Solver
=======================================
"""

from qiskit_research.vqls.vqls import VQLS
from qiskit_research.vqls.hadamard_test import HadammardTest
from qiskit_research.vqls.numpy_unitary_matrices import UnitaryDecomposition
from qiskit_research.vqls.variational_linear_solver import (
    VariationalLinearSolver,
    VariationalResult,
)

__all__ = [
    "VQLS",
    "HadammardTest",
    "UnitaryDecomposition",
    "VariationalLinearSolver",
    "VariationalResult",
]
