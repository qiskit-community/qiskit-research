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

"""Utilities for dealing with backends."""


from typing import Optional

from qiskit import BasicAer
from qiskit.providers import Backend, Provider
from qiskit_aer import AerSimulator


def get_backend(
    name: str, provider: Optional[Provider], seed_simulator: Optional[int] = None
) -> Backend:
    """Retrieve a backend."""
    if provider is not None:
        return provider.get_backend(name)
    if name == "aer_simulator":
        return AerSimulator(seed_simulator=seed_simulator)
    if name == "statevector_simulator":
        return BasicAer.get_backend("statevector_simulator")
    raise ValueError("The given name does not match any supported backends.")
