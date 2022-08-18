# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""ProteinFolding Test Case"""

from typing import Optional
from abc import ABC
import inspect
import os
import unittest
import time

# disable deprecation warnings that can cause log output overflow
# pylint: disable=unused-argument


def _noop(*args, **kargs):
    pass


# disable warning messages
# warnings.warn = _noop


class ProteinFoldingTestCase(unittest.TestCase, ABC):
    """Protein Folding Test Case"""

    moduleName = None

    def setUp(self) -> None:
        self._started_at = time.time()
        self._class_location = __file__

    def tearDown(self) -> None:
        elapsed = time.time() - self._started_at
        if elapsed > 5.0:
            print(f"({round(elapsed, 2):.2f}s)", flush=True)

    @classmethod
    def setUpClass(cls) -> None:
        cls.moduleName = os.path.splitext(inspect.getfile(cls))[0]

    def get_resource_path(self, filename: str, path: Optional[str] = None) -> str:
        """Get the absolute path to a resource.
        Args:
            filename: filename or relative path to the resource.
            path: path used as relative to the filename.
        Returns:
            str: the absolute path to the resource.
        """
        root = os.path.dirname(self._class_location)
        path = root if path is None else os.path.join(root, path)
        return os.path.normpath(os.path.join(path, filename))
