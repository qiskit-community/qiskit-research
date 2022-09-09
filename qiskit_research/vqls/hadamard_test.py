"""Hadammard test."""

from typing import Optional, List, Union
from qiskit import QuantumCircuit


class HadammardTest:
    r"""Class to compute the Hadamard Test
    """
    def __init__(
        self,
        operators: Union[QuantumCircuit, List[QuantumCircuit]],
        use_barrier: Optional[bool] = False,
        apply_control_to_operator: Optional[bool] = True,
        apply_initial_state: Optional[QuantumCircuit] = None,
        apply_measurement: Optional[bool] = False,
    ) -> List[QuantumCircuit]:
        r"""Create the quantum circuits required to compute the hadamard test:

        .. math::

            \\langle \\Psi | U | \\Psi \\rangle

        Args:
            operators (Union[QuantumCircuit, List[QuantumCircuit]]): quantum circuit or list of quantum circuits representing the U.
            use_barrier (Optional[bool], optional): introduce barriers in the description of the circuits.  Defaults to False.
            apply_control_to_operator (Optional[bool], optional): Apply control operator to the input quantum circuits. Defaults to True.
            apply_initial_state (Optional[QuantumCircuit], optional): Quantum Circuit to create |Psi> from |0>. If None, assume that the qubits are alredy in Psi.
            apply_measurement (Optional[bool], optional): apply explicit measurement. Defaults to False.

        Returns:
            List[QuantumCircuit]: List of quamtum circuits required to compute the Hadammard Test.
        """

        if isinstance(operators, QuantumCircuit):
            operators = [operators]

        if apply_control_to_operator:
            self.num_qubits = operators[0].num_qubits + 1
            if apply_initial_state is not None:
                if apply_initial_state.num_qubits != operators[0].num_qubits:
                    raise ValueError(
                        "The operator and the initial state circuits have different numbers of qubits"
                    )
        else:
            self.num_qubits = operators[0].num_qubits
            if apply_initial_state is not None:
                if apply_initial_state.num_qubits != operators[0].num_qubits - 1:
                    raise ValueError(
                        "The operator and the initial state circuits have different numbers of qubits"
                    )

        # classical bit for explicit measurement
        self.num_clbits = 1

        # build the circuits
        self.circuits = self._build_circuit(
            operators,
            use_barrier,
            apply_control_to_operator,
            apply_initial_state,
            apply_measurement,
        )

        # number of circuits required
        self.ncircuits = len(self.circuits)

        # var for iterator
        self.iiter = None

    def __iter__(self):
        self.iiter = 0
        return self

    def __next__(self):
        if self.iiter < self.ncircuits:
            out = self.circuits[self.iiter]
            self.iiter += 1
            return out
        raise StopIteration

    def __len__(self):
        return len(self.circuits)

    def __getitem__(self, index):
        return self.circuits[index]

    def _build_circuit(
        self,
        operators: List[QuantumCircuit],
        use_barrier: bool,
        apply_control_to_operator: bool,
        apply_initial_state: Optional[QuantumCircuit] = None,
        apply_measurement: Optional[bool] = False,
    ) -> List[QuantumCircuit]:
        """build the quantum circuits

        Args:
            operators (List[QuantumCircuit]): quantum circuit or list of quantum circuits representing the U.
            use_barrier (bool): introduce barriers in the description of the circuits.
            apply_control_to_operator (bool): Apply control operator to the input quantum circuits.
            apply_initial_state (Optional[QuantumCircuit], optional): Quantum Circuit to create |Psi> from |0>. If None, assume that the qubits are alredy in Psi.  Defaults to None.
            apply_measurement (Optional[bool], optional): apply explicit measurement. Defaults to False.

        Returns:
            List[QuantumCircuit]: List of quamtum circuits required to compute the Hadammard Test.
        """

        circuits = []

        for imaginary in [False, True]:

            if apply_measurement:
                qc = QuantumCircuit(self.num_qubits, self.num_clbits)
            else:
                qc = QuantumCircuit(self.num_qubits)

            if apply_initial_state is not None:
                qc.compose(
                    apply_initial_state, list(range(1, self.num_qubits)), inplace=True
                )

            if use_barrier:
                qc.barrier()

            # hadadmard gate on ctrl qbit
            qc.h(0)

            # Sdg on ctrl qbit
            if imaginary:
                qc.sdg(0)

            if use_barrier:
                qc.barrier()

            # matrix circuit
            for op in operators:
                if apply_control_to_operator:
                    qc.compose(
                        op.control(1),
                        qubits=list(range(0, self.num_qubits)),
                        inplace=True,
                    )
                else:
                    qc.compose(op, qubits=list(range(0, self.num_qubits)), inplace=True)
            if use_barrier:
                qc.barrier()

            # hadamard on ctrl circuit
            qc.h(0)

            # measure
            if apply_measurement:
                qc.measure(0, 0)

            circuits.append(qc)

        return circuits
