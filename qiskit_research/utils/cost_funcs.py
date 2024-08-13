# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Cost functions."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import Target
from qiskit.transpiler.passes.layout.vf2_utils import (
    build_average_error_map,
    build_interaction_graph,
    score_layout,
)


def avg_error_score(qc_isa: QuantumCircuit, target: Target) -> float:
    """Calculate average error score using vf2 utils

    Args:
        qc_isa (QuantumCircuit): transpiled circuit
        target (Target): backend target

    Returns:
        float: average error score determined by average error map
    """
    init_layout = qc_isa.layout.final_index_layout()
    dag = circuit_to_dag(qc_isa)

    layout = {idx: init_layout[idx] for idx in range(len(init_layout))}
    avg_error_map = build_average_error_map(target, None, None)
    im_graph, im_graph_node_map, reverse_im_graph_node_map, _ = build_interaction_graph(
        dag, strict_direction=False
    )
    return score_layout(
        avg_error_map, layout, im_graph_node_map, reverse_im_graph_node_map, im_graph
    )
