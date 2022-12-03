from qiskit import IBMQ, transpile, execute
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from qiskit_research.utils.convenience import scale_cr_pulses
from qiskit_research.utils.convenience import attach_cr_pulses
from qiskit_research.utils.convenience import add_pauli_twirls
from qiskit_research.utils.convenience import add_dynamical_decoupling
from qiskit_research.utils.convenience import transpile_paulis
# from qiskit_research.utils  import scale_cr_pulses, attach_cr_pulses, add_pauli_twirls, add_dynamical_decoupling, transpile_paulis


from qiskit.transpiler.passes import RemoveBarriers, RemoveFinalMeasurements
from sympy.combinatorics.partitions import Partition
from sympy.combinatorics.permutations import Permutation
import collections
from qiskit.providers.aer import AerSimulator, QasmSimulator

# The PassManager helps decide how a circuit should be optimized
# (https://qiskit.org/documentation/tutorials/circuits_advanced/04_transpiler_passes_and_passmanager.html)
from qiskit.transpiler import PassManager

# This function will pull pulse-level calibration values to build RZX gates and tell the PassManager to leave RZX gates alone
from qiskit.transpiler.passes import RZXCalibrationBuilderNoEcho, RZXCalibrationBuilder

# import mthree
import numpy as np



class Dual_state_verification:
    """
    Dual state purification class https://doi.org/10.1103/PhysRevA.105.022427
    wrap a list of circuits with the DST error mitigation procedure and compute observables
    include additional techniques such as dynamical decoupling and twirling


    """

    def __init__(
        self)->None:

        """
        Args:
            quantum_circuits: list of circuits to be run
            quantum_instance: backend or quantum_instance used to run the circuits
            ancilla_purification: flag to perform purification on the ancilla, default False
            dynamical_decoupling: flag to perform dynamical_decoupling on the ancilla during its idle, default False
            ancilla_purification: flag to perform twirling, default False
        """






    def getB_arbitrary(self, operator):
        """
        map the observable to Z1, i.e., find B such that BOB^dagger = Z1
        Args:
            operator: observable to be mapped
        Outputs:
            B, Bdagger and  qubit to be linked to the ancilla for the observable
        """

        if not isinstance(operator,list):
            operator = [operator]

        num_qubits = len(operator[0])

        qc = QuantumCircuit(num_qubits)

        # perform a basis change
        for i in range(num_qubits):
            if operator[0][i] == 'X':
                qc.h(i)
            elif operator[0][i] == 'Y':
                qc.h(i)
                qc.sdg(i)

        ##check if the first operator is Z1, if yes go with the original protocol
        #  if not find a another qubit to do the Z1
        new_first_qubit = 0
        if operator[0][0] != 'I':
            for j in range(1,num_qubits):
                if operator[0][j] != 'I':
                    qc.cx(j,0)
        else:
            new_first_qubit = find_arg(operator, 'I')[0]
            for j in range(1,num_qubits):
                if operator[0][j] != 'I' and j != new_first_qubit:
                    qc.cx(j,new_first_qubit)

        return qc, qc.inverse(), new_first_qubit



    def prepare_vd_circuit_arbitrary(self, qc,  operator, vd_ancilla = False, R='Z'):

        """
        wrap the circuit (physical register) with virtual distillation with measurement on ancilla (ancilla register)

        Args:
            qc: original circuit
            R: basis of the ancilla measurement
        Out:
            wrapped circuit
        """
        num_qubits = qc.num_qubits
        self.N = num_qubits

        nbr_ancilla = 1 + int(vd_ancilla)
        self.vd_ancilla = vd_ancilla

        qr_vd = QuantumRegister(num_qubits+  nbr_ancilla)
        cr_vd = ClassicalRegister(num_qubits+ nbr_ancilla)

        qc_vd = QuantumCircuit(qr_vd, cr_vd)

        ### Prepare the ideal circuit (A) ###
        qc_vd.compose(qc, qubits=[i for i in range(nbr_ancilla, num_qubits+nbr_ancilla)],inplace=True)
        qc_vd.barrier()

        ### Get B, B^dag ###
        qc_B, qc_B_inv, new_first_qubit= self.getB_arbitrary(operator)

        ### Prepare the dual circuit (B) ###
        qc_vd.compose(qc_B, qubits=[i for i in range(nbr_ancilla, num_qubits+nbr_ancilla)], inplace=True)

        qc_vd.barrier()

        ### connect with Ancilla
        qc_vd.cx(new_first_qubit+1+int(vd_ancilla), int(vd_ancilla))
        if vd_ancilla:
            qc_vd.swap(0,1)
            qc_vd.cx(new_first_qubit+1+int(vd_ancilla), 1)



        ### measure the Ancilla
        for i in range(nbr_ancilla):
            if R=='X':
                qc_vd.h(i)
            if R=='Y':
                qc_vd.sdg(i)
                qc_vd.h(i)

        if vd_ancilla:
            ### connect the ancillas

            qc_vd.compose(self.circ_B_VD,qubits=[0,1],inplace=True)
            pass



        # qc_vd.barrier()

        ### Prepare the dual circuit (B^dag) ###
        qc_vd.compose(qc_B_inv, qubits=[i for i in range(nbr_ancilla, num_qubits+nbr_ancilla)], inplace=True)

        # qc_vd.barrier()

        ### Prepare the ideal circuit (A^dag) ###

        qc_vd.compose(qc.inverse(), qubits=[i for i in range(nbr_ancilla, num_qubits+nbr_ancilla)], inplace=True)

        qc_vd.measure(qr_vd, cr_vd)

        return qc_vd



    def transpile(self,qc_list, backend, initial_layout = None, num_twirls = 0,
                  dynamical_decoupling= None, pulse_efficient =False, optimization_level = 1, seed = 12345):

        """
        transpile the circuits onto the backend using transpilation techniques for the mitigation
        Args:
            qc_list: list of quantum circuits to be ran
            backend: quantum backend
            inital_layout: choice of layout on the hardware, it is best practice to choose it by hand when running circuits with simple connectivity
            num_twirls: number of twirling repetion for Puali twirling transpilation, Default 0 (no twirling)
            dynamical_decoupling: DD sequence "XY4", "XY8" or None

        """

        basis_gates = ['id','rz','sx','cx','measure','barrier']
        if pulse_efficient:
            basis_gates = ['id','rz','sx','rzx','measure','barrier']

        self._num_twirls = num_twirls
        qc_transpiled = []
        self.final_layout = []

        for qc in qc_list:
            qct =  transpile(qc, backend = backend,
                            initial_layout = initial_layout,
                            optimization_level=optimization_level, basis_gates = basis_gates,  seed_transpiler  = seed)
            # Add the pulse-level calibrations
            if pulse_efficient:
                inst_sched_map = backend.defaults().instruction_schedule_map
                channel_map = backend.configuration().qubit_channel_mapping
                pm = PassManager([RZXCalibrationBuilder(inst_sched_map, channel_map)])
                qct = pm.run(qct)


            if  num_twirls>0:

                qct = add_pauli_twirls(qct, num_twirled_circuits = num_twirls, seed = seed,transpile_added_paulis=False)  #gets only the circuit
                qct = transpile_paulis(qct)

            if dynamical_decoupling is not None:
                try:
                    #pass if the backend does not accept dd
                    qct = add_dynamical_decoupling(qct, backend, dynamical_decoupling, add_pulse_cals=True)
                except:
                    pass
            if qct is not None:
                if isinstance(qct,list):

                    qc_transpiled += qct

                    # self.final_layout.append(list(mthree.utils.final_measurement_mapping(qct[0]).values()))
                else:
                    qc_transpiled.append(qct)
                    # self.final_layout.append(list(mthree.utils.final_measurement_mapping(qct).values()))




        return qc_transpiled



    def post_process(self,counts, split = 10):
        """
        combine the results of the different twilred circuits into one quasi probability distribution
        """

        # split the counts into smaller experiments
        total_counts = split_counts(counts, split = split)
        for c,counts in enumerate(total_counts):

            # calibration_counts = counts.copy()[-2:]
            # counts = counts.copy()[:-2]

            # undo twirling
            if self._num_twirls > 0:
                counts_twirls = counts[1*self._step:]

                quasi_probs = []

                for time_idx in range(int(len(counts_twirls)/self._num_twirls)):
                    quasi_prob_twirled = {}

                    for twidx in range(self._num_twirls):
                        for key in counts_twirls[time_idx * self._num_twirls + twidx].keys():
                            try:
                                quasi_prob_twirled[key]  += (
                                    counts_twirls[time_idx * self._num_twirls + twidx][key] / self._num_twirls
                                )
                            except:
                                quasi_prob_twirled[key]  = (
                                    counts_twirls[time_idx * self._num_twirls + twidx][key] / self._num_twirls
                                )

                    quasi_probs.append(quasi_prob_twirled)

                new_counts = counts[:self._step] + quasi_probs
                total_counts[c] = new_counts


        return total_counts




    def compute_expectation_value(self, total_counts, measured_operator):
        """
        compute the expectation value of measured_operator from the counts
        outputs:
         - expectation  [raw circuit, dual state verification, dual state verification with ancilla purification]
         - purity of the ancilla
        """
        total_expectation = []
        total_purity = []
        for counts in total_counts:
            expectation = []
            purity = []

            expectation.append([raw_expectation_value(c,measured_operator[0][::-1]) for c in counts[:self._step]])
            counts = counts[self._step:].copy()

            counts_copy = counts[:3*self._step]

            expectation.append([])
            expectation.append([])
            purity.append([])

            for j in range(self._step):

                exp_hist = {}

                for i, anc_op in enumerate(['X', 'Y', 'Z']):

                    exp_hist[anc_op] = helper_get_exp_value(counts_copy[3*j+i],self.vd_ancilla)

                expectation[-2].append(np.real(exp_hist['Z']/(1+exp_hist['X'])))
                rho_ancilla = 0.5*(sigmas['I']+sum(sigmas[op]*exp_hist[op] for op in ['X','Y','Z']))
                purity_ancilla = np.sum(np.linalg.eigh(rho_ancilla)[0]**2)
                purity[-1].append(purity_ancilla)

                princ_comp = np.linalg.eigh(rho_ancilla)[1][:,-1]
                exp_purified = {op:np.dot(princ_comp.conj(),sigmas[op]@princ_comp) for op in ['X','Y','Z']}
                expectation[-1].append(np.real(exp_purified['Z']/(1+exp_purified['X'])))
            counts = counts[3*self._step:]

            total_expectation.append(expectation)
            total_purity.append(purity)

        return total_expectation, total_purity






#### Helper Functions ####

def raw_expectation_value(counts, O):
    expectation = 0
    shots = 0
    for i in counts:
        shots += counts[i]

    for key in counts:
        e = 1
        for i,o in enumerate(O):
            if o=='I':
                continue

            if key[i]=='0':

                continue
            e = e*-1

        expectation += e*counts[key]/shots
    return expectation


import random

def split_counts(counts,split):
    """
    split an experiemnt into smaller experiemtn to obtain statistics
    input: list of counts
    split: number of sub experiment

    outputs: list of list of counts (so the first element if the list of counts for the first sub experiement)
    """
    new_counts = []
    shots = np.sum(counts[0][k] for k in counts[0])
    split_shots = int(shots/split)

    for i in range(split):
        new_counts.append([])

    for c in counts:
        measurements = []
        for key in c:
            for _ in range(c[key]):
                measurements.append(key)

        random.shuffle(measurements)
        for s in range(split):
            new_counts[s].append(string_to_dict(measurements[s*split_shots:(s+1)*split_shots]))
    return new_counts

def string_to_dict(binary_string):
    counts = {}
    for b in binary_string:
        if b in counts:
            counts[b]+=1
        else:
            counts[b]=1
    return counts
## find the argument of a character in a string
def find_arg(string, char):
    return [i for i, ltr in enumerate(string[0]) if ltr != char]

# Get expectation values for virtual distillation
sigmas = {'I':np.eye(2),'X':np.array([[0,1],[1,0]]),
                    'Y':np.array([[0,-1j],[1j,0]]),
                    'Z': np.array([[1,0],[0,-1]])}

def helper_get_exp_value(counts, vd_ancilla):

    n = len(list(counts.keys())[0]) # Get the number of qubits as the number of bits in the bitstring
    if vd_ancilla:
        for it in range(2**n):
            new_key = str(bin(it)[2:])

            while len(new_key)<n:
                new_key = '0'+new_key
            if new_key not in counts.keys():

                counts[new_key] = 0


        shots = (counts['0'*(n-2)+'00']+counts['0'*(n-2)+'10'] + counts['0'*(n-2)+'01'] + counts['0'*(n-2)+'11'])
        #
        # Z1 = (counts['0'*(n-2)+'00']+counts['0'*(n-2)+'10'] -counts['0'*(n-2)+'01'] - counts['0'*(n-2)+'11'] )/ shots
        # Z2 = (counts['0'*(n-2)+'00'] -counts['0'*(n-2)+'10'] +counts['0'*(n-2)+'01'] - counts['0'*(n-2)+'11'] )/ shots
        # mix = (counts['0'*(n-2)+'00'] -counts['0'*(n-2)+'10'] - counts['0'*(n-2)+'01'] + counts['0'*(n-2)+'11'] )/ shots
        #
        # E = 0.5*(Z1+Z2)
        # D = 0.5*(1+Z1-Z2+mix)
        #
        # return E/D
        Z1 = (counts['0'*(n-2)+'00']+counts['0'*(n-2)+'10'] -counts['0'*(n-2)+'01'] - counts['0'*(n-2)+'11'] )/ shots
        Z2 = (counts['0'*(n-2)+'00'] -counts['0'*(n-2)+'10'] +counts['0'*(n-2)+'01'] - counts['0'*(n-2)+'11'] )/ shots
        return 0.5*(Z1+Z2)

    else:
        if '0'*(n-1)+'1' not in counts.keys():
            return 1
        elif '0'*(n-1)+'0' not in counts.keys():
            return -1
        else:
            return (counts['0'*n]-counts['0'*(n-1)+'1'])/(counts['0'*n]+counts['0'*(n-1)+'1'])


def expectation_zzz(counts, shots, z_index_list=None):
    """
    :param shots: shots of the experiment
    :param counts: counts obtained from Qiskit's Result.get_counts()
    :param z_index_list: a list of indexes
    :return: the expectation value of ZZ...Z operator for given z_index_list
    """

    if z_index_list is None:
        z_counts = counts
    else:
        z_counts = cut_counts(counts, z_index_list)

    expectation = 0
    for key in z_counts:
        sign = -1
        if key.count('1') % 2 == 0:
            sign = 1
        expectation += sign * z_counts[key] / shots

    return expectation

def cut_counts(counts, bit_indexes):
    """
    :param counts: counts obtained from Qiskit's Result.get_counts()
    :param bit_indexes: a list of indexes
    :return: new_counts for the  specified bit_indexes
    """
    bit_indexes.sort(reverse=True)
    new_counts = {}
    for key in counts:
        new_key = ''
        for index in bit_indexes:
            new_key += key[-1 - index]
        if new_key in new_counts:
            new_counts[new_key] += counts[key]
        else:
            new_counts[new_key] = counts[key]

    return new_counts

def approximate_compiling(unitary, depth, layout='spin',connectivity='linear'):
    """
    compute an approximate quantum circuit of the unitary
    Args:
        unitary (np array): unitary to be compiled
        depth (int): number of cnot in the approximation
        layout (string):  spin, sequ, cart, cyclic_spin, cyclic_line
        connectivity (string): linear, circular, full

    Outputs:
        norm (float): Frobenius norm between approxiamtion and ground truth
        qc (qiskit qc): compilation of the unitary as a quantum circuit
    """

    num_qubits = int(round(np.log2(unitary.shape[0])))

    # Generate a network made of CNOT units
    cnots = make_cnot_network(
        num_qubits=num_qubits,
        network_layout=layout,
        connectivity_type=connectivity,
        depth=depth,
    )

    optimizer = L_BFGS_B()

    # Create an instance
    aqc = AQC(optimizer)

    # Create a template circuit that will approximate our target circuit
    approximate_circuit = CNOTUnitCircuit(num_qubits=num_qubits, cnots=cnots)

    # Create an objective that defines our optimization problem
    approximating_objective = DefaultCNOTUnitObjective(num_qubits=num_qubits, cnots=cnots)

    # Run optimization process to compile the unitary
    aqc.compile_unitary(
        target_matrix=unitary,
        approximate_circuit=approximate_circuit,
        approximating_objective=approximating_objective,
    )


    return  approximate_circuit
