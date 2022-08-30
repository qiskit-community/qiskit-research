
from multiprocessing.sharedctypes import Value
import numpy as np
from qiskit_ibm_runtime.program import ResultDecoder
from scipy.optimize import OptimizeResult
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes



class VQLSResultDecoder(ResultDecoder):
    @classmethod
    def decode(cls, data):
        data = super().decode(data)  # This is required to preformat the data returned.
        return OptimizeResult(data)

class RuntimeJobWrapper:
    """A simple Job wrapper that attaches interim results directly to the job object itself
    in the `interim_results attribute` via the `_callback` function.
    """

    def __init__(self):
        self._job = None
        self._decoder = VQLSResultDecoder
        self.interim_results = []

    def _callback(self, job_id, xk):
        """The callback function that attaches interim results:

        Parameters:
            job_id (str): The job ID.
            xk (array_like): A list or NumPy array to attach.
        """
        self.interim_results.append(xk)

    def __getattr__(self, attr):
        if attr == "result":
            return self.result
        else:
            if attr in dir(self._job):
                return getattr(self._job, attr)
            raise AttributeError("Class does not have {}.".format(attr))

    def result(self):
        """Get the result of the job as a SciPy OptimizerResult object.

        This blocks until job is done, cancelled, or errors.

        Returns:
            OptimizerResult: A SciPy optimizer result object.
        """
        return self._job.result(decoder=self._decoder)


def vqls_runner(
    backend,
    matrix,
    rhs,
    program_id,
    ansatz,
    x0=None,
    optimizer="SPSA",
    optimizer_config={"maxiter": 100},
    shots=8192,
    use_measurement_mitigation=False,
):

    """Routine that executes a given VQE problem via the sample-vqe program on the target backend.

    Parameters:
        backend (ProgramBackend): Qiskit backend instance.
        matrix (np.ndarray): the matrix of the linear system.
        rhs (np.ndarray): the right hand side of the linear system. 
        ansatz (str): Optional, name of ansatz quantum circuit to use, default='EfficientSU2'
        ansatz_config (dict): Optional, configuration parameters for the ansatz circuit.
        x0 (array_like): Optional, initial vector of parameters.
        optimizer (str): Optional, string specifying classical optimizer, default='SPSA'.
        optimizer_config (dict): Optional, configuration parameters for the optimizer.
        shots (int): Optional, number of shots to take per circuit.
        use_measurement_mitigation (bool): Optional, use measurement mitigation, default=False.

    Returns:
        OptimizeResult: The result in SciPy optimization format.
    """
    options = {"backend_name": backend.name}

    inputs = {}

    # validate the  size
    if matrix.shape[0] != rhs.shape[0]:
        raise ValueError("Matrix size ({}) and rhs size ({}) are incompatible". format(matrix.shape, rhs.shape))

    inputs['matrix'] = matrix
    inputs['rhs'] = rhs

    # number of qubits
    num_qubits = int(np.ceil(np.log2(matrix.shape[0])))

    if num_qubits != int(np.log2(matrix.shape[0])):
        raise ValueError('Ssytem size ({}) is not a power of 2'.format(matrix.shape[0]))


    #
    inputs["ansatz"] = ansatz

    # If given x0, validate its length against num_params in ansatz:
    if x0:
        x0 = np.asarray(x0)
        num_params = ansatz.num_parameters
        if x0.shape[0] != num_params:
            raise ValueError(
                "Length of x0 {} does not match number of params in ansatz {}".format(
                    x0.shape[0], num_params
                )
            )
    inputs["x0"] = x0

    # Set the rest of the inputs
    inputs["optimizer"] = optimizer
    inputs["optimizer_config"] = optimizer_config
    inputs["shots"] = shots
    inputs["use_measurement_mitigation"] = use_measurement_mitigation

    rt_job = RuntimeJobWrapper()
    service = QiskitRuntimeService()
    job = service.run(
        program_id, options=options, inputs=inputs, callback=rt_job._callback
    )
    rt_job._job = job

    return rt_job