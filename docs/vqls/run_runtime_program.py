# from qiskit_ibm_runtime import QiskitRuntimeService
# import numpy as np
# from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
# from qiskit_research.vqls.runtime import vqls_runner
# from qiskit.quantum_info import Statevector
# import matplotlib.pyplot as plt
# from qiskit.algorithms.linear_solvers.numpy_linear_solver import NumPyLinearSolver

# # credential
# ibmq_token = ''
# hub = ''
# group = '' # examnple 'escience'
# project = '' # example qradio

# # start service
# QiskitRuntimeService.save_account(channel="ibm_quantum",
#                                   token=ibmq_token,
#                                   instance=hub+'/'+group+'/'+project,
#                                   overwrite=True)
# service = QiskitRuntimeService()

# # define the system
# A = np.random.rand(4,4)
# A = (A+A.T)
# b = np.random.rand(4)

# # define the ansatz
# ansatz = RealAmplitudes(2, entanglement='full', reps=3, insert_barriers=False)

# # program id
# program_id = ''

# # define backend
# backend = service.backend("simulator_statevector")

# # run the job
# job = vqls_runner(backend, A, b, program_id, ansatz, shots=25000)
# res = job.result()


# # define the classical solution
# classical_solution = NumPyLinearSolver().solve(A,b/np.linalg.norm(b))
# ref_solution = classical_solution.state / np.linalg.norm(classical_solution.state)

# # define the vqls solution
# opt_parameters = dict(zip(ansatz.parameters, res.x))
# solution = ansatz.assign_parameters(opt_parameters)
# vqls_solution = np.real(Statevector(solution).data )

# # plot the results
# plt.scatter(ref_solution, vqls_solution)
# plt.plot([-1,1],[-1,1],'--')
