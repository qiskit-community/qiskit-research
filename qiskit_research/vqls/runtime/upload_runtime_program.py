############################################
# Upload the vqls_runtime_program to IBMQ
############################################

from qiskit_ibm_runtime import QiskitRuntimeService


def get_metadata():

    meta = {
        "name": "vqls",
        "description": "A VQLS program.",
        "max_execution_time": 100000,
        "spec": {},
    }

    meta["spec"]["parameters"] = {
        "$schema": "https://json-schema.org/draft/2019-09/schema",
        "properties": {
            "matrix": {
                "description": "Matrix of the linear system.",
                "type": "array",
            },
            "rhs": {
                "description": "Right hand side of the linear system.",
                "type": "array",
            },
            "ansatz": {
                "description": "Quantum Circuit of the ansatz",
                "type": "QauntumCircuit",
            },
            "optimizer": {
                "description": "Classical optimizer to use, default='SPSA'.",
                "type": "string",
                "default": "SPSA",
            },
            "x0": {
                "description": "Initial vector of parameters. This is a numpy array.",
                "type": "array",
            },
            "optimizer_config": {
                "description": "Configuration parameters for the optimizer.",
                "type": "object",
            },
            "shots": {
                "description": "The number of shots used for each circuit evaluation.",
                "type": "integer",
            },
            "use_measurement_mitigation": {
                "description": "Use measurement mitigation, default=False.",
                "type": "boolean",
                "default": False,
            },
        },
        "required": ["matrix", "rhs", "ansatz"],
    }

    meta["spec"]["return_values"] = {
        "$schema": "https://json-schema.org/draft/2019-09/schema",
        "description": "Final result in Scipy Optimizer format",
        "type": "object",
    }

    meta["spec"]["interim_results"] = {
        "$schema": "https://json-schema.org/draft/2019-09/schema",
        "description": "Parameter vector at current optimization step. This is a numpy array.",
        "type": "array",
    }

    return meta


# if __name__ == "__main__":

#     # credential
#     ibmq_token = ""
#     hub = ""
#     group = ""
#     project = ""

#     # init the service
#     QiskitRuntimeService.save_account(
#         channel="ibm_quantum", token=ibmq_token, overwrite=True
#     )
#     service = QiskitRuntimeService()

#     # if we want to delete an previously uploaded propgram
#     # old_token = ""
#     # service.delete_program(old_token)

#     # upload the code
#     program_id = service.upload_program(
#         data="vqls_runtime_program.py", metadata=get_metadata()
#     )
#     print("Program token:", program_id)

#     # query the details
#     prog = service.program(program_id)
#     print("Program details")
#     print(prog)
