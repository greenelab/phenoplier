"""
Implementation of a custom Kernel Manager for Papermill to avoid issues
when running notebooks in a cluster. Taken and adapted from:
https://github.com/nteract/papermill/issues/511
"""
import sys
import os
import logging
import uuid

import papermill as pm
from jupyter_client.manager import KernelManager
from jupyter_core.paths import jupyter_runtime_dir


class IPCKernelManager(KernelManager):
    def __init__(self, *args, **kwargs):
        kernel_id = str(uuid.uuid4())

        # make sure the Jupyter runtime dir exists
        os.makedirs(jupyter_runtime_dir(), exist_ok=True)

        connection_file = os.path.join(
            jupyter_runtime_dir(), f"kernel-{kernel_id}.json"
        )
        super().__init__(
            *args,
            transport="ipc",
            kernel_id=kernel_id,
            connection_file=connection_file,
            **kwargs,
        )


def get_parameters(parameters):
    if parameters is None or len(parameters.strip()) == 0:
        return None

    params = [s for s in parameters.split("-p ") if len(s) > 0]

    result = {}
    for param_pair in params:
        param_pair = param_pair.strip()
        param_pair = " ".join(param_pair.split())
        param_name, param_value = param_pair.split(" ")

        try:
            param_value = int(param_value)
        except ValueError:
            pass

        result[param_name] = param_value

    return result


def run_papermill(input_notebook, output_notebook, parameters):
    logging.basicConfig(level="INFO", format="%(message)s")

    notebook_parameters = {
        "PHENOPLIER_NOTEBOOK_FILEPATH": input_notebook,
    }

    parsed_parameters = get_parameters(parameters)
    if parsed_parameters is not None:
        notebook_parameters.update(parsed_parameters)

    pm.execute_notebook(
        input_notebook,
        output_notebook,
        progress_bar=False,
        log_output=True,
        stdout_file=sys.stdout,
        stderr_file=sys.stderr,
        request_save_on_cell_execute=True,
        parameters=notebook_parameters,
        kernel_manager_class="papermill_custom.IPCKernelManager",
    )
