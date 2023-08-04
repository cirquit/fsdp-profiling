import torch
import traceback
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

class TBLogger():
    """ """

    def __init__(self, run_name, log_path="./.tb_logs"):
        """Usage:
        ```
        with TBLogger(run_name = "my_group") as logger:
            my_function(logger)
        ```
        """
        self._run_name = run_name
        self._full_log_path = Path(log_path) / Path(self._run_name)
        self._logger = SummaryWriter(self._full_log_path)
        self._log_counter = 0

    def __enter__(self):
       """ """
       return self

    def __exit__(self, exc_type, exc_value, tb):
        """Closes automatically TB writer automatically on object
        out of scope
        """
        self._logger.close()
        # pass execption through
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            return False

    def _inc_log_counter(self):
        """ """
        self._log_counter += 1

    def log(self, name, value, commit=False):
        """ Wrapper around scalar logger with automatic move between cpu/gpu
        """
        if type(value) == torch.Tensor:
            if value.device.type != "cpu":
                value = value.to("cpu")
                value = value.to(torch.float32)
        self._logger.add_scalar(
            tag=name,
            scalar_value=value,
            global_step=self._log_counter,
            new_style=True
        )
        if commit:
            self._inc_log_counter()

    def flush(self):
        """ """
        self._logger.flush()
