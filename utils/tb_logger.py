import torch
import traceback
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

class NoOPLogger():
    """ """

    def __init__(self, run_name, log_path="./.tb_logs"):
        return
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, tb):
        return
    def log(self, name, value, commit=False):
        return
    def log_dict(self, dict_values, commit=False):
        return
    def log_text(self, name, text, commit=False):
        return
    def flush(self):
        return

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

    def log_dict(self, dict_values, commit=False):
        """Call self.log in a loop. Commit last call if commit=True.
        """
        dict_len = len(dict_values)
        for i, (key, value) in enumerate(dict_values.items()):
            if i == dict_len:
                commit = True
            self.log(name=key, value=value, commit=commit)

    def log_text(self, name, text, commit=False):
        """ """
        self._logger.add_text(
            tag=name,
            text_string=text,
            global_step=self._log_counter)
        if commit:
            self._inc_log_counter()

    def flush(self):
        """ """
        self._logger.flush()
