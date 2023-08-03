from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

class TBLogger():
    """ """

    def __init__(self, log_group, log_path="./tb_logs"):
        """Usage:
        ```
        with TBLogger(log_group = "my_group") as logger:
            my_function(logger)
        ```
        """
        self._log_group = log_group
        self._full_log_path = Path(log_path) / Path(self._log_group)
        self._logger = SummaryWriter(self._full_log_path)
        self._log_counter = 0

    def __enter__(self):
       """ """
       return self

    def __exit__(self):
        """Closes automatically TB writer automatically on object
        out of scope
        """
        self.logger.close()

    def _inc_log_counter(self):
        """ """
        self._log_counter += 1

    def log_hparams(self, hparams=None, metrics=None):
        """ """
        self.logger.add_hparams(hparam_dict=hparams,
                                metric_dict=metrics,
                                run_name=self._log_group)

    def log(self, name, value, commit=False):
        """ Wrapper around scalar logger with automatic move between cpu/gpu
        """
        if type(value) == torch.Tensor:
            if value.device.type != "cpu":
                value = value.to("cpu")
        self.logger.add_scalar(
            tag=name,
            scalar_value=value,
            global_step=self._log_counter,
            new_style=True
        )
        if commit:
            self._inc_log_counter()

    def flush(self):
        """ """
        self.logger.flush()
