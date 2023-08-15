import time
import torch

def cuda_sync():
    """Waits until cuda is fully finished (for reproducibility purposes)"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

class NoOPTimeIt(object):
    def __init__(
        self,
        name: str,
        logger,
        cuda: bool = False,
        commit: bool = False,
        verbose: bool = False
    ):
        return

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return

    def delta_time_s(self):
        return 0.0

class TBTimeIt(object):
    def __init__(
        self, name: str, logger, cuda: bool = False, commit: bool = False, verbose: bool = False
    ):
        """Times the inner code with perf_counter and process_time and logs to
        result to the TB Logger
        perf_counter counts actual time, process_time counts the process time,
        i.e. no sleep included
        https://docs.python.org/3/library/time.html#time.process_time
        """
        self.delta_time_perf_s = 0
        self.name_perf = "02_timing/" + name + "_time_s"
        self.commit = commit
        self.cuda = cuda
        self.verbose = verbose
        self.logger = logger

    def __enter__(self):
        if self.verbose:
            logger.info(f"Entered {self.name_perf}")
        if self.cuda:
            cuda_sync()
        self.start_perf = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.verbose:
            logger.info(f"Exited {self.name_perf}")
        if self.cuda:
            cuda_sync()
        end_perf = time.perf_counter()
        self.delta_time_perf_s = end_perf - self.start_perf
        self.logger.log(
            name=self.name_perf,
            value=self.delta_time_perf_s,
            commit=self.commit,
        )

    def delta_time_s(self):
        """The default is perf_counter time"""
        return self.delta_time_perf_s
