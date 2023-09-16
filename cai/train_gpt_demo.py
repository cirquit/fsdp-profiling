import os
from contextlib import nullcontext
from functools import partial
from time import time as mtime

import colossalai
import psutil
import torch
import torch.nn as nn
from colossalai.booster import Booster
from colossalai.booster.plugin import (GeminiPlugin, LowLevelZeroPlugin,
                                       TorchDDPPlugin)
from colossalai.lazy import LazyInitContext
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device
from packaging import version
from torch.nn.parallel import DistributedDataParallel as DDP

from commons.model_zoo import model_builder
from commons.utils import (get_data, get_profile_context, get_tflops,
                           get_time_stamp)

CAI_VERSION = colossalai.__version__

import time

from utils.monitor import Monitor
from utils.tb_logger import TBLogger
from utils.timers import TBTimeIt as timeit


def parse_args():
    parser = colossalai.get_default_parser()
    parser.add_argument(
        "--distplan",
        type=str,
        default='CAI_Gemini',
        help="The distributed plan [colossalai, zero1, zero2, torch_ddp, torch_zero].",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size per DP group of training.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt2_medium",
        help="model model scale",
    )
    parser.add_argument(
        "--train_step",
        type=int,
        default=40,
        help="training iterations for test",
    )
    parser.add_argument(
        "--gpu_num",
        type=int,
        default=2,
        help="number of gpus",
    )


    args = parser.parse_args()
    return args


class GPTLMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'


def get_model_size(model: nn.Module):
    total_numel = 0
    for module in model.modules():
        for p in module.parameters(recurse=False):
            total_numel += p.numel()
    return total_numel


def model_size_formatter(numel: int) -> str:
    GB_SIZE = 10**9
    MB_SIZE = 10**6
    KB_SIZE = 10**3
    if numel >= GB_SIZE:
        return f'{numel / GB_SIZE:.1f}B'
    elif numel >= MB_SIZE:
        return f'{numel / MB_SIZE:.1f}M'
    elif numel >= KB_SIZE:
        return f'{numel / KB_SIZE:.1f}K'
    else:
        return str(numel)


def set_cpu_maximum_parallelism():
    conf_str = torch.__config__.parallel_info()
    inter_str = conf_str.split("hardware_concurrency() : ")[1]
    max_concurrency = inter_str.split('\n')[0]
    os.environ["OMP_NUM_THREADS"] = max_concurrency
    print(f"environmental variable OMP_NUM_THREADS is set to {max_concurrency}.")


def main():
    # version check
    # this example is supposed to work for versions greater than 0.2.0
    assert version.parse(CAI_VERSION) >= version.parse("0.2.0")

    set_cpu_maximum_parallelism()
    args = parse_args()

    # if args.distplan not in ["colossalai", "torch_ddp", "torch_zero", "zero1", "zero2"]:
    if args.distplan not in ["CAI_ZeRO1", "CAI_ZeRO2", "CAI_Gemini", "Pytorch_DDP", "Pytorch_ZeRO"]:
        raise TypeError(f"{args.distplan} is error")

    # batch size per DP degree
    BATCH_SIZE = args.batch_size
    SEQ_LEN = 1024
    VOCAB_SIZE = 50257

    NUM_STEPS = args.train_step

    WARMUP_STEPS = 1
    assert WARMUP_STEPS < NUM_STEPS, "warmup steps should smaller than the total steps"
    assert (NUM_STEPS - WARMUP_STEPS) % 2 == 1, "the number of valid steps should be odd to take the median"
    PROF_FLAG = False    # The flag of profiling, False by default

    disable_existing_loggers()
    colossalai.launch_from_torch(config={})

    logger = get_dist_logger()
    logger.info(f"{args.model_type}, {args.distplan}, batch size {BATCH_SIZE}", ranks=[0])

    # build criterion
    criterion = GPTLMLoss()
    torch.manual_seed(123)
    if args.distplan.startswith("CAI"):
        ctx = LazyInitContext(default_device=get_current_device()) if args.distplan == "CAI_Gemini" else nullcontext()
        # build GPT model
        with ctx:
            model = model_builder(args.model_type)(checkpoint=True)

        # assign running configurations
        if args.distplan == "CAI_ZeRO1":
            zero_stage = 1
        elif args.distplan == "CAI_ZeRO2":
            zero_stage = 2
        elif args.distplan == "CAI_Gemini":
            zero_stage = 3
        else:
            raise RuntimeError

        plugin = None
        if args.distplan.startswith("CAI_ZeRO"):
            plugin = LowLevelZeroPlugin(stage=zero_stage,
                                        reduce_bucket_size_in_m=12,
                                        overlap_communication=True,
                                        verbose=True)
        elif args.distplan == "CAI_Gemini":
            plugin = GeminiPlugin(search_range_m=128, hidden_dim=model.config.n_embd)
        else:
            raise RuntimeError

        # build a highly optimized gpu/cpu optimizer
        optimizer = HybridAdam(model.parameters(), lr=1e-3)

        logger.info(get_mem_info(prefix='After init optim, '), ranks=[0])
    elif args.distplan.startswith("Pytorch"):
        assert args.tp_degree == 1, "The degree of TP should be 1 for DDP examples."
        model = model_builder(args.model_type)(checkpoint=True).cuda()
        plugin = TorchDDPPlugin()
        if args.distplan.endswith("DDP"):
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        elif args.distplan.endswith("ZeRO"):
            from torch.distributed.optim import ZeroRedundancyOptimizer
            optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.Adam, lr=1e-3)

    else:
        raise RuntimeError
    # wrap your model and optimizer
    booster = Booster(plugin=plugin)
    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

    # model is shared after TP
    numel = get_model_size(model)
    logger.info(f"the size of testing model size is {model_size_formatter(numel)}.")
    logger.info(get_mem_info(prefix='After init model, '), ranks=[0])

    # Tflops_per_GPU = global_batch * global_numel * seq_len * 8 / #gpu
    # = (batch_per_DP_group * dp_degree) * (numel * tp_degree) * seq_len * 8 / (tp_degree * dp_degree)
    # = batch_per_DP_group * numel * seq_len * 8
    get_tflops_func = partial(get_tflops, numel, BATCH_SIZE, SEQ_LEN)

    torch.cuda.synchronize()
    model.train()
    tflops_list = []

    def train_step(tblogger, step_counter, monitor, batch_size):
        step_time_start_s = time.perf_counter()
        
        # we just use randomly generated data here
        input_ids, attn_mask = get_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        dataload_time_s = time.perf_counter() - step_time_start_s
        optimizer.zero_grad()

        start = mtime()
        with timeit("forward", tblogger) as forward_timer:
            outputs = model(input_ids, attn_mask)
        loss = criterion(outputs, input_ids)
        torch.cuda.synchronize()
        fwd_end = mtime()
        fwd_time = fwd_end - start
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Forward '), ranks=[0])

        with timeit("backward", tblogger) as backward_timer:
            booster.backward(loss, optimizer)

        torch.cuda.synchronize()
        bwd_end = mtime()
        bwd_time = bwd_end - fwd_end
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Backward '), ranks=[0])
        with timeit("opt_step", tblogger) as opt_step_timer:
            optimizer.step()
        torch.cuda.synchronize()
        optim_time = mtime() - bwd_end
        step_time = mtime() - start

        tblogger.log_dict(monitor.get_sys_info())
        tblogger.log("02_timing/dataload_time_s", dataload_time_s)
        tblogger.log("01_general/step", step_counter)
        tblogger.log("02_timing/dataload_time_s", dataload_time_s)
        # restarting timer for dataload due to iterator
        actual_step_time_s = time.perf_counter() - step_time_start_s
        step_time_start_s = time.perf_counter()
        samples_per_second = batch_size / actual_step_time_s
        tokens_per_second = (batch_size * 1024) / actual_step_time_s
        tblogger.log("01_general/sps", samples_per_second)
        tblogger.log("01_general/tokens_per_s", tokens_per_second, commit=True)
 
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Optimizer step '), ranks=[0])

        step_tflops = get_tflops_func(step_time)
        logger.info(
            f"[{n + 1}/{NUM_STEPS}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s, TFLOPS: {get_tflops_func(step_time):.3f}, FWD time: {fwd_time:.3f}s, BWD time: {bwd_time:.3f}s, OPTIM time: {optim_time:.3f}s",
            ranks=[0],
        )
        if n >= WARMUP_STEPS:
            tflops_list.append(step_tflops)

    demo_profiler = get_profile_context(PROF_FLAG,
                                        WARMUP_STEPS,
                                        NUM_STEPS - WARMUP_STEPS,
                                        save_dir=f"profile/{get_time_stamp()}-demo")

    rank = os.getenv("RANK")
    run_name = f"{args.distplan}-{args.batch_size}-{NUM_STEPS}-r{rank}"

    monitor = Monitor(cuda_enabled=True)
    with demo_profiler as prof:
        with TBLogger(run_name=run_name) as tblogger:
            for key, val in vars(args).items():
                tblogger.log_text(f"00_cfg/{key}", str(val))

            tblogger.log_text(f"00_cfg/batch_size", args.batch_size)
            tblogger.log_text(f"00_cfg/max_steps", NUM_STEPS)
            for n in range(NUM_STEPS):
                train_step(tblogger=tblogger, step_counter=n, monitor=monitor, batch_size=args.batch_size)
                prof.step()

    tflops_list.sort()
    median_index = ((NUM_STEPS - WARMUP_STEPS) >> 1) + WARMUP_STEPS
    logger.info(f"Median TFLOPS is {tflops_list[median_index]:.3f}")
    torch.cuda.synchronize()

if __name__ == '__main__':
    main()
