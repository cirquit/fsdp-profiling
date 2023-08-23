# Copyright (c) 2022 Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.


# main benchmark file for t5 training and prediction
# use this to run fast test and/or profile with

import os
import argparse
import datasets_grammar as dg

#import nvidia_dlprof_pytorch_nvtx
#nvidia_dlprof_pytorch_nvtx.init()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from torchvision import datasets, transforms


# for grammar correction
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# for generation
from transformers.models.t5.modeling_t5 import T5Block
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq

import functools
import pickle
from functools import partial
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data import DataLoader

from ChildTuningOptimizer import ChildTuningAdamW

# from sklearn.model_selection import train_test_split
import time
from datetime import datetime

import verify
import policies
from policies import mixed_precision
from utils.tb_logger import TBLogger, NoOPLogger
from utils.monitor import Monitor
from utils.timers import TBTimeIt as timeit
#from utils.timers import NoOPTimeIt as timeit

import datasets_grammar as dg
import tqdm

import config

# some globals
g_port = "12369"
g_addr = "localhost"

def _is_rank_0():
    return 0 == os.getenv("RANK")

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch fsdp T5.11 Example")
    parser.add_argument("--save-dir", default="/model_chkpt", type=str)
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 2022)"
    )
    parser.add_argument(
        "--group_name", type=str, help="Logging group variable"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    args = parser.parse_args()
    return args

def get_policies(cfg):
    """establish current policies for mixed precision and fsdp wrapping"""
    mixed_precision_policy = None
    wrapping_policy = None

    if cfg.use_mixed_precision:
        bf16_ready = verify.bf16_ready

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = policies.bfSixteen
            if _is_rank_0():
                print(f"Precision: BF16 (param, grad, buffer)")
        elif cfg.use_fp16:
            mixed_precision_policy = policies.fpSixteen
            if _is_rank_0():
                print(f"Precision: FP16 (param, grad, buffer)")
        else:
            mixed_precision_policy = policies.fp32_policy
            if _is_rank_0():
                print(f"Precision: FP32 (para, grad, buffer)")
    else:
        mixed_precision_policy = policies.fp32_policy
        if _is_rank_0():
           print(f"Precision: FP32 (default)")
    wrapping_policy = policies.get_t5_wrapper()

    return mixed_precision_policy, wrapping_policy

def setup(rank, world_size, cfg):
    # os.environ["MASTER_ADDR"] = g_addr
    # os.environ["MASTER_PORT"] = cfg.host_port
    # initialize the process group
    dist.init_process_group("nccl")  # , rank=rank, world_size=world_size)

def setup_environ_flags(cfg, rank):
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    if cfg.nccl_debug_handler:
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    if cfg.distributed_debug:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    #os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["PYTHONFAULTHANDLER"] = str(1)

def clear_gpu_cache(rank=None):
    torch.cuda.empty_cache()
    print(f"Rank {rank}: Cleared GPU cache")

def setup_tasks(rank, world_size, cfg):
    """keep the basic setup list here"""
    setup(rank, world_size, cfg)
    # clear_gpu_cache() - need to call torch set device first?
    # set_printing()
    setup_environ_flags(cfg, rank)

def train(
    cfg,
    model,
    local_rank,
    rank,
    world_size,
    train_loader,
    optimizer,
    epoch,
    epoch_start_time_s,
    train_start_time_s,
    monitor,
    run_name_dir,
    sampler=None,
    profiler=None,
    scaler=None,
    logger=None,
    lr_scheduler=None,
):
    model.train()
    ddp_loss = torch.zeros(2).to(local_rank)

    if sampler:
        sampler.set_epoch(epoch)
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(cfg.max_step_count), colour="blue", desc="Training Epoch"
        )

    # starting timer for dataload due to iterator
    step_time_start_s = time.perf_counter()
    step_counter = 1

    #with torch.autograd.profiler.emit_nvtx():
    for batch in train_loader:
        dataload_time_s = time.perf_counter() - step_time_start_s
        # calculate for tokens/s
        token_count = sum([len(e) for e in batch["source_ids"]])

        with timeit("dataload_cuda_mode", logger) as dataload_cuda_move_timer:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
        with timeit("zero_grad", logger) as zero_grad_timer:
            optimizer.zero_grad()
        with timeit("forward", logger) as forward_timer:
            output = model(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                labels=batch["target_ids"],
            )

        loss = output["loss"]
        if scaler:
            with timeit("backward", logger) as backward_timer:
                scaler.scale(loss).backward()
            with timeit("opt_step", logger) as opt_step_timer:
                scaler.step(optimizer)
            scaler.update()  # adjust scaling for next minibatch
        else:
            with timeit("backward", logger) as backward_timer:
                loss.backward()
            with timeit("opt_step", logger) as opt_step_timer:
                optimizer.step()

        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(batch)
        if rank == 0:
            inner_pbar.update(1)
        if profiler:
            profiler.step()

        logger.log_dict(monitor.get_sys_info())
        logger.log("01_general/loss", loss)
        logger.log("01_general/epoch", epoch)
        logger.log("01_general/step", step_counter)
        logger.log("02_timing/dataload_time_s", dataload_time_s)
        logger.log("02_timing/calculated_step_time_s",
            dataload_time_s + \
            dataload_cuda_move_timer.delta_time_s() + \
            zero_grad_timer.delta_time_s() + \
            forward_timer.delta_time_s() + \
            backward_timer.delta_time_s() + \
            opt_step_timer.delta_time_s())
        # restarting timer for dataload due to iterator
        actual_step_time_s = time.perf_counter() - step_time_start_s
        step_time_start_s = time.perf_counter()
        # delta time for epoch and training for easier parsing
        running_epoch_time_s = step_time_start_s - epoch_start_time_s
        running_train_time_s = step_time_start_s - train_start_time_s
        logger.log("02_timing/actual_step_time_s", actual_step_time_s)
        samples_per_second = cfg.batch_size / actual_step_time_s
        tokens_per_second = token_count / actual_step_time_s
        logger.log("01_general/sps", samples_per_second)
        logger.log("01_general/tokens_per_s", tokens_per_second)
        logger.log("02_timing/running_epoch_time_s", running_epoch_time_s)
        logger.log("02_timing/running_training_time_s", running_train_time_s, commit=True)

        if cfg.memory_snapshotting and \
            cfg.memory_snapshot_step==step_counter:
            snapshot = torch.cuda.memory._snapshot()
            path = Path(run_name_dir) / Path(f"mem_snapshot_rank-{rank}_step-{step_counter}.pickle")
            print(path)
            with open(path, "wb") as f:
                pickle.dump(snapshot, f)

        step_counter += 1

        # we're currently only interested in the first few steps for profiling!
        if step_counter >= cfg.max_step_count:
            if rank == 0:
                print(f"Early stopping with {cfg.max_step_count=}")
            break

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    train_accuracy = ddp_loss[0] / ddp_loss[1]
#    if rank == 0:
#        inner_pbar.close()
    return train_accuracy

def validation(cfg, model, local_rank, rank, world_size, test_loader, scaler):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(local_rank)
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(test_loader)), colour="green", desc="Validation Epoch"
        )
    with torch.no_grad():
        for batch in test_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            output = model(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                labels=batch["target_ids"],
            )
            ddp_loss[0] += output["loss"].item()  # sum up batch loss
            ddp_loss[1] += len(batch)

            if rank == 0:
                inner_pbar.update(1)
            # pred = output.logits.argmax(
            #    dim=1, keepdim=True
            # )  # get the index of the max log-probability
            # ddp_loss[1] += pred.eq(batch["target_ids"].view_as(pred)).sum().item()
            # ddp_loss[2] += len(batch)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    val_loss = ddp_loss[0] / ddp_loss[1]

    if rank == 0:
        # test_loss = ddp_loss[0] / ddp_loss[1]
        inner_pbar.close()
        print(f"Validation Loss: {val_loss:.4f}")
    return val_loss

def log_config(logger, cfg):
    for key, val in vars(cfg).items():
        logger.log_text(f"00_cfg/{key}", str(val))

def log_monitor_config(logger, monitor):
    for key, val in monitor.get_static_info().items():
        logger.log_text(f"00_cfg/{key}", str(val))
    # log the version of torch nightly as well
    logger.log_text(f"00_cfg/torch_version", str(torch.__version__))
    major, mid, minor = torch.cuda.nccl.version()
    logger.log_text(f"00_cfg/nccl_version", f"{major}.{mid}.{minor}")
    logger.log_text(f"00_cfg/cuda_version", torch.version.cuda)

def fsdp_main(args, logger, run_name):
    """Main entry point
    """
    cfg = config.benchmark_config()
    log_config(logger=logger, cfg=cfg)
    monitor = Monitor(cuda_enabled=True)
    log_monitor_config(logger=logger, monitor=monitor)

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    setup_tasks(rank, world_size, cfg)

    #print(f"Rank {rank}: NCCL ENV = {os.getenv('NCCL_DEBUG')}")

    batch_size = cfg.batch_size
    val_batch_size = cfg.val_batch_size

    scaler = None  # only used for fp16

    mp_policy, wrapping_policy = get_policies(cfg)

    if cfg.use_fp16:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
        scaler = ShardedGradScaler()

    model_name = cfg.model_name
    printable_model_name = str.replace(model_name, "/", "=")
    save_name = model_name + "-"

    # grammar correction
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer, model_max_length=512)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if cfg.memory_snapshotting:
        torch.cuda.memory._record_memory_history(
            True,
            trace_alloc_max_entries=100000, # keep 100,000 alloc/free events from before the snapshot              
            trace_alloc_record_context=True # record stack information for the trace events
        )

    # summarization
    # model = T5ForConditionalGeneration.from_pretrained(model_name)
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    # dataset_name = "jfleg_train.csv"

            # print(f"{dataset_name} contains: {dataset.keys()}")
        # print("Size of {dataset_name} train dataset: ", dataset["train"].shape)
        # print(
        #    "Size of {dataset_name} Validation dataset: ", dataset["validation"].shape
        # )

    train_name = None
    if cfg.dataset_train:
        train_name = cfg.dataset_train

    train_dataset = dg.get_dataset(tokenizer, train_name, 512, 512, True)
    if _is_rank_0():
        print(f"Train {train_name} = {len(train_dataset)} samples")

    val_dataset = dg.get_dataset(tokenizer, cfg.dataset_test, 512, 512, True)
    if _is_rank_0():
        print(f"Val {cfg.dataset_test} = {len(val_dataset)} samples")

    sampler1 = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )
    sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    train_kwargs = {"batch_size": batch_size, "sampler": sampler1}
    test_kwargs = {"batch_size": val_batch_size, "sampler": sampler2}
    cuda_kwargs = {
        "num_workers": cfg.num_workers_dataloader,
        "pin_memory": False,
        "shuffle": False,
    }
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    torch.cuda.set_device(local_rank)
    clear_gpu_cache(local_rank)

    if cfg.sharding_strategy == "NO_SHARD":
        sharding_strategy = ShardingStrategy.NO_SHARD
    elif cfg.sharding_strategy == "SHARD_GRAD_OP":
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    elif cfg.sharding_strategy == "HYBRID_SHARD":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        sharding_strategy = ShardingStrategy.FULL_SHARD

    cpu_offload_fsdp = CPUOffload(offload_params=cfg.cpu_offloading)

    model = FSDP(
        model,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
        sharding_strategy=sharding_strategy,
        use_orig_params=True,
        cpu_offload=cpu_offload_fsdp
    )

    if cfg.fsdp_activation_checkpointing:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={T5Block}
        )
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            auto_wrap_policy=auto_wrap_policy
        )

    if _is_rank_0():
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        milli_params = round(total_params / 1e6, 2)
        print(f"{model_name}: {milli_params}M")

    lr = 0.0008
    gamma = 0.85
    if cfg.use_task_free:
        optimizer = ChildTuningAdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01,
            reserve_p=cfg.percent_F,
            mode="taskfree",
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    lr_scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    epochs = cfg.num_epochs

    best_train_accuracy = float("-inf")
    best_val_loss = float("inf")
    curr_val_loss = float("inf")

    profiler = None
    if cfg.enable_profiler:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=1,
                repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                Path(".tb_logs") / Path(run_name)
                #run_name
#                "./test-logs/test"
            ),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            with_flops=True
        )

    train_start_time_s = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start_time_s = time.perf_counter()
        train_accuracy = train(
            cfg,
            model,
            local_rank,
            rank,
            world_size,
            train_loader,
            optimizer,
            epoch=epoch,
            epoch_start_time_s=epoch_start_time_s,
            train_start_time_s=train_start_time_s,
            monitor=monitor,
            sampler=sampler1,
            profiler=profiler,
            scaler=scaler,
            logger=logger,
            lr_scheduler=lr_scheduler,
            run_name_dir=Path(".tb_logs") / Path(run_name)
        )
        if cfg.block_for_validation:
            dist.barrier()
            #if rank == 0:
            #    print(f"--> blocking ranks for pre-validation synching...")

        if cfg.run_validation:
            curr_val_loss = validation(
                cfg, model, local_rank, rank, world_size, test_loader, scaler=scaler
            )


    if profiler:
        profiler.stop()
#        profiler.export_chrome_trace(f"./trace-r{rank}.json")

    # init_end_event.record()
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_args()

    rank = int(os.environ["RANK"])
    run_name = f"{args.group_name}-r{rank}"

    with TBLogger(run_name=run_name) as logger:
        # torch run start
        fsdp_main(args, logger, run_name)
