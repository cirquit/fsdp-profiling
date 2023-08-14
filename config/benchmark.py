# Copyright (c) 2022 Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

@dataclass
class benchmark_config:
    # general
    host_port: str = "12368"

    # seed
    seed: int = 2022

    # model
    model_name: str = "t5-large"  # "google/t5-v1_1-small"
    tokenizer: str = "t5-large"
    # available models
    ## t5-base
    # google/t5-v1_1-small
    # google/t5-v1_1-base
    # google/t5-v1_1-large
    # google/t5-v1_1-xl  #3b
    # google/t5-v1_1-xxl #11b
    save_model: bool = False
    model_checkpoint: str = "t5_small_save.pt"
    print_sharding_plan: bool = False

    # dataloaders
    num_workers_dataloader: int = 4

    # policies
    use_mixed_precision: bool = True
    use_fp16: bool = True
    #sharding_strategy: str = "HYBRID_SHARD"   # sharing withing each node, but DDP between nodes
    sharding_strategy: str = "NO_SHARD"       # DDP Mode - each GPU keeps full copy of everything
    #sharding_strategy: str = "SHARD_GRAD_OP"  # Zero2 Mode - model parameters are not freed after forward pass
    #sharding_strategy: str = "FULL_SHARD"     # default - model, optim, grads are sharded

    fsdp_activation_checkpointing: bool = False

    # datasets
    dataset_train = "datasets_grammar/grammar_train.csv"
    dataset_test = "datasets_grammar/grammar_validation.csv"

    # training
    batch_size: int = 4
    num_epochs: int = 1
    max_step_count: int = 100
    # validation
    run_validation: bool = False
    val_batch_size: int = 4
    block_for_validation: bool = False

    # logging
    track_memory = True
    memory_report: bool = True
    nccl_debug_handler: bool = True
    distributed_debug: bool = True

    # Fine Tuning
    use_child_tuning: bool = False

    use_task_free: bool = False
    use_fisher_matrix: bool = False
    percent_F: float = 0.3
