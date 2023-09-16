mkdir -p .logcounter
EXP_NR=$(ls .logcounter/ | wc -w)
EXP_NAME="FSDP-$EXP_NR"
touch .logcounter/$EXP_NAME
echo $EXP_NAME

export DOWNLOADED_DATASETS_PATH="/dccstor/ais-model-store/hugginface"
export OMP_NUM_THREADS=2

# DLPROF_PATH="./dlprof/$EXP_NAME"

# --nsys_base_output_filename=$EXP_NAME"


#dlprof --mode=pytorch --nsys_opts="-t cuda,nvtx" --reports all --output_path="$DLPROF_PATH" \
#        --key_op="__getattr__" \
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=101 --rdzv_endpoint="localhost:5699" \
    main_benchmark.py  --group_name $EXP_NAME
