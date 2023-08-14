EXP_NR=$(ls .logcounter/ | wc -w)
EXP_NAME="FSDP-$EXP_NR"
touch .logcounter/$EXP_NAME
echo $EXP_NAME

export DOWNLOADED_DATASETS_PATH="/dccstor/ais-model-store/hugginface"
export OMP_NUM_THREADS=2

torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=101 --rdzv_endpoint="localhost:5699" \
    main_benchmark.py  --group_name $EXP_NAME
