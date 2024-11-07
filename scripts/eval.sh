CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
    --nproc_per_node=auto \
     --master_port 6080 \
    main.py \
        --method-name image_text_co_decomposition_b64 \
        --resume /output/checkpoints.pth \
        --eval
