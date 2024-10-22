CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.run \
    --nproc_per_node=auto \
     --master_port 6080 \
    main.py \
        --method-name image_text_co_decomposition_b64 \
        --resume /mnt/Disk16T/lxl/zjp/CoDe_multi_rate/output/image_text_co_decomposition_b64_240915_104657/ckpt_70000_miou40.07.pth \
        --eval
