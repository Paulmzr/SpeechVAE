SAVE_PATH=runs/vae-dim-128-kl-5e-4
LOG_FILE=log.re
mkdir -p ${SAVE_PATH}

torchrun --nproc_per_node 4 --nnodes 1 --master_port 29502 scripts/train.py \
    --args.load conf/vae/base_vae.yml \
    --save_path ${SAVE_PATH} \
    2>&1 |tee -a ${SAVE_PATH}/${LOG_FILE}
