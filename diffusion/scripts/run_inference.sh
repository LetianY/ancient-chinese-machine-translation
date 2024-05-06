python -u scripts/run_inference.py \
--model_dir diffusion_models/diffuseq_24history_h768_lr0.001_t200_sqrt_lossaware_seed102_train-24history20240506-06:37:28 \
--seed 123 \
--bsz 32 \
--step 200 \
--split test