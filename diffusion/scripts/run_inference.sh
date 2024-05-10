python -u scripts/run_inference.py \
--model_dir diffusion_models/diffuseq_24history_h768_lr0.001_t2000_sqrt_lossaware_seed102_train-24history20240510-02\:07\:02 \
--seed 123 \
--bsz 32 \
--step 2000 \
--split test