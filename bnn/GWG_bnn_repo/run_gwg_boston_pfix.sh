python main_bnn.py --store_path "run_boston" \
    --config "bnn_particle_boston.yml" \
    --permutation_path "permutation/boston_permutation.pt" \
    --exp_time 0 \
    --master_stepsize 0.001 \
    --f_iter 5 \
    --f_lr 0.001 \
    --p_norm_lr 0.0001 \
    --p_norm 2.0 \
    --f_latent_dim 300 \
    --H_alpha 1.0 \
    --beta 0.9\
    --n_epoch 2000