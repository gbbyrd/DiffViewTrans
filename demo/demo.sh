#!/bin/bash

if [ -f "depth_instance_diffusion_epoch=000299.ckpt" ]; then
    echo "Diff checkpoint downloaded"
else
    gdown https://drive.google.com/uc?id=1tZjf6CLib5lqFoZ0htTyrN8mKMkPnNIy
fi

if [ -f "3d_view_translation_config_depth_instance.yaml" ]; then
    echo "Diff config downloaded"
else
    gdown https://drive.google.com/uc?id=1dBACBuGS_s3-DPQmLedz6eLFwwNtnF1C
fi

if [ -f "vqgan_depth_instance_epoch_29.ckpt" ] ; then
    echo "Autoencoder checkpoint downloaded"
else
    gdown https://drive.google.com/uc?id=1vjonNsdNefh2drjzugZaLYrH735rO6jM
fi

python ../scripts/trans_diff_inference.py   --diff_ckpt depth_instance_diffusion_epoch=000299.ckpt \
                                            --autoencoder_ckpt vqgan_depth_instance_epoch_29.ckpt \
                                            --diff_config 3d_view_translation_config_depth_instance.yaml \
                                            --sample_data_folder demo_1d_dataset \
                                            --save_dir samples \
                                            --n_samples 4