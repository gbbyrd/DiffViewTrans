# Stable Diffusion Walaris

This repo is a modified version of the original stable diffusion repo found here: https://github.com/CompVis/stable-diffusion

It was created for the custom training and testing of stable diffusion models for synthetic data.

# Environment Setup
To set up the conda environment, type the following command after navigating to the base directory of the repository:

```conda env create -f environment.yaml```

Activate this environment with

```conda activate ldm```

# Training

In order to train an unconditional stable diffusion model, you must first train an autoencoder to take you to and from the latent space (where the diffusion takes place).

## Autoencoder
In order to train an autoencoder, you must first make changes in the /configs/autoencoder/<autoencoder_choice>.yaml file.

The main configurations to change are the target specifications for the train data. These configurations should be on lines 34 and 40 of the autoencoder_kl_64x64x3.yaml file (you can use different autoencoder.yaml files, this is just the one that I used.) For example, in that autoencoder_kl_64x64x3.yaml I commented out the original paths to the datasets and created my own custom pytorch datasets for the new data. 

*** IT IS IMPORTANT TO NOTE, YOUR DATASET MUST OUTPUT A PYTHON DICTIONARY IN THE FORM {'image': img} WHERE `img` IS A NUMPY ARRAY NORMALIZED BETWEEN -1 AND 1 ***


Once this is set up, run the following command to train the autoencoder:

```python main.py --base configs/autoencoder/<autoencoder.yaml file> -t --gpus 0,1```

Training information including the checkpoints will be saves in the log directory.

## Latent Diffusion Model

Once you have trained an autoencoder, you can now train your ldm. For this, you will want to modify the config file for the latent diffusion model that you are going to train as follows.

1. Specify the type of autoencoder.
2. Specify the file path for the checkpoint of that autoencoder.
3. Specify the path to the dataset that you will be training on.

Once you have configured your .yaml file, you can train your custom latent diffusion model with the following command:

```python main.py --base configs/latent-diffusion/<ldm_config_.yaml_file> -t --gpus 0,1```

Once again, the training information will appear in the log directory.

# Generating Novel Data

Once you have your pretrained encoder and latent diffusion model, you can generate novel images similar to your training data.

First, copy the config file that you used to train your ldm model into the `./logs/<model_name>/checkpoints` folder.

Next, you must add this repository to your python path.

```export PYTHONPATH = $PYTHONPATH:<path to repo>```

Then run the following command to generate novel samples:

```python scripts/thermal_ldm_inference.py -r <path to ldm checkpoint>```

By default, this will sample 100 images and save the files in the logs/<model_name>/samples directory. However, there are multiple arguments you can pass to modify the number of samples, batch size, and provide a custom directory in which to save the images. Please look at the arguments in the scripts/thermal_ldm_inference.py file to see what arguments you can use.

# Memorization Testing

Diffusion models are known to memorize and regurgitate training data if the training data contains duplicates. In order to run a test for Euclidean Distance (L2 Norm) between samples from your diffusion model and the training sanples, you can run the following command. The pre_process argument allows you to implement any preprocessing that was done in the training data by implementing a custom pre_process function in the check_for_memorization.py script.

```python scripts/check_for_memorization.py --training_image_folder <path to training image folder> --sample_image_folder <path to sample image folder> --pre_process <True/False>```

This script will place the results of this test in a folder within the samples folder titled 'mem_test_images'.