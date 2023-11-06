# Trans Diffusion - Clemson University Big Data Lab

This repo borrows heavily from the following repos:
1. https://github.com/CompVis/stable-diffusion
2. https://github.com/baofff/U-ViT

It was created for training and researching diffusion models for the purpose of view translation.

# Environment Setup

**NOTE: You cannot reuse this environment. The environment must be recreated using the below commands every time you clone this repository.**

To set up the conda environment, type the following command after navigating to the base directory of the repository:

```bash
conda env create -f environment.yaml
```

Activate this environment with

```bash
conda activate ldm
```

# LPIPs Loss Modification
After you have created the environment, you must add the below code snippet in line 64 of DiffViewTrans/src/taming-transformers/taming/modules/losses/lpips.py:

```python
def forward(self, inp):
    if inp.size()[1] == 4:                  # add this line
        inp = inp[:, :3, :, :]              # add this line
    return (inp - self.shift) / self.scale
```

This will allow the LPIPs loss to be used with 4 channel images (rgbd images in our case)

# Collecting Data

The data for this work is collected in the Carla simulator (Unreal Engine) and AutoDrive simulator (Unity). We are currently still working on adding functionality to the AutoDrive simulator, so it will not yet be supported in this README.

## Carla
Follow the instructions here to build carla from source on Linux: https://carla.readthedocs.io/en/latest/build_linux/

... or here to build from source on Windows: https://carla.readthedocs.io/en/latest/build_windows/

A sample data collection script that uses Carla synchronicity has been provided in the `carla_data_collection` folder. Carla provides a wide variety of sensors, and we encourage experimentation, but the current sensor configuration in that file should provide you with a label.json file and rgb, depth, and instance segmented images as shown below:

![alt text](https://github.com/gbbyrd/DiffViewTrans/blob/main/demo/demo_1d_dataset/rgb_img_0000000.png?raw=true)
![alt text](https://github.com/gbbyrd/DiffViewTrans/blob/main/demo/demo_1d_dataset/depth_img_0000000.png?raw=true)
![alt text](https://github.com/gbbyrd/DiffViewTrans/blob/main/demo/demo_1d_dataset/instance_segmentation_img_0000000.png?raw=true)

# Training

In order to train a trans diffusion model, you must first train an autoencoder to take you to and from the latent space (where the diffusion takes place).

## Autoencoder
In order to train an autoencoder, you must first make changes in the /configs/autoencoder/<autoencoder_choice>.yaml file.

The main configurations to change are the target specifications for the train data. These configurations should be on lines 34 and 40 of the autoencoder_kl_64x64x3.yaml file (you can use different autoencoder.yaml files, this is just the one that I used.) For example, in that autoencoder_kl_64x64x3.yaml I commented out the original paths to the datasets and created my own custom pytorch datasets for the new data. 

Once this is set up, run the following command to train the autoencoder:

```bash
python main.py --base configs/autoencoder/<autoencoder.yaml file> -t --gpus 0,
```

Training information including the checkpoints will be saves in the log directory.

## Trans Diffusion Model

Once you have trained an autoencoder, you can now train your ldm. For this, you will want to modify the config file for the latent diffusion model that you are going to train as follows.

1. Specify the type of autoencoder.
2. Specify the file path for the checkpoint of that autoencoder.
3. Specify the path to the dataset that you will be training on.

Once you have configured your .yaml file, you can train your custom latent diffusion model with the following command:

```bash
python main.py --base configs/latent-diffusion/<ldm_config_.yaml_file> -t --gpus 0,1
```

Once again, the training information will appear in the log directory.

# Inference

Once you have your pretrained encoder and latent diffusion model, you can run the view translation algorithm.

Check the argument parameters in `scripts/trans_diff_inference.py` and specify your diffusion and autoencoder checkpoints along with your diffusion configuration file. You must also specify your dataset folder path in the command line arguments when running the file. Other arguments can be used to fine tune your results.. Just check the argument list for information.

# Demo

A demo has been setup in the `demo` folder. To run this demo, simply navigate to the folder and run the following commands

```chmod +x demo.sh```
```./demo.sh```

This will download the pretrained checkpoints and config file and sample some translations from the data.

## Demo Samples

The demo should output results in the below format, with the original image on the left, the ground truth image in the middle, and the translated (fake) image on the right.

![alt text](https://github.com/gbbyrd/DiffViewTrans/blob/main/demo/samples_ref/sample_0.png?raw=true)
![alt text](https://github.com/gbbyrd/DiffViewTrans/blob/main/demo/samples_ref/sample_1.png?raw=true)
![alt text](https://github.com/gbbyrd/DiffViewTrans/blob/main/demo/samples_ref/sample_2.png?raw=true)
![alt text](https://github.com/gbbyrd/DiffViewTrans/blob/main/demo/samples_ref/sample_3.png?raw=true)