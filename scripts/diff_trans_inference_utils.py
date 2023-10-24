import os
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np

from ldm.datasets.custom_datasets import RGBDepthDatasetBase
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model

def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step

def preprocess_depth(depth_img):
        """ Normalize depth image and return h x w x 1 numpy array."""
        depth_img = depth_img[:,:,2] + 256 * depth_img[:,:,1] + 256 * 256 * depth_img[:,:,0]
        depth_img = depth_img / (256 * 256 * 256 - 1)
        
        # the distribution of depth values was HEAVILY skewed towards the lower end
        # therfore we will try to improve the distribution by clipping between
        # 0 and a threshold and normalizing based on these
        
        # need to test with clip_coefficient = 2
        clip_coefficient = 4

        depth_img = np.clip(depth_img, 0, 1/clip_coefficient)

        depth_img = depth_img * clip_coefficient

        depth_img = depth_img * 2 - 1

        return np.expand_dims(depth_img, axis=-1)

def run_translation_rgb_depth(model, opt):
    """Runs inference using depth - instance segmentation images and model.
    """

    # create dataloader
    dataset = RGBDepthDatasetBase(opt.sample_data_folder) # for depth instance sampling
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

    if opt.vanilla_sample:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        model.num_timesteps = opt.custom_steps
        print(f'Using DDIM sampling with {opt.custom_steps} sampling steps and eta={opt.eta}')

    # create DDIM sampler object
    model.num_timesteps = 1000
    sampler = DDIMSampler(model)
    
    # sample batches from the dataloader and run inference
    img_count = 0
    for batch in dataloader:
        z, c, x, xrec, xc, translation_label = model.get_input(batch, model.first_stage_key,
                                                           return_first_stage_outputs=True,
                                                           force_c_encode=True,
                                                           return_original_cond=True,
                                                           bs=opt.batch_size)
        

        samples, intermediates = sampler.sample(200, opt.batch_size, 
                                                shape=(3, 64, 64), 
                                                conditioning=c, verbose=False,
                                                translation_label=translation_label, 
                                                eta=1.0)

        # decode from latent with pretrained autoencoder
        imgs = model.decode_first_stage(samples)
        
        # save the translated imgs
        for i in range(opt.batch_size):
            # process translated img
            to_view = torch.clamp(imgs[i], -1., 1.)
            to_view = to_view.cpu().numpy()
            to_view = np.transpose(to_view, (1, 2, 0))
            to_view = denormalize(to_view)

            # get ground truth and conditioning imgs
            ground_truth = batch['to'][i].numpy()
            from_view = batch['from'][i].numpy()

            # denormalize data images loaded from dataloader
            ground_truth = denormalize(ground_truth)
            from_view = denormalize(from_view)

            # uncomment below to display images for 5 seconds

            # cv2.imshow('ground_truth', ground_truth)
            # cv2.imshow('aerial', aerial_view)
            # cv2.imshow('translated', trans_img)
            # cv2.waitKey(5000)

            # concatenate together
            full_img = np.concatenate([from_view, ground_truth, to_view], axis=1)

            # break each 4 channel image into depth and rgb images
            full_rgb_img = np.concatenate([from_view[:, :, :3], ground_truth[:, :, :3], to_view[:, :, :3]], axis=1)
            full_depth_img = np.concatenate([from_view[:, :, -1], ground_truth[:, :, -1], to_view[:, :, -1]], axis=1)

            # save the imgs
            # cv2.imwrite(os.path.join(opt.save_dir, f'processed_sample_{img_count}.png',),
            #             processed_data)
            cv2.imwrite(os.path.join(opt.save_dir, f'sample_{img_count}_rgb.png'),
                        full_rgb_img)
            cv2.imwrite(os.path.join(opt.save_dir, f'sample_{img_count}_depth.png'),
                        full_depth_img)
            img_count += 1
            
            # break out if number of samples reached
            if img_count >= opt.n_samples:
                break
        
        if img_count >= opt.n_samples:
                break
        
def denormalize(img): 
    """ Takes an img normalized between [-1, 1] and denormalizes to between 
    [0, 255]
    """
    img = (((img + 1.0) / 2.0) * 255).astype(np.uint8)

    return img