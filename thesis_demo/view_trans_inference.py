import torch
import numpy as np

from ldm.models.diffusion.ddim import DDIMSampler

def denormalize(img): 
    """ Takes an img normalized between [-1, 1] and denormalizes to between 
    [0, 255]
    """
    img = (((img + 1.0) / 2.0) * 255).astype(np.uint8)

    return img

def run_translation_rgb_depth(model, 
                              opt,
                              from_img,
                              translation_label):
    """Runs inference using depth - instance segmentation images and model.
    """

    if opt.vanilla_sample:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        model.num_timesteps = opt.custom_steps
        print(f'Using DDIM sampling with {opt.custom_steps} sampling steps and eta={opt.eta}')

    # create DDIM sampler object
    model.num_timesteps = 1000
    sampler = DDIMSampler(model)
    
    # sample batches from the dataloader and run inference
    encoded_c = model.get_learned_conditioning(from_img.to(model.device))

    sample, intermediates = sampler.sample(200, opt.batch_size, 
                                           shape=(3, 64, 64), 
                                           conditioning=encoded_c, verbose=False,
                                           translation_label=translation_label, 
                                           eta=1.0)
    
    # decode from latent with pretrained autoencoder
    translated_rgb_depth_img = model.decode_first_stage(sample)

    # process translated img
    translated_rgb_depth_img = torch.clamp(translated_rgb_depth_img, -1., 1.)
    translated_rgb_depth_img = translated_rgb_depth_img.cpu().numpy()
    translated_rgb_depth_img = np.transpose(translated_rgb_depth_img, (1, 2, 0))
    translated_rgb_depth_img = denormalize(translated_rgb_depth_img)

    output = {
         'rgb': translated_rgb_depth_img[:, :, :3],
         'depth': translated_rgb_depth_img[:, :, -1]
    }

    return output
    
    