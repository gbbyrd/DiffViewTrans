import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange

from omegaconf import OmegaConf
from PIL import Image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

from torch.utils.data import Dataset, DataLoader
import json
import cv2
from ldm.models.diffusion.ddim import DDIMSampler
import torchvision

rescale = lambda x: (x + 1.) / 2.

class SampleDataset(Dataset):
    def __init__(self, sample_data_folder_path):
        # self.base_data_folder = '/home/nianyli/Desktop/code/thesis/DiffViewTrans/data/3D_trans_diff_v1_256'
        # label_json_file_path = '/home/nianyli/Desktop/code/thesis/DiffViewTrans/data/3D_trans_diff_v1_256/labels.json'
        self.base_data_folder = sample_data_folder_path
        label_json_file_path = sample_data_folder_path+'/labels.json'
        self.img_paths = []

        with open(label_json_file_path, 'r') as file:
            data = json.load(file)

        self.labels = data['data']
        self.sensor_limits = data['sensor_limits']

        self.img_info = []

        normalize_dict = {}
        for key in self.sensor_limits:
            if key == 'num_sensors':
                continue
            if self.sensor_limits[key][0] == self.sensor_limits[key][1]:
                normalize_dict[key] = False
            else:
                normalize_dict[key] = True

        # normalize all sensor location data
        for idx, label in enumerate(self.labels):
            for label_idx, img_name in enumerate(label):
                for key in label[img_name]['location']:
                    if normalize_dict[f'{key}_limits']:
                        min_lim = self.sensor_limits[f'{key}_limits'][0]
                        max_lim = self.sensor_limits[f'{key}_limits'][1]

                        # normalize between 0 and 1
                        label[img_name]['location'][key] = (label[img_name]['location'][key] - min_lim) / (max_lim-min_lim)

                        # normalize between -1 and 1
                        label[img_name]['location'][key] = label[img_name]['location'][key] * 2 - 1
                    else:
                        label[img_name]['location'][key] = 0

            self.labels[idx] = label

        # get all of the keys
        keys = list(self.labels[0].keys())
        keys.sort()

        for label in self.labels:
            for key in label:
                if key == keys[-1]:
                    continue
                info = {
                    'ground': os.path.join(self.base_data_folder, label[keys[-1]]['img_name']),
                    'aerial': os.path.join(self.base_data_folder, label[key]['img_name']),
                    'location': label[key]['location']
                }
                self.img_info.append(info)

    def __getitem__(self, idx):
        
        # generate random noise for ground sample
        ground_img = cv2.imread(self.img_info[idx]['ground'])
        aerial_img = cv2.imread(self.img_info[idx]['aerial'])
        # ground_img = np.random.randn(aerial_img.shape[0], aerial_img.shape[1])

        # normalize between [-1, 1]
        aerial_img = aerial_img / 127.5 - 1
        ground_img = ground_img / 127.5 - 1

        # get the location
        x = self.img_info[idx]['location']['x']
        y = self.img_info[idx]['location']['y']
        z = self.img_info[idx]['location']['z']
        z_angle = self.img_info[idx]['location']['z_angle']

        output_dict = {
            'ground': ground_img,
            'aerial': aerial_img,
            'location': np.array([[x, y, z, z_angle]], dtype='float32')
        }

        return output_dict
    
    def __len__(self):
        return len(self.img_info)

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0,):


    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                         make_prog_row=True)
        else:
            sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                    eta=eta)

        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log

def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000, nplog=None):
    if vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')


    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir,'*.png')))
    # path = logdir
    if model.cond_stage_model is None:
        all_images = []

        print(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
            logs = make_convolutional_sample(model, batch_size=batch_size,
                                             vanilla=vanilla, custom_steps=custom_steps,
                                             eta=eta)
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            all_images.extend([custom_to_np(logs["sample"])])
            if n_saved >= n_samples:
                print(f'Finish after generating {n_saved} samples')
                break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        all_img = all_img.squeeze()
        np.savez(nppath, all_img)

    else:
       raise NotImplementedError('Currently only sampling for unconditional models supported.')

    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")

def run_translation(model, opt):

    nplog = os.path.join(opt.logdir, "numpy")

    # create dataloader
    dataset = SampleDataset(opt.sample_data_folder)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    if opt.vanilla_sample:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        model.num_timesteps = opt.custom_steps
        print(f'Using DDIM sampling with {opt.custom_steps} sampling steps and eta={opt.eta}')

    # create DDIM sampler object
    model.num_timesteps = 1000
    sampler = DDIMSampler(model)
    
    img_count = 0
    for batch in dataloader:
        z, c, x, xrec, xc, translation_label = model.get_input(batch, model.first_stage_key,
                                                           return_first_stage_outputs=True,
                                                           force_c_encode=True,
                                                           return_original_cond=True,
                                                           bs=opt.batch_size)
        

        samples, intermediates = sampler.sample(200, opt.batch_size, (3, 64, 64), conditioning=c, verbose=False,
                                                translation_label=translation_label, eta=1.0)
        imgs = model.decode_first_stage(samples)
        
        # save the translated imgs
        for i in range(opt.batch_size):
            # process translated img
            trans_img = torch.clamp(imgs[i], -1., 1.)
            trans_img = trans_img.cpu().numpy()
            trans_img = np.transpose(trans_img, (1, 2, 0))
            trans_img = denormalize(trans_img)

            # get ground truth and conditioning imgs
            ground_truth = batch['ground'][i].numpy()
            aerial_view = batch['aerial'][i].numpy()

            ground_truth = denormalize(ground_truth)
            aerial_view = denormalize(aerial_view)

            # cv2.imshow('ground_truth', ground_truth)
            # cv2.imshow('aerial', aerial_view)
            # cv2.imshow('translated', trans_img)

            # cv2.waitKey(0)

            # concatenate together
            full_img = np.concatenate([aerial_view, ground_truth, trans_img], axis=1)

            # save the img
            cv2.imwrite(os.path.join(opt.save_dir, f'sample_{img_count}.png'),
                        full_img)
            img_count += 1
            
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

def save_sample_results(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
    root = os.path.join(save_dir, "images", split)
    for k in images:
        grid = torchvision.utils.make_grid(images[k], nrow=4)
        if self.rescale:
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        grid = grid.numpy()
        grid = (grid * 255).astype(np.uint8)
        filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
            k,
            global_step,
            current_epoch,
            batch_idx)
        path = os.path.join(root, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        Image.fromarray(grid).save(path)

def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path,'img', f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-r",
    #     "--resume",
    #     type=str,
    #     nargs="?",
    #     help="load from logdir or checkpoint in logdir",
    # )
    parser.add_argument(
        "--diff_ckpt",
        type=str,
        nargs="?",
        help="diffusion model checkpoint path",
    )
    parser.add_argument(
        "--autoencoder_ckpt",
        type=str,
        nargs="?",
        help="autoencoder checkpoint path",
    )
    parser.add_argument(
        "--diff_config",
        type=str,
        nargs="?",
        help="diffusion model config path",
    )
    parser.add_argument(
        "--autoencoder_config",
        type=str,
        nargs="?",
        help="autoencoder config path",
    )
    parser.add_argument(
        "--sample_data_folder",
        type=str,
        nargs="?",
        help="specify the folder containing the data to use to sample translation",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        nargs="?",
        help="specify the folder to save the sampled images to",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=100
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    return parser


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


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = opt.diff_ckpt

    # if not os.path.exists(opt.resume):
    #     raise ValueError("Cannot find {}".format(opt.resume))
    # if os.path.isfile(opt.resume):
    #     # paths = opt.resume.split("/")
    #     try:
    #         logdir = '/'.join(opt.resume.split('/')[:-1])
    #         # idx = len(paths)-paths[::-1].index("logs")+1
    #         print(f'Logdir is {logdir}')
    #     except ValueError:
    #         paths = opt.resume.split("/")
    #         idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
    #         logdir = "/".join(paths[:idx])
    #     ckpt = opt.resume
    # else:
    #     assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
    #     logdir = opt.resume.rstrip("/")
    #     ckpt = os.path.join(logdir, "model.ckpt")

    # base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    # base_configs = sorted(glob.glob(logdir+'/*.yaml'))
    opt.base = [opt.diff_config] #, opt.autoencoder_config]

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    # if opt.logdir != "none":
    #     locallog = logdir.split(os.sep)[-1]
    #     if locallog == "": locallog = logdir.split(os.sep)[-2]
    #     print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
    #     logdir = os.path.join(opt.logdir, locallog)
    if opt.logdir != "none":
        logdir = opt.logdir

    print(config)
    # ckpt = '/home/grayson/Desktop/Code/stable-diffusion/logs/FLIR_ldm/checkpoints/epoch=000296.ckpt'
    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    print(f"global step: {global_step}")
    print(75 * "=")
    print("logging to:")
    logdir = os.path.join(logdir, "samples", now)
    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")

    os.makedirs(imglogdir)
    os.makedirs(numpylogdir)
    print(logdir)
    print(75 * "=")

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)

    run_translation(model, opt)


    # run(model, logdir, batch_size=opt.batch_size, vanilla=True, 
    #     custom_steps=opt.custom_steps, eta=None, 
    #     n_samples=opt.n_samples, nplog=numpylogdir)

    # print("done.")