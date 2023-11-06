import carla

import os
import sys
import random
import time
import numpy as np
import cv2
import shutil
import pygame
import queue
import json
import glob
from tqdm import tqdm
import argparse
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
import matplotlib.pyplot as plt

from math import sqrt
from diff_trans_inference_utils import *

# global variables for quick data collection finetuning
SAVE_DIR = '/home/nianyli/Desktop/code/thesis/DiffViewTrans/data/default_vid_dave_path'
NUM_FRAMES = 100
IM_HEIGHT, IM_WIDTH = 256, 256

# define the x and z locations that the 'to' spawned cameras will spawn at
FRONT_X = 2.5
FRONT_Z = 2.5

class CarlaSyncMode(object):
    """Class for running Carla in synchronous mode. Allows frame rate sync to
    ensure no data is lost and all backend code runs and is completed before
    'ticking' to the next frame."""
    
    def __init__(self, world, **kwargs):
        self.world = world
        self.sensors = []
        self.actors = []
        self.vehicle = kwargs.get('vehicle', None)
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None
        self.sensor_params = kwargs.get('sensor_params', None)
        self.blueprint_library = world.get_blueprint_library()
        self.sensor_info = []
        
        self.translation_label_type = kwargs.get('translation_label_type', None)
        
    def __enter__(self):
        # get some basic carla metadata required for functionality
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode = False,
            synchronous_mode = True, 
            fixed_delta_seconds = self.delta_seconds
        ))
        
        # create queues for each sensor that store the data as the sensor reads it
        self.make_queue(self.world.on_tick)

        self.actors.append(self.vehicle)

        self.initialize_front_sensors()

        self.randomize_aerial_sensor_location()
        
        return self
    
    def initialize_front_sensors(self):
        """ Spawn the front rgb and depth sensors for the vehicle."""
        
        sensor_types = self.sensor_params['sensor_types']
        blueprint_attributes = self.sensor_params['blueprint_attributes']
        
        # spawn the front sensor
        for sensor_type in sensor_types:
            # create each sensor
            sensor_bp = self.blueprint_library.find(sensor_type)
            for attribute in blueprint_attributes:
                sensor_bp.set_attribute(attribute, blueprint_attributes[attribute])

            # spawn the sensor at 0,0,0
            spawn_point = carla.Transform(carla.Location(x=FRONT_X, y=0, z=FRONT_Z),
                                            carla.Rotation(roll=0, pitch=0, yaw=0))        
            sensor = self.world.spawn_actor(sensor_bp, spawn_point, attach_to=self.vehicle)
            self.sensors.append(sensor)

        # add each sensor to the queue
        for sensor in self.sensors:
            self.make_queue(sensor.listen)

    
    def make_queue(self, register_event):
        # create a q for the event to register data to
        q = queue.Queue()
        # define q.put as the function that is called when the event recieves data
        register_event(q.put)
        # add q to the list of _queues
        self._queues.append(q)
    
    def tick(self, timeout):
        """Call this function to step one frame through the simulation"""
        
        # get the next frame from the world.. this should automatically 
        # update the data in the sensor queues as well
        self.frame = self.world.tick()
        
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data
    
    def randomize_aerial_sensor_location(self):
        """Randomize the sensors within the limits specified in 
        self.sensor_params
        """
        
        sensor_types = self.sensor_params['sensor_types']
        blueprint_attributes = self.sensor_params['blueprint_attributes']
        initial_spawn_limits = self.sensor_params['initial_spawn_limits']
        
        # clear sensor information
        self.sensor_info.clear()

        # clear out the rest all sensors (make sure to keep the two front sensors)
        for sensor in self.sensors[2:]:
            sensor.destroy()
        self.sensors = self.sensors[:2]
        
        # get random spawn point for initial sensor
        x_lim_init = initial_spawn_limits['x']
        y_lim_init = initial_spawn_limits['y']
        z_lim_init = initial_spawn_limits['z']
        roll_lim_init = initial_spawn_limits['roll']
        pitch_lim_init = initial_spawn_limits['pitch']
        yaw_lim_init = initial_spawn_limits['yaw']
        
        x_init = random.uniform(x_lim_init[0], x_lim_init[1])
        y_init = random.uniform(y_lim_init[0], y_lim_init[1])
        z_init = random.uniform(z_lim_init[0], z_lim_init[1])
        roll_init = random.uniform(roll_lim_init[0], roll_lim_init[1])
        pitch_init = random.uniform(pitch_lim_init[0], pitch_lim_init[1])
        yaw_init = random.uniform(yaw_lim_init[0], yaw_lim_init[1])

        for sensor_type in sensor_types:
            # create sensor
            sensor_bp = self.blueprint_library.find(sensor_type)
            for attribute in blueprint_attributes:
                sensor_bp.set_attribute(attribute, blueprint_attributes[attribute])

            spawn_point = carla.Transform(carla.Location(x=x_init, y=y_init, z=z_init),
                                            carla.Rotation(roll=roll_init, yaw=yaw_init, pitch=pitch_init))
            
            # spawn the sensor relative to the first sensor
            sensor = self.world.spawn_actor(sensor_bp, spawn_point, attach_to=self.vehicle)
            self.sensors.append(sensor)
            
            # save the relative location
            if self.translation_label_type == 'from':
                translation_label = {
                    'x': x_init-FRONT_X,
                    'y': y_init,
                    'z': z_init-FRONT_Z,
                    'yaw': 0
                }
            else:
                translation_label = {
                    'x': -x_init+FRONT_X,
                    'y': -y_init,
                    'z': -z_init+FRONT_Z,
                    'yaw': 0
                }
            self.sensor_info.append(translation_label)

            # just store the translation label in its own variable for convenience
            self.translation_label = translation_label
                
        # replace the queues

        # do not replace the first two sensors (front depth and front rgb)
        self._queues = self._queues[:3]
        for sensor in self.sensors[2:]:
            self.make_queue(sensor.listen)
     
    def __exit__(self, *args, **kwargs):
        # make sure to clean up the memory
        self.world.apply_settings(self._settings)
        for sensor in self.sensors:
            sensor.destroy()
        # for actor in self.actors:
        #     actor.destroy()
        
    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data       

def save_carla_sensor_img(image, img_name, base_path):
    """Processes a carla sensor.Image object and a velocity int

    Args:
        image (Carla image object): image object to be post processed
        img_name (str): name of the image
        base_path (str): file path to the save folder

    Returns:
        i3 (numpy array): numpy array of the rgb data of the image
    """
    i = np.array(image.raw_data) # the raw data is initially just a 1 x n array
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4)) # raw data is in rgba format
    i3 = i2[:, :, :3] # we just want the rgb data
    
    # save the image
    full_save_path = os.path.join(base_path,
                                    img_name)
    cv2.imwrite(full_save_path, i3)
    return i3

def proximity_filter(depth_image,
                     semantic_map):
    """We do not want to collect data where the 'from' image has large occlusions
    from objects that are directly in front of the camera. Therefore, if any object
    other than a road is within a certain distance from the camera and also fills
    up a threshold of the pixels, we will filter this image out and skip those frames.
    
    Args:
        depth_image (np.array): depth image in BGR format
    """
    # define the filter thresholds
    min_acceptable_distance = 0.01 # will be empirically optimized
    
    # the below is the max percentage of the image that can have objects closer
    # than the specified min acceptable distance
    max_acceptable_occlusion = 0.1 
    
    # get normalized depth image (image is in BGR format) between 0 and 1
    normalized = ((depth_image[:, :, 2] 
                  + 256 * depth_image[:, :, 1] 
                  + 256 * 256 * depth_image[:, :, 0])
                  / (256 * 256 * 256 - 1))
    
    # get the number of pixels that are too close
    temp = normalized < min_acceptable_distance
    
    # we are not worried about whether the road, sidewalk, or road lines are
    # too close, so we will filter out all of the pixels that are too close that
    # are a member of these categories
    road_mult = semantic_map[:, :, 2] != 1
    sidewalk_mult = semantic_map[:, :, 2] != 2
    road_lines_mult = semantic_map[:, :, 2] != 24
    
    num_too_close = np.sum(temp * road_mult * sidewalk_mult * road_lines_mult)
    
    max_close_pixels = IM_HEIGHT * IM_WIDTH * max_acceptable_occlusion
    
    if num_too_close > max_close_pixels:
        return False
    else:
        return True
    
def preprocess_frame_data(aerial_rgb, aerial_depth, translation_label, sensor_params):
    """Preprocess the frame data and get input for the diff view translation
    model.

    This function is modelled after the __getitem__ in the 
    ldm.dataset.custom_dataset.RGBDepthDatasetBase class
    
    """

    # front_img_rgb = frame_info['front_rgb'] / 127.5 - 1
    aerial_img_rgb = aerial_rgb / 127.5 - 1

    aerial_img_depth = preprocess_depth(aerial_depth)

    # front_rgbd = np.concatenate((front_img_rgb, front_img_depth), axis=2)
    aerial_rgbd = np.concatenate((aerial_img_rgb, aerial_img_depth), axis=2)

    aerial_rgbd = torch.tensor(aerial_rgbd).unsqueeze(0)

    # the below is how the model arranges the condition.. taken directly from
    # ldm/model/diffusion/ddpm.py - get_input(self, batch, k)
    ############################################################################
    x = aerial_rgbd
    if len(x.shape) == 3:
        x = x[..., None]
    x = rearrange(x, 'b h w c -> b c h w')
    x = x.to(memory_format=torch.contiguous_format).float()
    ############################################################################

    aerial_rgbd = x

        # normalize the translation label before returning model input
    for key in translation_label:
        min_val = sensor_params['relative_spawn_limits'][key][0]
        max_val = sensor_params['relative_spawn_limits'][key][1]
        if min_val != max_val:
            translation_label[key] = ((translation_label[key] - min_val) / (max_val - min_val)) * 2 - 1

    x = translation_label['x']
    y = translation_label['y']
    z = translation_label['z']
    yaw = translation_label['yaw']

    translation_label = torch.tensor(np.array([[x, y, z, yaw]], dtype='float32')).unsqueeze(0)

    return aerial_rgbd, translation_label

def postprocess_translated_latent_img(img):
    # process translated img
    img = img.squeeze()
    translated_img = torch.clamp(img, -1., 1.)
    translated_img = translated_img.cpu().numpy()
    translated_img = np.transpose(translated_img, (1, 2, 0))
    def denormalize(img): 
        """ Takes an img normalized between [-1, 1] and denormalizes to between 
        [0, 255]
        """
        img = (((img + 1.0) / 2.0) * 255).astype(np.uint8)

        return img
    translated_img = denormalize(translated_img)

    return translated_img
        
def main(model, sampler, opt):

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    # carla boilerplate code
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(2.0)
    world = client.load_world(opt.world)
    # world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    clock = pygame.time.Clock()

    ########################################################################
    # create ego vehicle
    ########################################################################
    # get vehicle blueprint
    bp = blueprint_library.filter("model3")[0]
    
    # spawn vehicle
    if opt.spawn_point == -1:
        spawn_point1 = random.choice(world.get_map().get_spawn_points()) # get a random spawn point from those in the map
    else:
        spawn_points = world.get_map().get_spawn_points()    
        spawn_point1 = spawn_points[opt.spawn_point]
    vehicle1 = world.spawn_actor(bp, spawn_point1) # spawn car at the random spawn point

    ########################################################################
    # define sensor parameters
    ########################################################################

    # TODO: pass this as a config file

    # define initial sensor spawning limits relative to the ego vehicle

    # # CAREFUL: these limits are from the initial sensor relative to the car 
    # initial_spawn_limits = {
    #     'x': [-5, -3],
    #     'y': [-2, 2],
    #     'z': [3, 5.5],
    #     'roll': [0, 0],
    #     'pitch': [0, 0],
    #     'yaw': [0, 0]        
    # }
    
    # blueprint_attributes = {
    #     'image_size_x': f"{IM_HEIGHT}",
    #     'image_size_y': F"{IM_WIDTH}",
    #     'fov': "90"
    # }
    
    # # define what sensors to collect data from at each spawn point
    # sensor_types = [
    #     "sensor.camera.depth",                      # * depth should always be first
    #     # "sensor.camera.semantic_segmentation",
    #     # "sensor.camera.instance_segmentation",
    #     "sensor.camera.rgb"
    # ]
    
    # sensor_params = {
    #     'sensor_types': sensor_types,
    #     'blueprint_attributes': blueprint_attributes,
    #     'initial_spawn_limits': initial_spawn_limits,
    # }
    
    # load the sensor params from the dataset file
    labels_json_path = os.path.join(opt.train_dataset_path,
                                    'labels.json')
    with open(labels_json_path, 'r') as file:
        sensor_params = json.load(file)['sensor_params']
        
    sensor_params['initial_spawn_limits'] = sensor_params['initial_spawn_limits']
    sensor_types = sensor_params['sensor_types']
    
    ########################################################################
    # start the simulation
    ########################################################################
    
    with CarlaSyncMode(world, fps=opt.fps, vehicle=vehicle1, 
                       sensor_params=sensor_params, translation_label_type=opt.trans_label_type) as sync_mode:
        
        # start vehicle autopilot
        sync_mode.vehicle.set_autopilot(True)

        # create variable to hold the information for each frame
        video_info_json = []

        num_frames = opt.fps * opt.vid_duration
        
        # run simulation and collect data
        for frame_count in range(num_frames):
            
            # get frame information
            clock.tick()
            world_data  = sync_mode.tick(timeout=2.0)        
            snapshot = world_data[0]
            front_sensor_data = world_data[1:3]
            aerial_sensor_data = world_data[3:]
            sim_fps = round(1.0 / snapshot.timestamp.delta_seconds)
            true_fps = clock.get_fps()

            frame_info = dict()
            
            # collect the front sensor information
            front_sensor_numpy = dict()
            for idx in range(len(front_sensor_data)):
                data = front_sensor_data[idx]
                sensor_type = sensor_types[idx].split('.')[-1]
                sensor_location = 'front'

                sensor_save_name = f'{sensor_location}_{sensor_type}_{str(frame_count).zfill(6)}'

                # get img from sensor data and save to save_path directory
                img = save_carla_sensor_img(data,
                                            sensor_save_name+'.png',
                                            opt.save_dir)
                
                front_sensor_numpy[sensor_type] = img

                # add img info to the frame info for the the video.json file
                sensor_loc_and_type = f'{sensor_location}_{sensor_type}'
                frame_info[sensor_loc_and_type] = sensor_save_name
            
            # collect the aerial sensor information
            aerial_sensor_numpy = dict()
            for idx in range(len(aerial_sensor_data)):
                data = aerial_sensor_data[idx]
                sensor_type = sensor_types[idx].split('.')[-1]
                sensor_location = 'aerial'

                sensor_save_name = f'{sensor_location}_{sensor_type}_{str(frame_count).zfill(6)}'

                # get img from sensor data and save to save_path directory
                img = save_carla_sensor_img(data,
                                            sensor_save_name+'.png',
                                            opt.save_dir)
                
                aerial_sensor_numpy[sensor_type] = img

                # add img info to the frame info for the video.json file
                sensor_loc_and_type = f'{sensor_location}_{sensor_type}'
                frame_info[sensor_loc_and_type] = sensor_save_name

            # get the translaion label of the aerial sensor
            frame_info['translation_label'] = sync_mode.translation_label

            conditioning, translation_label = preprocess_frame_data(aerial_sensor_numpy['rgb'],
                                                                    aerial_sensor_numpy['depth'],
                                                                    translation_label=frame_info['translation_label'],
                                                                    sensor_params=sensor_params)

            # pass the preprocessed conditioning image (aerial image) through
            # the trained autoencoder to go to the latent space for the diffusion
            conditioning = model.get_learned_conditioning(conditioning.to(model.device))

            # get translated, latent space image from the model
            latent_translated_img, intermediates = sampler.sample(200,
                                                                    1,
                                                                    shape=(3,64,64),
                                                                    conditioning=conditioning,
                                                                    verbose=False,
                                                                    translation_label=translation_label,
                                                                    eta=1.0)
            
            normalized_img = model.decode_first_stage(latent_translated_img)

            translated_img_rgbd = postprocess_translated_latent_img(normalized_img)
            translated_img_rgb = translated_img_rgbd[:, :, :3]
            translated_img_depth = translated_img_rgbd[:, :, -1]

            # save the translated img
            translated_img_rgb_name = f'translated_rgb_{str(frame_count).zfill(6)}'
            translated_img_depth_name = f'translated_depth_{str(frame_count).zfill(6)}'
            translated_img_rgb_save_path = os.path.join(opt.save_dir,
                                                        translated_img_rgb_name+'.png')
            translated_img_depth_save_path = os.path.join(opt.save_dir,
                                                            translated_img_depth_name+'.png')
            cv2.imwrite(translated_img_rgb_save_path, 
                        translated_img_rgb)
            cv2.imwrite(translated_img_depth_save_path, 
                        translated_img_depth)

            frame_info['translated_rgb'] = translated_img_rgb_name
            frame_info['translated_depth'] = translated_img_depth_name

            video_info_json.append(frame_info)
    
            # pick a random location for the aerial sensor
            sync_mode.randomize_aerial_sensor_location()

            # save the complete image for the video
            
            # # convert from BGR image to RGB for pillow compatibility
            # front_rgb_pillow = Image.fromarray(cv2.cvtColor(front_sensor_numpy['rgb'],
            #                                             cv2.COLOR_BGR2RGB))
            # aerial_rgb_pillow = Image.fromarray(cv2.cvtColor(cv2.resize(aerial_sensor_numpy['rgb'], (1024, 1024)),
            #                                             cv2.COLOR_BGR2RGB))
            # translated_rgb_pillow = Image.fromarray(cv2.cvtColor(translated_img_rgb,
            #                                                         cv2.COLOR_BGR2RGB))
            
            # aerial_rgb_pillow.paste(front_rgb_pillow, (0, 0))
            # aerial_rgb_pillow.paste(translated_rgb_pillow, (front_rgb_pillow.width, 0))
            
            # # convert back to open cv2 to display
            # complete_img = cv2.cvtColor(np.array(aerial_rgb_pillow),
            #                             cv2.COLOR_RGB2BGR)
            
            full_front = np.concatenate((front_sensor_numpy['rgb'], 
                                         translated_img_rgb), axis=1)
            aerial_rgb_pillow = Image.fromarray(cv2.cvtColor(cv2.resize(aerial_sensor_numpy['rgb'], (512, 512)),
                                                        cv2.COLOR_BGR2RGB))
            
            # create the 3d location of the aerial view
            fig = plt.figure(figsize=(1, 1))
            ax = fig.add_subplot(111, projection='3d')
            x, y, z, yaw = translation_label.squeeze().cpu().numpy()
            ax.scatter(x, y, z, c='r', s=35)
            ax.view_init(elev=30, azim=45, roll=0)
            ax.set_xlim(0, 1)
            ax.set_xlabel(None)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            location_img_name = os.path.join(opt.save_dir,
                                             f'location_{str(frame_count).zfill(6)}.png')
            plt.savefig(location_img_name)
            
            # cleanup the plt frames every 10 figures
            if frame_count % 10 == 0:
                plt.close('all')
            
            translation_label_img = Image.open(location_img_name)
            
            # paste the translation label image in the bottom right of the image
            paste_position = (aerial_rgb_pillow.height - translation_label_img.height - 1,
                              aerial_rgb_pillow.width - translation_label_img.width - 1)
            aerial_rgb_pillow.paste(translation_label_img, paste_position)
            
            complete_aerial_rgb = np.array(aerial_rgb_pillow)
            
            # concatenate the np arrays together
            complete_img = np.concatenate((full_front, complete_aerial_rgb), axis=0)
            
            if opt.show:
                cv2.imshow('demo img', complete_img)
                cv2.waitKey(500)

            demo_img_save_name = os.path.join(opt.save_dir,
                                        f'demo_frame_{str(frame_count).zfill(6)}.png')

            cv2.imwrite(demo_img_save_name, complete_img)

        with open(os.path.join(opt.save_dir, 'vid_info.json'), 'w') as file:
            json.dump(video_info_json, file)

    # finally:
            
    #     print("All cleaned up!")

def create_video(opt):
    demo_imgs = glob.glob(opt.save_dir+'/*demo_frame*')
    demo_imgs.sort()

    # create video writer
    video_name = os.path.join('demo_vid.mp4')
    video = cv2.VideoWriter(video_name,
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            opt.fps,
                            (512+256, 512))

    for img in demo_imgs:
        img = cv2.imread(img)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()    

def get_parser():
    parser = argparse.ArgumentParser()

    # video recording parameters
    parser.add_argument(
        "--verify_dataset", 
        action='store_true', 
        help='Run dataset verification')
    
    parser.add_argument(
        '--train_dataset_path',
        action='store',
        type=str,
        help='Specify the location of the train dataset folder')
    
    parser.add_argument(
        '--trans_label_type',
        action='store',
        type=str,
        choices=['to', 'from'],
        help='Specify the type of translation label')
    
    parser.add_argument(
        '--vid_duration',
        action='store',
        default=60,
        type=int,
        help='Specify the duration of the video to take')
    
    parser.add_argument(
        '--fps',
        action='store',
        default=30,
        type=int,
        help='Specify the fps at which to record a video')
    
    parser.add_argument(
        '--save_dir',
        action='store',
        default=SAVE_DIR,
        type=str,
        help='Path to store video information.')
    
    parser.add_argument(
        '--world',
        action='store',
        default='town01',
        type=str,
        help='Specify the world to collect data on')
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Specify whether you want to show the images while running the demo')
    
    parser.add_argument(
        '--spawn_point',
        action='store',
        default='-1',
        type=int,
        help='Specify spawn point. -1 means random')
    
    # default diffusion inference parameters
    parser.add_argument(
        "--diff_ckpt",
        type=str,
        nargs="?",
        help="diffusion model checkpoint path",
    )
    # parser.add_argument(
    #     "--autoencoder_ckpt",
    #     type=str,
    #     nargs="?",
    #     help="autoencoder checkpoint path",
    # )
    parser.add_argument(
        "--diff_config",
        type=str,
        nargs="?",
        help="diffusion model config path",
    )
    # parser.add_argument(
    #     "--autoencoder_config",
    #     type=str,
    #     nargs="?",
    #     help="autoencoder config path",
    # )
    # parser.add_argument(
    #     "--sample_data_folder",
    #     type=str,
    #     nargs="?",
    #     help="specify the folder containing the data to use to sample translation",
    # )
    # parser.add_argument(
    #     "--save_dir",
    #     type=str,
    #     nargs="?",
    #     help="specify the folder to save the sampled images to",
    # )
    # parser.add_argument(
    #     "-n",
    #     "--n_samples",
    #     type=int,
    #     nargs="?",
    #     help="number of samples to draw",
    #     default=100
    # )
    # parser.add_argument(
    #     "-e",
    #     "--eta",
    #     type=float,
    #     nargs="?",
    #     help="eta for ddim sampling (0.0 yields deterministic sampling)",
    #     default=1.0
    # )
    # parser.add_argument(
    #     "-v",
    #     "--vanilla_sample",
    #     default=False,
    #     action='store_true',
    #     help="vanilla sampling (default option is DDIM sampling)?",
    # )
    # parser.add_argument(
    #     "-l",
    #     "--logdir",
    #     type=str,
    #     nargs="?",
    #     help="extra logdir",
    #     default="none"
    # )
    # parser.add_argument(
    #     "-c",
    #     "--custom_steps",
    #     type=int,
    #     nargs="?",
    #     help="number of steps for ddim and fastdpm sampling",
    #     default=50
    # )
    # parser.add_argument(
    #     "--batch_size",
    #     type=int,
    #     nargs="?",
    #     help="the bs",
    #     default=10
    # )
    return parser
         
if __name__=='__main__':
    
    # load all of the configs for the model
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = opt.diff_ckpt
    opt.base = [opt.diff_config]

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    # if opt.logdir != "none":
    #     logdir = opt.logdir

    print(config)

    # instantiate the trained model
    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    sampler = DDIMSampler(model)

    try:
        main(model, sampler, opt)
        create_video(opt)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')