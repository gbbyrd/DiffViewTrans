# ==============================================================================
# -- set carla path -------------------------------------------------------------------
# ==============================================================================

import os
import sys

CARLA_EGG_PATH = os.environ.get('CARLA_EGG')

try:
    sys.path.append(CARLA_EGG_PATH)
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

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

# global variables for quick data collection finetuning
DATASET_PATH = 'C:\\Users\\gbbyrd\\Desktop\\thesis\\data\\3d_trans_multi_sensor_v3_large'
NUM_IMAGES = 400
IM_HEIGHT, IM_WIDTH = 256, 256

class CarlaSyncMode(object):
    """Class for running Carla in synchronous mode. Allows frame rate sync to
    ensure no data is lost and all backend code runs and is completed before
    'ticking' to the next frame."""
    
    def __init__(self, world, **kwargs):
        self.world = world
        self.sensors = []
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None
        self.sensor_params = kwargs.get('sensor_params', None)
        self.blueprint_library = world.get_blueprint_library()
        self.sensor_info = []
        
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

        self.initialize_sensors()
        
        return self
    
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
    
    def initialize_sensors(self):
        """Create and spawn all sensors at location (0, 0, 0)"""
        
        num_aux_sensors = self.sensor_params['num_aux_sensors']
        sensor_types = self.sensor_params['sensor_types']
        blueprint_attributes = self.sensor_params['blueprint_attributes']
        initial_spawn_limits = self.sensor_params['initial_spawn_limits']
        relative_spawn_limits = self.sensor_params['relative_spawn_limits']
        
        # there will be num_aux_sensors + 1 sensor locations (1 for the initial
        # sensor). each sensor location will spawn 1 sensor for every sensor type
        # specified, so the total number of sensors spawned will be:
        #
        # total_num_sensors = (num_aux_sensors + 1) * len(sensor_types)
        for i in range(num_aux_sensors+1):
            for sensor_type in sensor_types:
                # create each sensor
                sensor_bp = self.blueprint_library.find(sensor_type)
                for attribute in blueprint_attributes:
                    sensor_bp.set_attribute(attribute, blueprint_attributes[attribute])

                # spawn the sensor at 0,0,0
                spawn_point = carla.Transform(carla.Location(x=0, y=0, z=0),
                                              carla.Rotation(roll=0, pitch=0, yaw=0))
                sensor = self.world.spawn_actor(sensor_bp, spawn_point)
                self.sensors.append(sensor)

        # add each sensor to the queue
        for sensor in self.sensors:
            self.make_queue(sensor.listen)
    
    def randomize_sensors(self):
        """Randomize the sensors within the limits specified in 
        self.sensor_params
        """
        
        num_aux_sensors = self.sensor_params['num_aux_sensors']
        sensor_types = self.sensor_params['sensor_types']
        blueprint_attributes = self.sensor_params['blueprint_attributes']
        initial_spawn_limits = self.sensor_params['initial_spawn_limits']
        relative_spawn_limits = self.sensor_params['relative_spawn_limits']
        
        # clear sensor information
        self.sensor_info.clear()
        
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
        
        # change intitial sensor location (1 for each type of sensor)
        sensor_idx = 0
        while sensor_idx < len(sensor_types):
            self.sensors[sensor_idx].set_transform(carla.Transform(carla.Location(x_init, y_init, z_init),
                                                        carla.Rotation(roll_init, yaw_init, pitch_init)))
            
            # save the relative location data (0 since this is the init sensor loc)
            relative_location = {
                'x': 0,
                'y': 0,
                'z': 0,
                'yaw': 0
            }
            self.sensor_info.append(relative_location)
            sensor_idx += 1
            
        # get spawn limits for auxiliary sensors
        x_lim_rel = relative_spawn_limits['x']
        y_lim_rel = relative_spawn_limits['y']
        z_lim_rel = relative_spawn_limits['z']
        roll_lim_rel = relative_spawn_limits['roll']
        pitch_lim_rel = relative_spawn_limits['pitch']
        yaw_lim_rel = relative_spawn_limits['yaw']
        
        # clear out the rest of the sensors. since we have to attach the aux
        # sensors to the original one, we will have to destroy them and create
        # all new sensors..
        for idx in range(sensor_idx, len(self.sensors)):
            sensor = self.sensors[idx]
            sensor.destroy()
        self.sensors = self.sensors[:sensor_idx]
        
        # create and spawn aux sensors relative to the initial sensor
        for _ in range(num_aux_sensors):
            
            # generate random relative location
            x_rel = random.uniform(x_lim_rel[0], x_lim_rel[1])
            y_rel = random.uniform(y_lim_rel[0], y_lim_rel[1])
            z_rel = random.uniform(z_lim_rel[0], z_lim_rel[1])
            roll_rel = random.uniform(roll_lim_rel[0], roll_lim_rel[1])
            pitch_rel = random.uniform(pitch_lim_rel[0], pitch_lim_rel[1])
            yaw_rel = random.uniform(yaw_lim_rel[0], yaw_lim_rel[1])
            
            # get global location
            x = x_init + x_rel
            y = y_init + y_rel
            z = z_init + z_rel
            roll = roll_init + roll_rel
            pitch = pitch_init + pitch_rel
            yaw = yaw_init + yaw_rel
            
            # ensure that the global location is within the init limits..
            # truncate to the max/min global location if needed
            x = max(x, x_lim_init[0])
            x = min(x, x_lim_init[1])
            y = max(y, y_lim_init[0])
            y = min(y, y_lim_init[1])
            z = max(z, z_lim_init[0])
            z = min(z, z_lim_init[1])
            roll = max(roll, roll_lim_init[0])
            roll = min(roll, roll_lim_init[1])
            pitch = max(pitch, pitch_lim_init[0])
            pitch = min(pitch, pitch_lim_init[1])
            yaw = max(yaw, yaw_lim_init[0])
            yaw = min(yaw, yaw_lim_init[1])
            
            # get final, truncated relative location
            x_rel = x - x_init
            y_rel = y - y_init
            z_rel = z - z_init
            yaw_rel = yaw - yaw_init
            
            # spawn the new sensor/s
            for sensor_type in sensor_types:
                # create sensor
                sensor_bp = self.blueprint_library.find(sensor_type)
                for attribute in blueprint_attributes:
                    sensor_bp.set_attribute(attribute, blueprint_attributes[attribute])

                spawn_point = carla.Transform(carla.Location(x=x_rel, y=y_rel, z=z_rel),
                                              carla.Rotation(roll=0, yaw=yaw_rel, pitch=0))
                
                # spawn the sensor relative to the first sensor
                sensor = self.world.spawn_actor(sensor_bp, spawn_point, attach_to=self.sensors[0])
                self.sensors.append(sensor)
                
                # save the relative location
                relative_location = {
                    'x': x_rel,
                    'y': y_rel,
                    'z': z_rel,
                    'yaw': yaw_rel
                }
                self.sensor_info.append(relative_location)
                sensor_idx += 1
                
        # replace the queues
        self._queues = self._queues[:len(sensor_types)+1]
        for sensor in self.sensors[len(sensor_types):]:
            self.make_queue(sensor.listen)
     
    def __exit__(self, *args, **kwargs):
        # make sure to clean up the memory
        self.world.apply_settings(self._settings)
        for sensor in self.sensors:
            sensor.destroy()
        
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

def proximity_filter(depth_image):
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
    num_too_close = np.sum(temp)
    
    max_close_pixels = IM_HEIGHT * IM_WIDTH * max_acceptable_occlusion
    
    if num_too_close > max_close_pixels:
        return False
    else:
        return True
    
def main():
    
    # make life easier..
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
        
    actor_list = []

    try:
        # carla boilerplate variables
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        clock = pygame.time.Clock()
        
        ########################################################################
        # define sensor parameters (fine tune data control)
        ########################################################################
        
        # define sensor spawning limits
        initial_spawn_limits = {
            'x': [-100, 100],
            'y': [-80, -50],
            'z': [0.6, 25],
            'roll': [0, 0],
            'pitch': [0, 0],
            'yaw': [-180, 180]        
        }
        
        # how far can an auxiliary sensor spawn from the initial sensor
        relative_spawn_limits = {
            'x': [5, 7],
            'y': [-2, 2],
            'z': [-2, 2],
            'roll': [0, 0],
            'pitch': [0, 0],
            'yaw': [0, 0]
        }
        
        blueprint_attributes = {
            'image_size_x': f"{IM_HEIGHT}",
            'image_size_y': F"{IM_WIDTH}",
            'fov': "90"
        }
        
        # define what sensors to collect data from at each spawn point
        num_aux_sensors = 3
        sensor_types = [
            "sensor.camera.depth",                      # * depth should always be first
            # "sensor.camera.semantic_segmentation",
            "sensor.camera.instance_segmentation",
            "sensor.camera.rgb"
        ]
        
        sensor_params = {
            'num_aux_sensors': num_aux_sensors,
            'sensor_types': sensor_types,
            'blueprint_attributes': blueprint_attributes,
            'initial_spawn_limits': initial_spawn_limits,
            'relative_spawn_limits': relative_spawn_limits
        }
        
        ########################################################################
        # functionality for adding to existing dataset
        ########################################################################
        
        # search to see if there are any images/labels in the data directory
        img_files = glob.glob(DATASET_PATH+'\\*.png')
        
        # if files were found, do some additional work to add data to the dataset
        if len(img_files) > 0:
            # load the labels file
            with open(DATASET_PATH+'\\labels.json', 'r') as file:
                prev_data = json.load(file)
                prev_label_data = prev_data['data']
                prev_sensor_params = prev_data['sensor_params']
        
            # find the count of the final image and set the train_img_count
            train_img_count = len(img_files) // len(sensor_types)
            
            # increase NUM_IMAGES by the initial train_img_count
            num_images = train_img_count + NUM_IMAGES
        else:
            train_img_count = 0
            prev_label_data = None
            prev_sensor_params = None
            num_images = NUM_IMAGES
        
        ########################################################################
        # start the simulation
        ########################################################################
        
        with CarlaSyncMode(world, fps=30, sensor_params=sensor_params) as sync_mode:
            
            # ensure that if you add to existing datasets, your dataset parameters
            # are the same
            if prev_sensor_params:
                assert prev_sensor_params == sensor_params, "Error: To add to a dataset you must have the same sensor limits."
            
            # create variables for storing data
            labels = dict()
            sensor_groups = []
            
            pbar = tqdm(desc='Generating training images', total=num_images-train_img_count)
            
            frame_count = 0
            sync_mode.randomize_sensors()
            
            # run simulation and collect data
            while train_img_count < num_images:
                
                # get frame information
                clock.tick()
                world_data  = sync_mode.tick(timeout=2.0)        
                snapshot = world_data[0]
                sensor_data = world_data[1:]
                sim_fps = round(1.0 / snapshot.timestamp.delta_seconds)
                true_fps = clock.get_fps()
                
                # define dictionary for saving the frame data
                sensor_group_data = dict()
                
                # save the rest of the training imgs
                save_labels = True
                for idx, data in enumerate(sensor_data):
                    # if initial sensor, specify from, else to
                    if idx // len(sensor_types) == 0:
                        data_type = 'from'
                    else:
                        data_type = 'to'
                    
                    # save image
                    sensor_type = sensor_types[idx % len(sensor_types)].split('.')[-1]
                    train_img_name = f'{data_type}_{sensor_type}_{train_img_count}.png'
                    img = save_carla_sensor_img(data,
                                      train_img_name,
                                      DATASET_PATH)
                    
                    # filter out images with extreme occlusions
                    if sensor_type == 'depth' and data_type == 'from':
                        if not proximity_filter(img):
                            save_labels = False
                            break
                    
                    # get img information to save to labels dictionary
                    train_img_info = dict()
                    train_img_info['img_name'] = train_img_name
                    train_img_info['location'] = sync_mode.sensor_info[idx]
                    sensor_group_data[f'{sensor_type}_img_{train_img_count}_info'] = train_img_info
                    
                    # increment train_img_count every -len(sensor_types)- img saves
                    if (idx+1) % len(sensor_types) == 0:
                        train_img_count += 1
                
                if save_labels:
                    # add frame dict to label dict
                    sensor_groups.append(sensor_group_data)
                    
                    frame_count += 1
                    pbar.update(len(sensor_data)//len(sensor_types))
                    
                # randomize sensor locations
                sync_mode.randomize_sensors()
                
            labels['data'] = sensor_groups
            if prev_label_data:
                labels['data'] += prev_label_data
            labels['sensor_params'] = sensor_params
            
            labels_path = os.path.join(DATASET_PATH,
                                       'labels.json')
            with open(labels_path, 'w') as file:
                json.dump(labels, file)
                
        # create a synchronous mode context
        time.sleep(1)
        
    finally:
        # for actor in actor_list:
        #     actor.destroy()
        
        # save the labels if there is an error that broke the simulation before
        # completing the data collection
        labels_path = os.path.join(DATASET_PATH,
                                       'labels.json')
        if not os.path.exists(labels_path):
            labels['data'] = sensor_groups
            labels['sensor_params'] = sensor_params
            
            with open(labels_path, 'w') as file:
                json.dump(labels, file)
            
        print("All cleaned up!")
        
def verify_dataset():
    labels_json_path = os.path.join(DATASET_PATH,
                                    'labels.json')
    
    # load the labels
    with open(labels_json_path, 'r') as file:
        data = json.load(file)
        
    label_data = data['data']
    sensor_params = data['sensor_params']
    sensor_types = sensor_params['sensor_types']
    
    while 1:
        idx = random.randint(0, len(label_data))
        label = label_data[idx]
        
        # save all sensor names grouped by image location
        sensor_group_img_names = list()
        group_names = list()
        for idx, img_info in enumerate(label.values()):
            group_names.append(os.path.join(DATASET_PATH,
                                            img_info['img_name']))
            if (idx + 1) % len(sensor_types) == 0:
                sensor_group_img_names.append(group_names.copy())
                group_names.clear()
                
        # loop through, read images, concatenate group-wise and then concatenate
        # each group
        concatenated_imgs = []
        for group_idx, group in enumerate(sensor_group_img_names):
            # create tuple of size len(sensor_types)
            imgs = []
            
            for img_name in group:
                imgs.append(cv2.imread(img_name))
                
            # concatenate images
            concated_img = np.concatenate(tuple(imgs), axis=1)
            concatenated_imgs.append(concated_img.copy())
        
        # concatenate all imgs and display
        full_concat = np.concatenate(tuple(concatenated_imgs), axis=0)
        
        cv2.imshow('sensor_group_data', full_concat)
        cv2.waitKey(5000)
        
        
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--verify_dataset", 
                        action='store_true', 
                        help='Run dataset verification')
    
    args = parser.parse_args()
    
    try:
        if args.verify_dataset:
            verify_dataset()
        else:
            main()
        
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')