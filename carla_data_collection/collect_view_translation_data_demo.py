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

from math import sqrt

# global variables for quick data collection finetuning
DATASET_PATH = '/home/nianyli/Desktop/code/thesis/DiffViewTrans/data/vt_town01_dataset'
NUM_FRAMES = 100
IM_HEIGHT, IM_WIDTH = 256, 256

# define the x and z locations that the 'to' spawned cameras will spawn at
FRONT_X = 2.5
FRONT_Z = 0.7

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

        self.randomize_sensors()
        
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
    
    def initialize_front_sensors(self):
        """Create and spawn all sensor types at front location"""
        
        num_aux_sensors = self.sensor_params['num_aux_sensors']
        sensor_types = self.sensor_params['sensor_types']
        blueprint_attributes = self.sensor_params['blueprint_attributes']
        initial_spawn_limits = self.sensor_params['initial_spawn_limits']
        relative_spawn_limits = self.sensor_params['relative_spawn_limits']
        
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

        # # add each sensor to the queue
        # for sensor in self.sensors:
        #     self.make_queue(sensor.listen)
    
    def randomize_sensors(self):
        """Randomize the sensors within the limits specified in 
        self.sensor_params
        """
        
        # need to subtract num_aux_sensors by 1 because we add the ground 
        # truth front location of the car at every frame
        num_aux_sensors = self.sensor_params['num_aux_sensors']-1
        sensor_types = self.sensor_params['sensor_types']
        blueprint_attributes = self.sensor_params['blueprint_attributes']
        initial_spawn_limits = self.sensor_params['initial_spawn_limits']
        relative_spawn_limits = self.sensor_params['relative_spawn_limits']
        
        # clear sensor information
        self.sensor_info = []

        # clear out the rest all sensors
        for sensor in self.sensors:
            sensor.destroy()
        self.sensors = []
        
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
            relative_location = {
                'x': 0,
                'y': 0,
                'z': 0,
                'yaw': 0
            }
            self.sensor_info.append(relative_location)
            
        # get spawn limits for auxiliary sensors (don't need x and z limits as
        # these values are fixed)
        y_lim_rel = relative_spawn_limits['y'].copy()
        roll_lim_rel = relative_spawn_limits['roll'].copy()
        pitch_lim_rel = relative_spawn_limits['pitch'].copy()
        yaw_lim_rel = relative_spawn_limits['yaw'].copy()
        
        # bin each sensor spawn to regulate the data distribution
        bins = []
        intervals = np.linspace(relative_spawn_limits['y'][0],
                                relative_spawn_limits['y'][1],
                                num_aux_sensors+1)
        for idx in range(len(intervals)-1):
            bins.append([intervals[idx], intervals[idx+1]])
        
        # create and spawn aux sensors relative to the initial sensor
        for idx in range(num_aux_sensors):

            # generate random relative location
            x_rel = FRONT_X
            y_rel = random.uniform(bins[idx][0], bins[idx][1]) + y_init
            z_rel = FRONT_Z
            roll_rel = random.uniform(roll_lim_rel[0], roll_lim_rel[1])
            pitch_rel = random.uniform(pitch_lim_rel[0], pitch_lim_rel[1])
            yaw_rel = random.uniform(yaw_lim_rel[0], yaw_lim_rel[1])
            
            # spawn the new sensor/s
            for sensor_type in sensor_types:
                # create sensor
                sensor_bp = self.blueprint_library.find(sensor_type)
                for attribute in blueprint_attributes:
                    sensor_bp.set_attribute(attribute, blueprint_attributes[attribute])

                spawn_point = carla.Transform(carla.Location(x=x_rel, y=y_rel, z=z_rel),
                                              carla.Rotation(roll=roll_rel, yaw=yaw_rel, pitch=pitch_rel))
                
                # spawn the sensor relative to the first sensor
                sensor = self.world.spawn_actor(sensor_bp, spawn_point, attach_to=self.vehicle)
                self.sensors.append(sensor)
                
                # save the location of the initial sensor relative to the auxiliary sensor
                relative_location = {
                    'x': x_init - x_rel,
                    'y': y_init - y_rel,
                    'z': z_init - z_rel,
                    'yaw': yaw_init - yaw_rel
                }
                self.sensor_info.append(relative_location)
                
        # create the front sensors for the frame
        self.initialize_front_sensors()
        
        # add the relative location for the sensor spawned at the front of the
        # vehicle
        front_relative_to_initial = {
            'x': x_init-FRONT_X,
            'y': y_init,
            'z': z_init-FRONT_Z,
            'yaw': yaw_init
        }
        
        for sensor_type in sensor_types:
            self.sensor_info.append(front_relative_to_initial)
                
        # replace the queues
        self._queues = self._queues[:1]
        for sensor in self.sensors:
            self.make_queue(sensor.listen)
     
    def __exit__(self, *args, **kwargs):
        # make sure to clean up the memory
        self.world.apply_settings(self._settings)
        for sensor in self.sensors:
            sensor.destroy()
        for actor in self.actors:
            actor.destroy()
        
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
    
def main():
    
    # make life easier..
    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)
        
    actor_list = []

    try:
        # carla boilerplate variables
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(2.0)
        # world = client.load_world(args.world)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        clock = pygame.time.Clock()
        
        ########################################################################
        # define sensor parameters (fine tune data control)
        ########################################################################
        
        """ The data collection works in the following way:

        1. Spawn an expert vehicle that runs on Carla's autopilot.
        2. At each frame:
            a. Spawn an initial sensor that represents the 'from' image at a
                random location that is within defined boundaries
            b. Spawn multiple auxiliary sensors off of that initial sensor
                randomly, within a set of defined boundaries
        3. Collect the depth, rgb, and semantic segmentation images for the
            dataset
        
        """

        # define initial sensor spawning limits relative to the ego vehicle

        # CAREFUL: these limits are from the initial sensor relative to the car 
        initial_spawn_limits = {
            'x': [-5, -3],
            'y': [-2, 2],
            'z': [3, 5.5],
            'roll': [0, 0],
            'pitch': [0, 0],
            'yaw': [0, 0]        
        }
        
        # define how far an auxiliary sensor can spawn from the initial sensor
        # note: the aux sensors will always spawn on the x and z axis at the front of
        # the vehicle, but the y value will vary to ensure that the model learns
        # to use the distance label

        # CAREFUL: these limits are from the initial sensor relative to the auxiliary sensor
        relative_spawn_limits = {
            'x': [-5 - FRONT_X, -3 - FRONT_X], # this is needed
            'y': [-2, 2],
            'z': [3 - FRONT_Z, 5.5 - FRONT_Z],
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
        num_aux_sensors = 8
        sensor_types = [
            "sensor.camera.depth",                      # * depth should always be first
            # "sensor.camera.semantic_segmentation",
            # "sensor.camera.instance_segmentation",
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
        img_files = glob.glob(args.dataset_path+'/*.png')
        
        # if files were found, do some additional work to add data to the dataset
        if len(img_files) > 0:
            # load the labels file
            with open(args.dataset_path+'/labels.json', 'r') as file:
                prev_data = json.load(file)
                prev_label_data = prev_data['data']
                prev_sensor_params = prev_data['sensor_params']
        
            # find the count of the final image and set the train_img_count
            train_img_count = len(img_files) // len(sensor_types)
            
            # increase NUM_IMAGES by the initial train_img_count
            num_images = train_img_count + args.num_frames
        else:
            train_img_count = 0
            prev_label_data = None
            prev_sensor_params = None
            num_images = args.num_frames

        ########################################################################
        # create ego vehicle
        ########################################################################
        # get vehicle blueprint
        bp = blueprint_library.filter("model3")[0]
        
        # spawn vehicle
        spawn_point1 = random.choice(world.get_map().get_spawn_points()) # get a random spawn point from those in the map
        # spawn_points = world.get_map().get_spawn_points()    
        # spawn_point1 = spawn_points[0]
        vehicle1 = world.spawn_actor(bp, spawn_point1) # spawn car at the random spawn point
        actor_list.append(vehicle1)

        ########################################################################
        # start the simulation
        ########################################################################
        
        with CarlaSyncMode(world, fps=30, vehicle=vehicle1, sensor_params=sensor_params) as sync_mode:
            
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
            sync_mode.vehicle.set_autopilot(True)
            
            # run simulation and collect data
            while train_img_count < num_images:
                
                # get frame information
                clock.tick()
                world_data  = sync_mode.tick(timeout=2.0)        
                snapshot = world_data[0]
                sensor_data = world_data[1:]
                sim_fps = round(1.0 / snapshot.timestamp.delta_seconds)
                true_fps = clock.get_fps()

                # only collect data from every 5th frame
                if frame_count % args.skip_frames != 0:
                    frame_count += 1
                    continue

                # if the car is stopped (like at a stop light) do not collect data
                # as this will heavily bias the dataset
                vel = sqrt(sync_mode.vehicle.get_velocity().x**2 + sync_mode.vehicle.get_velocity().y**2)
                if vel < .001:
                    frame_count += 1
                    continue
                
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
                    train_img_name = f'{data_type}_{sensor_type}_{str(train_img_count).zfill(6)}.png'
                    img = save_carla_sensor_img(data,
                                      train_img_name,
                                      args.dataset_path)
                    
                    # filter out images with extreme occlusions
                    # if sensor_type == 'depth' and data_type == 'from':

                    #     if not proximity_filter(img):
                    #         save_labels = False
                    #         break
                    
                    # get img information to save to labels dictionary
                    train_img_info = dict()
                    train_img_info['img_name'] = train_img_name
                    train_img_info['location'] = sync_mode.sensor_info[idx]
                    train_img_info['sensor_type'] = sensor_types[idx % len(sensor_types)].split('.')[-1]
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
            
            labels_path = os.path.join(args.dataset_path,
                                       'labels.json')
            with open(labels_path, 'w') as file:
                json.dump(labels, file)
                
        # create a synchronous mode context
        time.sleep(1)
        
    finally:
        for actor in actor_list:
            actor.destroy()
        
        # save the labels if there is an error that broke the simulation before
        # completing the data collection
        labels['data'] = sensor_groups
        if prev_label_data:
            labels['data'] += prev_label_data
        labels['sensor_params'] = sensor_params
        
        labels_path = os.path.join(args.dataset_path,
                                    'labels.json')
        with open(labels_path, 'w') as file:
            json.dump(labels, file)
            
        print("All cleaned up!")
        
def verify_dataset():
    """This function is meant to ensure that the above collected dataset adheres
    to the following rules:
    
    1. Every img name in the json labels file has a corresponding saved image in
    the dataset file path.
    2. Every saved image file has a corresponding json labels file img name.
    3. Every collected image from each sensor is sequential with no gaps.
    
    WARNING: This only works if you collect sensor data from CAMERAS
    """

    # TODO: There is an error in the way the data is collected. Switch the 
    # sensor type in the img name from 'semantic' to 'semantic_segmentation'
    dataset_path = args.dataset_path

    # get the dataset information
    with open(dataset_path+'/labels.json', 'r') as file:
        dataset = json.load(file)
        
    sensor_params = dataset['sensor_params']
    data = dataset['data']
    
    sensor_types = sensor_params['sensor_types']
    for idx, sensor_type in enumerate(sensor_types):
        sensor_types[idx] = sensor_type.split('.')[-1]
    
    # get img names in the dataset
    dataset_img_names = set()
    for group_info in data:
        for img in group_info:
            dataset_img_names.add(group_info[img]['img_name'])
    
    # get all of the saved images
    saved_img_paths = glob.glob(dataset_path+'/*.png')
    
    for idx, saved_img_path in enumerate(saved_img_paths):
        saved_img_paths[idx] = saved_img_path.split('/')[-1]
    
    saved_imgs = dict()
    for sensor_type in sensor_types:
        saved_imgs[sensor_type] = []
    
    for saved_img_path in saved_img_paths:
        sensor_type = saved_img_path.split('_')[1]

        assert sensor_type in saved_imgs, 'Error: invalid sensor type in img name!'

        saved_imgs[sensor_type].append(saved_img_path)
    
    def get_substr(string):
        return string[-10:-4]
    
    for sensor_type in saved_imgs:
        # TODO: Fix the below hardcoding.. ew
        saved_imgs[sensor_type].sort(key=lambda x: x[-10:-4])
    
    # verify that every saved image has a corresponding image name in the labels
    # file and that the images are saved in sequential order with no numbers
    # missing
    assert len(dataset_img_names) == len(saved_img_paths), 'Error: Mismatched saved images and labels!'
    
    imgs_not_saved = dict()
    for sensor_type in sensor_types:
        imgs_not_saved[sensor_type] = []

    depth_imgs_not_saved = []
    rgb_imgs_not_saved = []
    semantic_segmentation_imgs_not_saved = []
    
    for sensor_type in saved_imgs:
        count = 0
        idx = 0
        while idx < len(saved_imgs[sensor_type]):
            img_path = saved_imgs[sensor_type][idx]
            if int(img_path[-10:-4]) == count:
                idx += 1
            else:
                imgs_not_saved[sensor_type].append(str(count).zfill(6))

            count += 1
        
    # loop through each saved image and ensure that there is a corresponding label
    # for that image
    missing_labels = dict()
    for sensor_type in sensor_types:
        missing_labels[sensor_type] = []

    missing_depth_labels = []
    missing_rgb_labels = []
    missing_semantic_segmentation_labels = []
    
    for sensor_type in missing_labels:
        for img_name in saved_imgs[sensor_type]:
            if img_name not in dataset_img_names:
                missing_labels[sensor_type].append(img_name)

    for sensor_type in imgs_not_saved:
        print(f'skipped {sensor_type} imgs: {len(imgs_not_saved[sensor_type])}')

    for sensor_type in missing_labels:
        print(f'missing {sensor_type} labels: {len(missing_labels[sensor_type])}')
    
def clean_dataset():

    dataset_path = args.dataset_path
    
    with open(os.path.join(dataset_path, 'labels.json'), 'r') as file:
        dataset_info = json.load(file)

    data = dataset_info['data']
    names_from_json = set()
    for idx, dic in enumerate(data):
        for key in dic:
            names_from_json.add(dic[key]['img_name'])
        
    saved_imgs = glob.glob(dataset_path+'/*.png')
    for idx, img in enumerate(saved_imgs):
        saved_imgs[idx] = img.split('/')[-1]
        
    saved_imgs.sort()

    deleted_images = []

    for img_name in saved_imgs:
        if img_name not in names_from_json:
            os.remove(os.path.join(dataset_path, img_name))
            deleted_images.append(img_name)
         
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--verify_dataset", 
        action='store_true', 
        help='Run dataset verification')
    
    parser.add_argument(
        "--clean_dataset", 
        action='store_true', 
        help='Run dataset cleaning')
    
    parser.add_argument(
        '--num_frames',
        action='store',
        default=NUM_FRAMES,
        type=int,
        help='Specify number of data frames to collect')
    
    parser.add_argument(
        '--dataset_path',
        action='store',
        default=DATASET_PATH,
        type=str,
        help='Specify the path to save the dataset')
    
    parser.add_argument(
        '--world',
        action='store',
        default='town01',
        type=str,
        help='Specify the world to collect data on')
    
    parser.add_argument(
        '--skip_frames',
        action='store',
        default=5,
        type=int,
        help='Specify the frame interval with which to collect the data.')
    
    args = parser.parse_args()
    
    try:
        if args.verify_dataset:
            verify_dataset()
        elif args.clean_dataset:
            clean_dataset()
        else:
            main()
        
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')