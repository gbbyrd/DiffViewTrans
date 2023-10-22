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
DATASET_PATH = '/home/nianyli/Desktop/code/thesis/DiffViewTrans/data/town01_bc_dataset'
NUM_FRAMES = 100
IM_HEIGHT, IM_WIDTH = 256, 256

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
        
        sensor_types = self.sensor_params['sensor_types']
        blueprint_attributes = self.sensor_params['blueprint_attributes']
        
        # spawn the front sensor
        for sensor_type in sensor_types:
            # create each sensor
            sensor_bp = self.blueprint_library.find(sensor_type)
            for attribute in blueprint_attributes:
                sensor_bp.set_attribute(attribute, blueprint_attributes[attribute])

            # spawn the sensor at 0,0,0
            spawn_point = carla.Transform(carla.Location(x=2.5, y=0, z=0.7),
                                            carla.Rotation(roll=0, pitch=0, yaw=0))
            sensor = self.world.spawn_actor(sensor_bp, spawn_point, attach_to=self.vehicle)
            self.sensors.append(sensor)

        # add each sensor to the queue
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
        
        blueprint_attributes = {
            'image_size_x': f"{IM_HEIGHT}",
            'image_size_y': F"{IM_WIDTH}",
            'fov': "90"
        }
        
        # define what sensors to collect data from at each spawn point
        sensor_types = [
            "sensor.camera.depth",                      # * depth should always be first
            # "sensor.camera.semantic_segmentation",
            # "sensor.camera.instance_segmentation",
            "sensor.camera.rgb"
        ]
        
        sensor_params = {
            'sensor_types': sensor_types,
            'blueprint_attributes': blueprint_attributes
        }
        
        ########################################################################
        # functionality for adding to existing dataset
        ########################################################################
        
        # search to see if there are any images/labels in the data directory
        img_files = glob.glob(DATASET_PATH+'/*.png')
        
        # if files were found, do some additional work to add data to the dataset
        if len(img_files) > 0:
            # load the labels file
            with open(DATASET_PATH+'/labels.json', 'r') as file:
                prev_data = json.load(file)
                prev_label_data = prev_data['data']
                prev_sensor_params = prev_data['sensor_params']
        
            # find the count of the final image and set the train_img_count
            prev_num_frames = len(prev_label_data)
        else:
            prev_num_frames = 0
            prev_label_data = None
            prev_sensor_params = None

        ########################################################################
        # create ego vehicle
        ########################################################################
        # get vehicle blueprint
        bp = blueprint_library.filter("model3")[0]
        
        # spawn vehicle
        # spawn_point1 = random.choice(world.get_map().get_spawn_points()) # get a random spawn point from those in the map
        spawn_points = world.get_map().get_spawn_points()    
        spawn_point1 = spawn_points[0]
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
            added_frames = []

            frames_to_add = args.num_frames
            pbar = tqdm(desc='Generating training images', total=frames_to_add)
            
            frame_count = 0
            sync_mode.vehicle.set_autopilot(True)

            frames_added = 0
            control = None

            # run simulation and collect data
            frames_stopped = 0
            while frames_added < frames_to_add:
                
                # get frame information
                clock.tick()
                world_data  = sync_mode.tick(timeout=2.0)        
                snapshot = world_data[0]
                sensor_data = world_data[1:]
                sim_fps = round(1.0 / snapshot.timestamp.delta_seconds)
                true_fps = clock.get_fps()

                if len(added_frames) != 0 and 'control' not in added_frames[-1]:
                    # get new ground truth control reading
                    added_frames[-1]['control'] = sync_mode.vehicle.get_control()
                

                # only collect data from every 5th frame
                if frame_count % args.skip_frames != 0:
                    frame_count += 1
                    continue

                # if the car is stopped (like at a stop light) do not collect data
                # for the entire duration. collect 3 frames if the car is stopped.
                # also, collect frames if the car is stopped but is getting throttle
                # values that are greater than 1. we want the agent to learn to
                # start from a stopped position
                vel = sqrt(sync_mode.vehicle.get_velocity().x**2 + sync_mode.vehicle.get_velocity().y**2)
                if vel > .001:
                    frames_stopped = 0
                if vel < .001:
                    if sync_mode.vehicle.get_control().throttle == 0 and frames_stopped >= 3:
                        frame_count += 1
                        continue
                    else:
                        frames_stopped += 1
                
                # define dictionary for saving the frame data
                frame = dict()
                frame['img_names'] = []
                for idx, data in enumerate(sensor_data):

                    # save image
                    sensor_type = sensor_types[idx % len(sensor_types)].split('.')[-1]
                    train_img_name = f'{sensor_type}_{prev_num_frames+frames_added}.png'
                    img = save_carla_sensor_img(data,
                                      train_img_name,
                                      DATASET_PATH)
                    
                    # filter out images with extreme occlusions
                    # if sensor_type == 'depth' and data_type == 'from':

                    #     if not proximity_filter(img):
                    #         save_labels = False
                    #         break
                    
                    # get img information to save to labels dictionary

                    frame['img_names'].append(train_img_name)

                frame['vel'] = vel

                # add frame dict to label dict
                added_frames.append(frame)
                
                frame_count += 1
                frames_added += 1

                pbar.update(1)
                
            # remove the last dictionary in the 'added frames' list in case
            # it does not yet have a 'control' data point since this is added
            # in the frame after the rest of the data was collected
            added_frames.pop()

            labels['data'] = added_frames
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
            labels['data'] = added_frames
            if prev_label_data:
                labels['data'] += prev_label_data
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
        '--skip_frames',
        action='store',
        default=5,
        type=int,
        help='Specify the frame interval with which to collect the data.')
    
    args = parser.parse_args()
    
    try:
        if args.verify_dataset:
            verify_dataset()
        else:
            main()
        
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')