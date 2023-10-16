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

NUM_SENSORS_Y = 10
NUM_SENSORS_X = 5
NUM_SENSORS_Z = 3
SENSOR_MAX_DIST = 5
IM_WIDTH = 128
IM_HEIGHT = 128
DATASET_PATH = 'C:\\Users\\gbbyrd\\Desktop\\thesis\\data\\experiment3_large'

class CarlaSyncMode(object):
    
    def __init__(self, world, sensors, vehicle, **kwargs):
        self.world = world
        self.sensors = sensors
        self.vehicle = vehicle
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None
        
        # sensor blueprint
        blueprint_library = world.get_blueprint_library()
        self.cam_bp = blueprint_library.find("sensor.camera.rgb")
        self.cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.cam_bp.set_attribute("fov", "110")
    
    # allows the class to be used with: with ____ as ____:    
    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode = False,
            synchronous_mode = True, 
            fixed_delta_seconds = self.delta_seconds
        ))
        
        # create queues for each sensor that store the data as the sensor reads it
        self.make_queue(self.world.on_tick)
        for sensor in self.sensors:
            self.make_queue(sensor.listen)
        return self
    
    def make_queue(self, register_event):
        # create a q for the event to register data to
        q = queue.Queue()
        # define q.put as the function that is called when the event recieves data
        register_event(q.put)
        # add q to the list of _queues
        self._queues.append(q)
    
    # call this function to step once through the simulation
    def tick(self, timeout):
        # get the next frame from the world.. this should automatically update the sensor queues as well
        self.frame = self.world.tick()
        
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data
    
    def randomize_sensors(self, sensor_limits):
        """Delete all current sensors and respawn specified sensors at random
        orientations and location to the vehicle. Spawns an initial sensor within
        a certain distance from the ground truth image via the sensor_limits dictionary.
        Spawns all subsequent sensors within these limits from both the initial
        sensor and the ground truth image. Therefore, all images can be paired
        with each other during training to improve the dataset diversity. 
        
        Args:
            sensor_limits = {
                'num_sensors': int,
                'x_limits': [float, float],
                'y_limits': [float, float],
                'z_limits': [float, float],
                'z_angle_limits': [float, float]
            }
        
        Returns:
            None
        """
        
        # define ground truth sensor info
        X = 2.5
        Y = 0
        Z = 0.7
        Z_ANGLE = 0
        
        # get initial limits
        num_sensors = sensor_limits['num_sensors']
        x_lim = sensor_limits['x_limits']
        y_lim = sensor_limits['y_limits']
        z_lim = sensor_limits['z_limits']
        z_angle_lim = sensor_limits['z_angle_limits']
        
        assert num_sensors > 1, "Error: The specified number of sensors must be greater than 1."
        
        # delete all previous sensors and clear the array
        for sensor in self.sensors:
            sensor.destroy()
        
        # clear previous sensors and sensor information
        self.sensors = []
        self.sensor_info = []
        
        # clear out the self._queues except for the world queue (index 0)
        self._queues = self._queues[:1]
        
        # spawn one initial sensor, then spawn the rest of the sensors off of
        # the initial sensor. This is done to ensure that the maximum translation
        # distance is not exceeded.
        
        # the initial sensor is spawned anywere in the original possible vicinity
        x = random.uniform(x_lim[0], x_lim[1])
        y = random.uniform(y_lim[0], y_lim[1])
        z = random.uniform(z_lim[0], z_lim[1])
        
        # the sensor z angle will change based off the y location to ensure that
        # the ground truth features will stay in frame somewhat
        z_angle_lim_0 = [0, 0]
        if y < 0:
            z_angle_lim_0[0] = (z_angle_lim[0]
                                + y/abs(y_lim[0])*z_angle_lim[0])
            z_angle_lim_0[1] = z_angle_lim[1]
        else:
            z_angle_lim_0[0] = z_angle_lim[0]
            z_angle_lim_0[1] = (z_angle_lim[1]
                                - y/(y_lim[1])*z_angle_lim[1])
        
        z_angle = random.uniform(z_angle_lim_0[0], z_angle_lim_0[1])
        
        # spawn the initial sensor and add to relevant lists
        spawn_point = carla.Transform(carla.Location(x=X+x, y=Y+y, z=Z+z),
                                          carla.Rotation(yaw=Z_ANGLE+z_angle))
        sensor = self.world.spawn_actor(self.cam_bp, spawn_point, attach_to=self.vehicle)
        self.sensors.append(sensor)
        info = {
            'x': X+x,
            'y': Y+y,
            'z': Z+z,
            'z_angle': Z_ANGLE+z_angle
        }
        self.sensor_info.append(info)
        self.make_queue(sensor.listen)
        
        # the following sensors must:
        # 1. be spawned within the initial vicinity
        # 2. be spawned within a smaller, translation vicinity
        #   this is because we can use the translation from the training images
        #   to other training images, but we do not want to, for example,
        #   translate all the way from the very left side of the initial vicinity to the very
        #   right side of the initial vicinity. Therefore, we limit the spawning
        #   of the subsequent sensors to be at least as close to the initial sensor
        #   as the furthest point from the ground truth image the initial sensor 
        #   could spawn in the initial vicinity
        
        # get the maximum possible distances from the original vicinity
        max_x_dist = max(abs(x_lim[0]), abs(x_lim[1]))
        max_y_dist = max(abs(y_lim[0]), abs(y_lim[1]))
        max_z_dist = max(abs(z_lim[0]), abs(z_lim[1]))
        max_z_angle_dist = max(abs(z_angle_lim[0]), abs(z_angle_lim[1]))
        
        # the max distances must be divided by two to ensure that all subsequently
        # spawned sensors are within the required distances of each other
        max_x_dist = max_x_dist / 2
        max_y_dist = max_y_dist / 2
        max_z_dist = max_z_dist / 2
        max_z_angle_dist = max_z_angle_dist / 2
        
        # get the new limits for subsequent sensors
        x_lim_new = [max(x_lim[0], x-max_x_dist), min(x_lim[1], x+max_x_dist)]
        y_lim_new = [max(y_lim[0], y-max_y_dist), min(y_lim[1], y+max_y_dist)]
        z_lim_new = [max(z_lim[0], z-max_y_dist), min(z_lim[1], z+max_z_dist)]
    
        z_angle_lim_new = [0, 0]
    
        # spawn the rest of the sensors
        for i in range(sensor_limits['num_sensors']-1):
            x = random.uniform(x_lim_new[0], x_lim_new[1])
            y = random.uniform(y_lim_new[0], y_lim_new[1])
            z = random.uniform(z_lim_new[0], z_lim_new[1])
 
            # calculate the z_angle based off of the y dimension
            if y < 0:
                z_angle_lim_new[0] = (z_angle_lim[0]
                                  + y/abs(y_lim[0])*z_angle_lim[0])
                z_angle_lim_new[1] = z_angle_lim[1]
            else:
                z_angle_lim_new[0] = z_angle_lim[0]
                z_angle_lim_new[1] = (z_angle_lim[1]
                                  - y/(y_lim[1])*z_angle_lim[1])
            
            z_angle_lim_new[0] = max(z_angle_lim_new[0], z_angle-max_z_angle_dist)
            z_angle_lim_new[1] = min(z_angle_lim_new[1], z_angle+max_z_angle_dist)
            
            z_angle = random.uniform(z_angle_lim_new[0], z_angle_lim_new[1])
            
            spawn_point = carla.Transform(carla.Location(x=X+x, y=Y+y, z=Z+z),
                                          carla.Rotation(yaw=Z_ANGLE+z_angle))
            sensor = self.world.spawn_actor(self.cam_bp, spawn_point, attach_to=self.vehicle)
            self.sensors.append(sensor)
            info = {
            'x': X+x,
            'y': Y+y,
            'z': Z+z,
            'z_angle': Z_ANGLE+z_angle
        }
            self.sensor_info.append(info)
            self.make_queue(sensor.listen)
            
        # finally, spawn the ground truth sensor
        spawn_point = carla.Transform(carla.Location(x=X, y=Y, z=Z),
                                      carla.Rotation(yaw=Z_ANGLE))
        sensor = self.world.spawn_actor(self.cam_bp, spawn_point, attach_to=self.vehicle)
        self.sensors.append(sensor)
        info = {
            'x': X+x,
            'y': Y+y,
            'z': Z+z,
            'z_angle': Z_ANGLE+z_angle
        }
        self.sensor_info.append(info)
        self.make_queue(sensor.listen)
    
    # allows the class to be used with: with ____ as ____: 
    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)
        for sensor in self.sensors:
            sensor.destroy()
        
    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data       

def process_img(image, img_name, base_path):
    """Processes a carla sensor.Image object and a velocity int

    Args:
        obs (tuple): (Carla image object, int)

    Returns:
        obs (tuple): (np.array(rbg array image), int(vel value))
    """
    # image, velocity = obs
    
    i = np.array(image.raw_data) # the raw data is initially just a 1 x n array
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4)) # raw data is in rgba format
    i3 = i2[:, :, :3] # we just want the rgb data
    
    # save the image
    full_save_path = os.path.join(base_path,
                                    img_name)
    cv2.imwrite(full_save_path, i3)
    return i3

def display_camera_sensor(image):
    i = np.array(image.raw_data) # the raw data is initially just a 1 x n array
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4)) # raw data is in rgba format
    i3 = i2[:, :, :3] # we just want the rgb data
    
    cv2.imshow('img', i3)
    cv2.waitKey(1)
    return i3
    
def experiment3(sync_mode, 
                clock, 
                sensor_info,
                ground_truth_count,
                train_img_count):
    
    clock.tick()

    world_data  = sync_mode.tick(timeout=2.0)
    snapshot = world_data[0]
    sensor_data = world_data[1:]
    sim_fps = round(1.0 / snapshot.timestamp.delta_seconds)
    true_fps = clock.get_fps()
    
    # process the data from the sensors
    sensor_group = dict()
    
    # ground truth sensor first
    ground_truth_name = f'ground_truth_{str(ground_truth_count).zfill(6)}.png'
    ground_truth_img = process_img(sensor_data[-1],
                                    ground_truth_name, 
                                    DATASET_PATH)
    ground_truth_count += 1
    
    sensor_group['ground_truth_img_name'] = ground_truth_name
    
    train_img_count += 1
    new_labels = []
    for idx, data in enumerate(sensor_data[:-1]):
        name = f'train_img_{str(train_img_count).zfill(6)}.png'
        img = process_img(sensor_data[idx],
                            name,
                            DATASET_PATH)
        sensor_group[f'train_img_{idx}_info'] = {
            'img_name': name,
            'location': sync_mode.sensor_info[idx]
        }
        train_img_count += 1

    count = ground_truth_count
    
    sync_mode.randomize_sensors(sensor_info)

    if count == 100:
        # return True to signal a stop to the syncmode loop
        return True, ground_truth_count, train_img_count, [sensor_group]
    
    # return False to signal a continue to the syncmode loop
    return False, ground_truth_count, train_img_count, [sensor_group]

def main():
    
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
        
    actor_list = []

    try:
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(2.0)
        
        world = client.get_world()
        
        blueprint_library = world.get_blueprint_library()
        
        # get vehicle blueprint
        bp = blueprint_library.filter("model3")[0]
        print(bp)
        
        # spawn vehicle
        # spawn_point = random.choice(world.get_map().get_spawn_points()) # get a random spawn point from those in the map
        spawn_points = world.get_map().get_spawn_points()    
        spawn_point1 = spawn_points[0]
        vehicle1 = world.spawn_actor(bp, spawn_point1) # spawn car at the random spawn point
        actor_list.append(vehicle1)
        
        clock = pygame.time.Clock()
        
        sensor_list = []
        
        with CarlaSyncMode(world, sensor_list, vehicle1, fps=30) as sync_mode:
            count = 0
            
            ground_truth_count = 0
            train_img_count = 0
            label_dictionary = dict()
            label_list = []
            
            isStop = False
            
            # sensor_info = {
            #     'num_sensors': 5,
            #     'x_limits': [0, 10],
            #     'y_limits': [-7, 7],
            #     'z_limits': [0, 5],
            #     'z_angle_limits': [-30, 30]
            # }
            
            sensor_info = {
                'num_sensors': 5,
                'x_limits': [0, 0],
                'y_limits': [-.01, 10],
                'z_limits': [-.01, .01],
                'z_angle_limits': [-30, 30]
            }
            
            # initialize random sensors
            sync_mode.randomize_sensors(sensor_info)
            
            sensor_group_list = []
            
            # turn on autopilot
            vehicle1.set_autopilot(True)
            
            while not isStop:
                isStop, ground_truth_count, train_img_count, sensor_group = experiment3(sync_mode, clock, sensor_info,
                                     ground_truth_count, train_img_count)
                sensor_group_list += sensor_group
            
            label_dictionary['data'] = sensor_group_list
            with open(DATASET_PATH+'\\labels.json', 'w') as file:
                json.dump(label_dictionary, file)
                
        # create a synchronous mode context
        time.sleep(1)
        
    finally:
        for actor in actor_list:
            actor.destroy()
            
        print("All cleaned up!")
        
if __name__=='__main__':
    
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')