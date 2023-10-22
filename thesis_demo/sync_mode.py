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

class CarlaSyncMode(object):
    """Class for running Carla in synchronous mode. Allows frame rate sync to
    ensure no data is lost and all backend code runs and is completed before
    'ticking' to the next frame."""
    
    def __init__(self, world, **kwargs):
        self.world = world
        self.sensors = []
        self.sensor_types = kwargs.get('sensor_types', None)
        self.vehicles = []
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None
        self.blueprint_library = world.get_blueprint_library()
        
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
    
    # def initialize_vehicles(self):
    #     """Create and spawn all vehicle actors.
    #     """
        
    #     # this is a dumb way to initialize, but all of the vehicles will initially
    #     # spawn way up in the sky to avoid collisions. they will be randomly 
    #     # moved in the next frame, so this won't matter
    #     for i in range(self.num_vehicles):
    #         # get a random vehicle blueprint
    #         vehicle_bp = random.choice(blueprint_library.filter('vehicle.*.*'))
            
    #         transform = carla.Transform(carla.Location(x=i*3, y=0, z=100))
    #         actor = world.spawn_actor(vehicle_bp, transform)
            
    #         self.vehicles.append(actor)
    
    def initialize(self):
        """Create all vehicles and spawn all sensors."""

        # create ground vehicle actor
        ground_vehicle_bp = self.blueprint_library('tesla')
        transform = carla.Transforms(x=123, y=123, z=123)
        self.ground_vehicle_actor = self.world.spawn_actor(ground_vehicle_bp, 
                                                           transform)
        
        depth_sensor_bp = self.blueprint_library('depth_sensor')
        rgb_sensor_bp = self.blueprint_library('rgb_sensor')

        # attach sensors to ground vehicle actor
        self.ground_vehicle_depth_sensor = self.world.spawn_actor(depth_sensor_bp,
                                                                  attach_to=self.ground_vehicle_actor)
        self.ground_vehicle_rgb_sensor = self.world.spawn_actor(rgb_sensor_bp,
                                                                attach_to=self.ground_vehicle_actor)
        
        # add ground vehicle sensors to the queues
        self.make_queue(self.ground_vehicle_depth_sensor.listen)
        self.make_queue(self.ground_vehicle_rgb_sensor.listen)

        # create drone object
        self.drone = self.create_drone_actor(self.sensor_types)

    def create_drone_actor():
        pass
     
    def __exit__(self, *args, **kwargs):
        # make sure to clean up the memory
        self.world.apply_settings(self._settings)
        for sensor in self.sensors:
            sensor.destroy()
            
        for vehicle_actor in self.vehicles:
            vehicle_actor.destroy()
        
    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data       