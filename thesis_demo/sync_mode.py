class CarlaSyncMode(object):
    """Class for running Carla in synchronous mode. Allows frame rate sync to
    ensure no data is lost and all backend code runs and is completed before
    'ticking' to the next frame."""
    
    def __init__(self, world, **kwargs):
        self.world = world
        self.sensors = []
        self.vehicles = []
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self.num_vehicles = kwargs.get('num_vehicles', 15)
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
            
            # z rel cannot be lower than 0.5 globally
            z = max(.7, z)
            
            # ensure that the global location is within the init limits..
            # truncate to the max/min global location if needed
            x = max(x, x_lim_init[0])
            x = min(x, x_lim_init[1])
            y = max(y, y_lim_init[0])
            y = min(y, y_lim_init[1])
            # we want the z to be able to sink below the init sensor spawn
            # location as the ground view will always be lower than the aeriel
            # drone view
            # z = max(z, z_lim_init[0])
            # z = min(z, z_lim_init[1])
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
            
        for vehicle_actor in self.vehicles:
            vehicle_actor.destroy()
        
    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data       