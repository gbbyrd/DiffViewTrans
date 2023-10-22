import numpy as np
import cv2
import os

IM_HEIGHT = 256
IM_WIDTH = 256

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

def get_carla_sensor_img(image):
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