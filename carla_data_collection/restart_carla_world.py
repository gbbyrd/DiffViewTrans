import carla
import argparse

argparser = argparse.ArgumentParser()

argparser.add_argument(
    '--world',
    action='store',
    default=False,
    help='Specify world to load')

args = argparser.parse_args()

client = carla.Client('localhost', 2000)
world = client.load_world(args.world)