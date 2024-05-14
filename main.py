import argparse
import yaml
import logging
import pprint

from trainer import main as app_main

# Setup the argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='params-ijepa.yaml')
parser.add_argument(
    '--device', type=str, default='cuda:0',
    help='which device to use')

# Main function to load parameters and run the app
def main(args):
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Set the device environment (useful if the framework checks for this environment variable)
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[-1]

    try:
        # Load script params
        with open(args.fname, 'r') as y_file:
            params = yaml.load(y_file, Loader=yaml.FullLoader)
            logger.info('Loaded params:')
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(params)

        # Run the application main function
        app_main(args=params)

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)










"""
import argparse

import multiprocessing as mp

import pprint
import yaml

from utils.distributed import init_distributed
from trainer import main as app_main
"""








"""
parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='params-ijepa.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')


def process_main(rank, fname, world_size, devices):
    import os
    import logging

    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    if rank == 0:
        logger.setLevel(logging.INFO)
    
    try:
        # Load script params
        with open(fname, 'r') as y_file:
            params = yaml.load(y_file, Loader=yaml.FullLoader)
            logger.info('Loaded params...')
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(params)
        
        world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
        logger.info(f'Running... (rank: {rank}/{world_size})')
        app_main(args=params)

    except Exception as e:
        logger.error(f"An error occurred in process {rank}: {e}", exc_info=True)
        print(f"An error occurred in process {rank}: {e}")



if __name__ == '__main__':
    args = parser.parse_args()

    num_gpus = len(args.devices)
    mp.set_start_method('spawn', force=True)
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(target=process_main, args=(rank, args.fname, num_gpus, args.devices))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
"""