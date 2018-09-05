import argparse
import logging
import sys
import random
import time
from pathlib import Path, PosixPath
from queue import Queue
from subprocess import call
from threading import Thread, current_thread

SRC_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_PATH))
from src.utils import func_profile, log_handler

LOGGER = logging.getLogger(__name__)
DEMO_BASH = str(SRC_PATH / 'demo.bash')

def argparser():
    parser = argparse.ArgumentParser(
        description='recursively process the pipeline in given folder')
    parser.add_argument('-g', '--gpus', dest='gpus', help='1,2,...')
    parser.add_argument('-r', '--recursive', dest='root', help='folders with multi-video')
    parser.add_argument('-o', '--option', dest='option', \
                        help='optional argument pass to demo.bash')
    return parser

def demo_one_video(gpu_queue: Queue, video_dirpath: PosixPath, action: str = ''):
    # process and gpu config
    thread_info = current_thread()
    gpuid = None
    while True:
        if not gpu_queue.empty():
            gpuid = gpu_queue.get()
            break
        time.sleep(random.randint(1, 10))

    # video infomation
    video_name = video_dirpath.name
    video_path = video_dirpath / '{}.avi'.format(video_name)
    LOGGER.info('{} {} -> GPU {} for {}'.format('='*10, thread_info, gpuid, video_name))
    call([DEMO_BASH, gpuid, str(video_path), action])
    gpu_queue.put(gpuid)
    LOGGER.info('Complete {}'.format(thread_info))

@func_profile
def main(args):
    log_handler(LOGGER)
    LOGGER.info(args)
    gpu_candidate = args.gpus.split(',')

    # prepare arguments
    root_path = Path(args.root)
    root_folders = [i for i in root_path.iterdir() if i.is_dir()]
    gpu_queue = Queue()
    for i in gpu_candidate:
        gpu_queue.put(i)

    # multithreading
    th_jobs = []
    all_th_args = zip([gpu_queue]*len(root_folders), \
                      root_folders, \
                      [args.option]*len(root_folders))
    for th_args in all_th_args:
        job = Thread(target=demo_one_video, args=th_args)
        th_jobs.append(job)
        job.start()
    for j in th_jobs:
        j.join()
    LOGGER.info('Complete multithread processing with pipeline ')

if __name__ == '__main__':
    main(argparser().parse_args())
