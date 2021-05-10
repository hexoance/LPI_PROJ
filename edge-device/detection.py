# code borrowed from datitran/object_detector_app

import cv2
import time
import argparse
import multiprocessing
import numpy as np
import sounddevice as sd

from audio import AudioInference
from video import VideoInference
from utils.app_utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool

from openhab import OpenHAB

base_url = 'http://localhost:8080/rest'
openhab = OpenHAB(base_url)

item_label = openhab.get_item('Input')
item_label_audio = openhab.get_item('Input2')

fs = 16000  # sample rate (Hz)
duration = 1.92  # seconds, multiple of 0.96 (length of the sliding window)
samples = int(duration * fs)
recording = np.zeros((0, 1))  # initialize recording shape


def worker(input_q, output_q):

    fps = FPS().start()

    audio = AudioInference(item_label_audio)
    video = VideoInference(output_q, item_label)
    while True:
        fps.update()
        item = input_q.get()
        if (item['type'] == 'audio'):
            audio.inference(item['data'])
        elif (item['type'] == 'video'):
            video.inference(item['data'])

    fps.stop()
    sess.close()


def callback(indata, frames, time, status):
    global recording

    recording = np.concatenate((recording, indata), axis=0)

    if recording.size >= samples:
        item = {'type': 'audio', 'data': recording[:samples]}
        input_q.put(item)
        recording = recording[samples:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=4, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q))

    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()
    fps = FPS().start()
    with sd.InputStream(samplerate=fs, channels=1, callback=callback):

        while True:  # fps._numFrames < 120
            frame = video_capture.read()
            item = {'type': 'video', 'data': frame}
            input_q.put(item)

            t = time.time()

            output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
            cv2.imshow('Video', output_rgb)
            fps.update()

            print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()