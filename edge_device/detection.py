import cv2
import argparse
import multiprocessing

from inference.audio import AudioInference
from inference.video import VideoInference
from capture.audio import MicrophoneAudioStream
from capture.video import WebcamVideoStream
from multiprocessing import Queue, Pool

from openhab import OpenHAB

base_url = 'http://192.168.1.67:8080/rest'
openhab = OpenHAB(base_url)

item_label = openhab.get_item('Input')
item_label_audio = openhab.get_item('Input2')


def inference_worker(in_q):

    audio = AudioInference(output_q, item_label_audio)
    video = VideoInference(output_q, item_label)
    inference = {'audio': audio.inference, 'video': video.inference}

    while True:
        item = in_q.get()
        inference[item['type']](item['data'])


def data_processing_worker(in_q, out_q):

    while True:
        item = in_q.get()
        out_q.put(item)


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

    output_q = Queue(maxsize=args.queue_size)
    data_captured_q = Queue(maxsize=args.queue_size)
    data_processed_q = Queue(maxsize=args.queue_size)
    prediction_q = Queue(maxsize=args.queue_size)

    pool = Pool(args.num_workers, inference_worker, (data_processed_q,))
    processing_pool = Pool(args.num_workers, data_processing_worker, (data_captured_q, data_processed_q))

    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height,
                                      in_q=data_captured_q,
                                      out_q=output_q).start()

    audio_capture = MicrophoneAudioStream(src=0, in_q=data_captured_q).start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        pass

    pool.terminate()
    processing_pool.terminate()
    audio_capture.stop()
    video_capture.stop()
    cv2.destroyAllWindows()

    print('[INFO] elapsed time (total): {:.2f}'.format(video_capture.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(video_capture.fps()))
