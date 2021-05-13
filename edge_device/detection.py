import argparse
import datetime

from inference.audio_yamnet import AudioInference
from inference.audio_yamnet_retrained import AudioRetrainedInference
from inference.video import VideoInference
from capture.audio import MicrophoneAudioStream
from capture.video import WebcamVideoStream
from multiprocessing import Queue, Pool
from edge_device.const import MODEL_THRESHOLD, MODEL_DIFERENCE, MODEL_UNKOWN

from openhab import OpenHAB

base_url = 'http://localhost:8080/rest'
openhab = OpenHAB(base_url)

item_label_data = openhab.get_item('EdgeDeviceData')

models = {
    'audio': [
        {'name': 'yamnet_1', 'duration': 0.96, 'frequency': 16000, 'model': AudioInference},
        {'name': 'new_yamnet', 'duration': 0.96, 'frequency': 16000, 'model': AudioRetrainedInference}
    ],
    'video': [
        {'name': 'charades', 'model': VideoInference}
    ]
}


def data_processing_worker(in_q, out_q):
    while True:
        item = in_q.get()
        for model in models[item['type']]:
            new_item = {'type': item['type'], 'name': model['name'], 'data': item['data'][:]}
            out_q.put(new_item)


def inference_worker(in_q, out_q):
    local_models = {}
    for audio_model in models['audio']:
        model = audio_model['model'](audio_model)
        local_models[audio_model['name']] = model

    for video_model in models['video']:
        model = video_model['model']()
        local_models[video_model['name']] = model

    while True:
        item = in_q.get()
        prediction = local_models[item['name']].inference(item['data'])
        if prediction != MODEL_UNKOWN:
            out_q.put({'prediction': prediction, 'type': item['type'], 'timestamp': str(datetime.datetime.now())})


def network_worker(in_q):
    while True:
        item = in_q.get()
        item_label_data.state = str(item)


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

    # logger = multiprocessing.log_to_stderr()
    # logger.setLevel(multiprocessing.SUBDEBUG)

    data_captured_q = Queue(maxsize=args.queue_size)
    data_processed_q = Queue(maxsize=args.queue_size)
    prediction_q = Queue(maxsize=args.queue_size)

    processing_pool = Pool(2, data_processing_worker, (data_captured_q, data_processed_q))
    inference_pool = Pool(args.num_workers, inference_worker, (data_processed_q, prediction_q))
    network_pool = Pool(2, network_worker, (prediction_q,))

    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height,
                                      in_q=data_captured_q)

    if video_capture.found:
        video_capture.start()

    audio_capture = MicrophoneAudioStream(src=0, in_q=data_captured_q).start()

    # start the timer
    start = datetime.datetime.now()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        pass

    # stop the timer
    end = datetime.datetime.now()

    network_pool.terminate()
    inference_pool.terminate()
    processing_pool.terminate()

    audio_capture.stop()
    video_capture.stop()

    print('[INFO] elapsed time (seconds): {:.2f}'.format((end - start).total_seconds()))
