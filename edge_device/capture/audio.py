import datetime
import numpy as np
import sounddevice as sd

fs = 16000  # sample rate (Hz)
duration = 1.92  # seconds, multiple of 0.96 (length of the sliding window)
samples = int(duration * fs)
recording = np.zeros((0, 1))  # initialize recording shape


class MicrophoneAudioStream:
    def __init__(self, src, in_q):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = sd.InputStream(samplerate=fs, channels=1, callback=self.update)

        self.in_q = in_q

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()

        # start the thread to read frames from the video stream, and the thread to process the frames
        self.stream.start()

        return self

    def update(self, indata, frames, time, status):
        global recording

        recording = np.concatenate((recording, indata), axis=0)

        if recording.size >= samples:
            item = {'type': 'audio', 'data': recording[:samples]}
            self.in_q.put(item)
            recording = recording[samples:]

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

        self.stream.stop()

        # indicate that the thread should be stopped
        self.stopped = True

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()