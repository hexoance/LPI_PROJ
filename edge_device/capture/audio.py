import numpy as np
import sounddevice as sd

fs = 16000  # sample rate (Hz)
duration = 0.96  # seconds, multiple of 0.96 (length of the sliding window)
samples = int(duration * fs)


class MicrophoneAudioStream:
    def __init__(self, src, in_q):
        # initialize the audio input stream
        self.stream = sd.InputStream(samplerate=fs, channels=1, callback=self.update)
        self.in_q = in_q
        self.recording = np.zeros((0, 1))

    def start(self):
        # start the stream to read data from the microphone
        self.stream.start()

        return self

    def update(self, indata, frames, time, status):
        self.recording = np.concatenate((self.recording, indata), axis=0)

        if self.recording.size >= samples:
            item = {"type": "audio", 'data': self.recording[:samples]}
            self.in_q.put(item)
            self.recording = self.recording[samples:]

    def stop(self):
        self.stream.stop()
