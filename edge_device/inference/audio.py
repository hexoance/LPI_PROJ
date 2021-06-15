import numpy as np
from tflite_runtime.interpreter import Interpreter
from pydub import AudioSegment
import csv


class AudioInference:

    def __init__(self, audio_model):
        self.fs = audio_model['frequency']  # sample rate (Hz)
        duration = audio_model['duration']  # seconds, ex. multiple of 0.96 for yamnet (length of the sliding window)
        self.samples = int(duration * self.fs)
        self.model_name = audio_model['name']
        self.threshold = audio_model['threshold']  # threshold from 0 to 1, ex. 0.85

        # Load Model
        self.interpreter = Interpreter(f'models/{self.model_name}.tflite')
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        self.waveform_input_index = inputs[0]['index']
        self.scores_output_index = outputs[0]['index']

        # Read the csv file containing the model classes
        class_map_path = f'models/{self.model_name}_class_map.csv'
        with open(class_map_path) as class_map_csv:
            self.class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
        self.class_names = self.class_names[1:]  # Skip CSV header

    def remove_middle_silence(self, sound):
        silence_threshold = -45.0  # dB
        chunk_size = 100  # ms
        sound_ms = 0  # ms
        trimmed_sound = AudioSegment.empty()

        while sound_ms < len(sound):
            if sound[sound_ms:sound_ms + chunk_size].dBFS >= silence_threshold:
                trimmed_sound += sound[sound_ms:sound_ms + chunk_size]
            sound_ms += chunk_size

        return trimmed_sound.set_sample_width(2)

    def inference(self, waveform):
        waveform.shape = (self.samples,)
        waveform = waveform.astype('float32')

        # audio = AudioSegment.from_wav('tmp.wav')
        # audio = self.remove_middle_silence(audio)
        # audio.export(filename, format="wav")

        self.interpreter.resize_tensor_input(self.waveform_input_index, [len(waveform)], strict=True)
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.waveform_input_index, waveform)
        self.interpreter.invoke()
        scores = self.interpreter.get_tensor(self.scores_output_index)

        # compute softmax activations
        if self.model_name == 'yamnet':
            class_probabilities = np.mean(scores, axis=0)  # yamnet non-retrained model uses different activations
        else:
            class_probabilities = np.exp(scores) / np.sum(np.exp(scores), axis=-1)  # simulate tf softmax function

        top_class = np.argmax(class_probabilities)
        top_score = class_probabilities[top_class]
        inferred_class = self.class_names[top_class]

        if top_score < self.threshold:
            inferred_class = 'Unknown'

        print(f'[AUDIO - \'{self.model_name}\'] {inferred_class} ({top_score})')

        return inferred_class
