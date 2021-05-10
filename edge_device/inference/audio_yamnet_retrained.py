import tensorflow as tf
import soundfile as sf
import tensorflow_io as tfio
from tflite_runtime.interpreter import Interpreter
from pydub import AudioSegment
import io
import csv
from edge_device.const import MODEL_THRESHOLD, MODEL_DIFERENCE, MODEL_UNKOWN


class AudioRetrainedInference:

    def __init__(self, audio_model):
        self.fs = audio_model['frequency']  # sample rate (Hz)
        duration = audio_model['duration']  # seconds, ex. multiple of 0.96 for yamnet (length of the sliding window)
        self.samples = int(duration * self.fs)

        # Load Model
        self.interpreter = Interpreter('../models-tflite/' + audio_model['name'] + '.tflite')
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        self.waveform_input_index = inputs[0]['index']
        self.scores_output_index = outputs[0]['index']

        # Find the name of the class with the top score when mean-aggregated across frames.
        class_map_path = '../models/' + audio_model['name'] + '/assets/yamnet_class_map.csv'
        class_map_csv = io.StringIO(tf.io.read_file(class_map_path).numpy().decode('utf-8'))
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
        filename = 'tmp.wav'
        sf.write(filename, waveform, self.fs)

        # audio = AudioSegment.from_wav(filename)
        # audio = self.remove_middle_silence(audio)
        # audio.export(filename, format="wav")

        """ read in a waveform file and convert to 16 kHz mono """
        file_contents = tf.io.read_file(filename)
        wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)

        self.interpreter.resize_tensor_input(self.waveform_input_index, [len(wav)], strict=True)
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.waveform_input_index, wav)
        self.interpreter.invoke()
        results = self.interpreter.get_tensor(self.scores_output_index)

        your_top_class = tf.argmax(results)
        your_infered_class = self.class_names[your_top_class]
        class_probabilities = tf.nn.softmax(results, axis=-1)

        your_top_score = class_probabilities[your_top_class]
        second_top_score = class_probabilities[tf.nn.top_k(results, k=2).indices.numpy()[1]]

        if your_top_score - second_top_score <= MODEL_DIFERENCE or your_top_score < MODEL_THRESHOLD:
            your_infered_class = MODEL_UNKOWN

        print(f'[AUDIO - DomesticSounds] {your_infered_class} ({your_top_score})')

        return your_infered_class
