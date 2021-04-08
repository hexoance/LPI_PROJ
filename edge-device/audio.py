import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import soundfile as sf
import io
import csv

fs = 16000  # sample rate (Hz)
duration = 1.92  # seconds, multiple of 0.96 (length of the sliding window)
samples = int(duration * fs)
recording = np.zeros((0, 1))  # initialize recording shape

#DATASETS_PATH = './datasets/'
DATASETS_PATH ='D:/datasets/'

datasets = ['ESC-50-master', 'FSD50k']
DATASET_NAME = datasets[1]

classes = []

def readClasses(file):
    with open(DATASETS_PATH + DATASET_NAME + '/' + file, newline='') as f:
        reader = csv.reader(f)
        iterreader = iter(reader)
        next(iterreader)
        for row in iterreader:
            classes.append(row[0])

        #print(classes)


readClasses('classes.csv')


class AudioInference:

    def __init__(self, item_label):
        self.item_label = item_label
        # Load Models
        self.model_audio = tf.saved_model.load('../models/new_yamnet')

        # Find the name of the class with the top score when mean-aggregated across frames.
        # class_map_path = self.model_audio.class_map_path().numpy()
        # class_map_csv = io.StringIO(tf.io.read_file(class_map_path).numpy().decode('utf-8'))
        # self.class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
        # self.class_names = self.class_names[1:]  # Skip CSV header

    def inference2(self, waveform):
        # Reshape numpy array
        waveform.shape = (samples,)

        # Run the model.
        scores, embeddings, log_mel_spectrogram = self.model_audio(waveform)

        # Scores is a matrix of (time_frames, num_classes) classifier scores.
        # Average them along time to get an overall classifier output for the clip.
        prediction = np.mean(scores, axis=0)

        # Report the highest-scoring classes and their scores.
        top5 = np.argsort(prediction)[::-1][:5]
        print('[AUDIO] Scores:\n' + '\n'.join(
            '  {:12s}: {:.3f}'.format(self.class_names[i], prediction[i]) for i in top5))

        self.item_label.state = self.class_names[top5[0]]
        # plot_results(waveform, scores, log_mel_spectrogram)
        # print(class_names[scores.numpy().mean(axis=0).argmax()])  # Prints top score.

    def inference(self, waveform):
        filename = 'tmp.wav'
        sf.write(filename, waveform, fs)

        """ read in a waveform file and convert to 16 kHz mono """
        file_contents = tf.io.read_file(filename)
        wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)

        results = self.model_audio(wav)
        your_top_class = tf.argmax(results)
        your_infered_class = classes[your_top_class]
        class_probabilities = tf.nn.softmax(results, axis=-1)
        your_top_score = class_probabilities[your_top_class]
        print(f'[Your model] The main sound is: {your_infered_class} ({your_top_score})')
