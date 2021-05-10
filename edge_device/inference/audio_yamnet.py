import numpy as np
import tensorflow as tf
import io
import csv
from edge_device.const import MODEL_THRESHOLD, MODEL_DIFERENCE, MODEL_UNKOWN

class AudioInference:

    def __init__(self, audio_model):
        fs = audio_model['frequency']  # sample rate (Hz)
        duration = audio_model['duration']  # seconds, ex. multiple of 0.96 for yamnet (length of the sliding window)
        self.samples = int(duration * fs)

        # Load Models
        self.model_audio = tf.saved_model.load('../models/' + audio_model['name'])

        # Find the name of the class with the top score when mean-aggregated across frames.
        class_map_path = self.model_audio.class_map_path().numpy()
        class_map_csv = io.StringIO(tf.io.read_file(class_map_path).numpy().decode('utf-8'))
        self.class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
        self.class_names = self.class_names[1:]  # Skip CSV header

    def inference(self, waveform):
        # Reshape numpy array
        waveform.shape = (self.samples,)

        # Run the model.
        scores, embeddings, log_mel_spectrogram = self.model_audio(waveform)

        # Scores is a matrix of (time_frames, num_classes) classifier scores.
        # Average them along time to get an overall classifier output for the clip.
        prediction = np.mean(scores, axis=0)

        # Report the highest-scoring classes and their scores.
        top5 = np.argsort(prediction)[::-1][:5]

        your_infered_class = self.class_names[top5[0]]
        your_top_score = prediction[top5[0]]
        second_top_score = prediction[top5[1]]

        if your_top_score - second_top_score <= MODEL_DIFERENCE or your_top_score < MODEL_THRESHOLD:
            your_infered_class = MODEL_UNKOWN

        print(f'[AUDIO - GenericSounds] {your_infered_class} ({your_top_score})')

        # print('[AUDIO - GenericSounds] Scores\n'+'\n'.join('  {:12s}: {:.3f}'.format(self.class_names[i], prediction[i]) for i in top5))
        # plot_results(waveform, scores, log_mel_spectrogram)
        # print(class_names[scores.numpy().mean(axis=0).argmax()])  # Prints top score.

        return your_infered_class
