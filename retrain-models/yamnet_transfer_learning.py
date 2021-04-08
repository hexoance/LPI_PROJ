"""
# Transfer Learning with YAMNet for environmental sound classification

[YAMNet](https://tfhub.dev/google/yamnet/1) is an audio event classifier that can predict audio events from [521 classes](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv), like laughter, barking, or a siren. 

 In this tutorial you will learn how to:

- Load and use the YAMNet model for inference.
- Build a new model using the YAMNet embeddings to classify cat and dog sounds.
- Evaluate and export your model.

## Import TensorFlow and other libraries

Start by installing [TensorFlow I/O](https://www.tensorflow.org/io), which will make it easier for you to load audio files off disk.
"""
import csv
import os

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
from IPython import display

"""## About YAMNet

YAMNet is an audio event classifier that takes audio waveform as input and makes independent predictions 
for each of 521 audio events from the [AudioSet](https://research.google.com/audioset/) ontology.

Internally, the model extracts "frames" from the audio signal and processes batches of these frames. 
This version of the model uses frames that are 0.96s long and extracts one frame every 0.48s.

The model accepts a 1-D float32 Tensor or NumPy array containing a waveform of arbitrary length, 
represented as mono 16 kHz samples in the range `[-1.0, +1.0]`. This tutorial contains code to help you 
convert a `.wav` file into the correct format.

The model returns 3 outputs, including the class scores, embeddings (which you will use for transfer learning), 
and the log mel spectrogram. You can find more details [here](https://tfhub.dev/google/yamnet/1), 
and this tutorial will walk you through using these in practice.

One specific use of YAMNet is as a high-level feature extractor: the `1024-D` embedding output of YAMNet can be used as 
the input features of another shallow model which can then be trained on a small amount of data for a particular task. 
This allows the quick creation of specialized audio classifiers without requiring a lot of labeled data and without 
having to train a large model end-to-end.

You will use YAMNet's embeddings output for transfer learning and train one or more 
[Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) layers on top of this.

First, you will try the model and see the results of classifying audio. You will then construct the data pre-processing pipeline.

### Loading YAMNet from TensorFlow Hub

You are going to use YAMNet from [Tensorflow Hub](https://tfhub.dev/) to extract the embeddings from the sound files.
Loading a model from TensorFlow Hub is straightforward: choose the model, copy its URL and use the `load` function.
Note: to read the documentation of the model, you can use the model url in your browser.
"""

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

"""With the model loaded and following the [models's basic usage tutorial](https://www.tensorflow.org/hub/tutorials/yamnet) you'll download a sample wav file and run the inference.
"""

testing_wav_file_name = tf.keras.utils.get_file('miaow_16k.wav',
                                                'https://storage.googleapis.com/audioset/miaow_16k.wav',
                                                cache_dir='./',
                                                cache_subdir='test_data')

print(testing_wav_file_name)

"""You will need a function to load the audio files. They will also be used later when working with the training data.
Note: The returned `wav_data` from `load_wav_16k_mono` is already normalized to values in `[-1.0, 1.0]` (as stated in the model's [documentation](https://tfhub.dev/google/yamnet/1)).
"""


# Util functions for loading audio files and ensure the correct sample rate

@tf.function
def load_wav_16k_mono(filename):
    """ read in a waveform file and convert to 16 kHz mono """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


testing_wav_data = load_wav_16k_mono(testing_wav_file_name)

_ = plt.plot(testing_wav_data)

# Play the audio file.
display.Audio(testing_wav_data, rate=16000)

"""### Load the class mapping
It's important to load the class names that YAMNet is able to recognize.
The mapping file is present at `yamnet_model.class_map_path()`, in the `csv` format.
"""

class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names = list(pd.read_csv(class_map_path)['display_name'])

for name in class_names[:20]:
    print(name)
print('...')

"""### Run inference
YAMNet provides frame-level class-scores (i.e., 521 scores for every frame).
In order to determine clip-level predictions, the scores can be aggregated per-class 
across frames (e.g., using mean or max aggregation). This is done below by `scores_np.mean(axis=0)`. 
Finally, in order to find the top-scored class at the clip-level, we take the maximum of the 521 aggregated scores.
"""

scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
class_scores = tf.reduce_mean(scores, axis=0)
top_class = tf.argmax(class_scores)
infered_class = class_names[top_class]

print(f'The main sound is: {infered_class}')
print(f'The embeddings shape: {embeddings.shape}')

"""Note: The model correctly inferred an animal sound. Your goal is to increase accuracy for specific classes. Also, notice that the the model generated 13 embeddings, 1 per frame.

## ESC-50 dataset

The [ESC-50 dataset](https://github.com/karolpiczak/ESC-50#repository-content), 
well described [here](https://www.karolpiczak.com/papers/Piczak2015-ESC-Dataset.pdf), 
is a labeled collection of 2000 environmental audio recordings (each 5 seconds long). 
The data consists of 50 classes, with 40 examples per class.

Next, you will download and extract it.
"""

"""
_ = tf.keras.utils.get_file('esc-50.zip',
                            'https://github.com/karoldvl/ESC-50/archive/master.zip',
                            cache_dir='./',
                            cache_subdir='datasets',
                            extract=True)
"""

"""### Explore the data
The metadata for each file is specified in the csv file at `./datasets/ESC-50-master/meta/data_mapping.csv` and 
all the audio files are in `./datasets/ESC-50-master/audio/`
You will create a pandas dataframe with the mapping and use that to have a clearer view of the data.
"""

datasets = ['ESC-50-master', 'FSD50k']
DATASET_NAME = datasets[1]
fold_val = 2
fold_eval = 3

#DATASETS_PATH = './datasets/'
DATASETS_PATH ='D:/datasets/'

files_csv = DATASETS_PATH + DATASET_NAME + '/data_mapping.csv'
base_data_path = DATASETS_PATH + DATASET_NAME + '/audio/'

pd_data = pd.read_csv(files_csv)
pd_data.head()

"""### Filter the data
Given the data on the dataframe, you will apply some transformations:
- filter out rows and use only the selected classes (dog and cat). If you want to use any other classes, this is where you can choose them.
- change the filename to have the full path. This will make loading easier later.
- change targets to be within a specific range. In this example, dog will remain 0, but cat will become 1 instead of its original value of 5.
"""

my_classes = []


def readClasses(file):
    with open(DATASETS_PATH + DATASET_NAME + '/' + file, newline='') as f:
        reader = csv.reader(f)
        iterreader = iter(reader)
        next(iterreader)
        for row in iterreader:
            my_classes.append(row[0])

        #print(my_classes)


readClasses('classes.csv')

map_class_to_id = {}

for i in range(len(my_classes)):
    map_class_to_id[my_classes[i]] = i

filtered_pd = pd_data[pd_data.category.isin(my_classes)]

class_id = filtered_pd['category'].apply(lambda name: map_class_to_id[name])
filtered_pd = filtered_pd.assign(target=class_id)

full_path = filtered_pd['filename'].apply(lambda row: os.path.join(base_data_path, row))
filtered_pd = filtered_pd.assign(filename=full_path)

filtered_pd.head(10)

"""### Load the audio files and retrieve embeddings
Here you'll apply the `load_wav_16k_mono` and prepare the wav data for the model.
When extracting embeddings from the wav data, you get an array of shape `(N, 1024)` where `N` is the number of frames that YAMNet found (one for every 0.48 seconds of audio).
Your model will use each frame as one input so you need to to create a new column that has one frame per row. You also need to expand the labels and fold column to proper reflect these new rows.
The expanded fold column keeps the original value. You cannot mix frames because, when doing the splits, you might end with parts of the same audio on different splits and that would make our validation and test steps less effective.
"""

filenames = filtered_pd['filename']
targets = filtered_pd['target']
folds = filtered_pd['fold']

main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))
main_ds.element_spec


def load_wav_for_map(filename, label, fold):
    return load_wav_16k_mono(filename), label, fold


main_ds = main_ds.map(load_wav_for_map)
main_ds.element_spec


# applies the embedding extraction model to a wav data
def extract_embedding(wav_data, label, fold):
    ''' run YAMNet to extract embedding from the wav data '''
    scores, embeddings, spectrogram = yamnet_model(wav_data)
    num_embeddings = tf.shape(embeddings)[0]
    return (embeddings,
            tf.repeat(label, num_embeddings),
            tf.repeat(fold, num_embeddings))


# extract embedding
main_ds = main_ds.map(extract_embedding).unbatch()
main_ds.element_spec

"""### Split the data
You will use the `fold` column to split the dataset into train, validation and test.
The fold values are so that files from the same original wav file are keep on the same split, you can find more information on the [paper](https://www.karolpiczak.com/papers/Piczak2015-ESC-Dataset.pdf) describing the dataset.
The last step is to remove the `fold` column from the dataset since we're not going to use it anymore on the training process.
"""

cached_ds = main_ds.cache()
train_ds = cached_ds.filter(lambda embedding, label, fold: fold < fold_val)
val_ds = cached_ds.filter(lambda embedding, label, fold: fold == fold_val)
test_ds = cached_ds.filter(lambda embedding, label, fold: fold == fold_eval)

# remove the folds column now that it's not needed anymore
remove_fold_column = lambda embedding, label, fold: (embedding, label)

train_ds = train_ds.map(remove_fold_column)
val_ds = val_ds.map(remove_fold_column)
test_ds = test_ds.map(remove_fold_column)

train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)

"""## Create your model
You did most of the work!
Next, define a very simple Sequential Model to start with -- one hiden layer and 2 outputs to recognize cats and dogs.
"""

my_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024), dtype=tf.float32, name='input_embedding'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(my_classes))
], name='my_model')

my_model.summary()

my_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=['accuracy']
)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

history = my_model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=callback)

"""Lets run the evaluate method on the test data just to be sure there's no overfitting."""

loss, accuracy = my_model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

"""You did it!
## Test your model
Next, try your model on the embedding from the previous test using YAMNet only.
"""

scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
result = my_model(embeddings).numpy()

infered_class = my_classes[result.mean(axis=0).argmax()]
print(f'The main sound is: {infered_class}')

"""## Save a model that can directly take a wav file as input
Your model works when you give it the embeddings as input.
In a real situation you'll want to give it the sound data directly.
To do that you will combine YAMNet with your model into one single model that you can export for other applications.
To make it easier to use the model's result, the final layer will be a `reduce_mean` operation. 
When using this model for serving, as you will see bellow, you will need the name of the final layer. 
If you don't define one, TF will auto define an incremental one that makes it hard to test as it will keep changing 
everytime you train the model. When using a raw tf operation you can't assign a name to it. 
To address this issue, you'll create a custom layer that just apply `reduce_mean` and you will call it 'classifier'.
"""


class ReduceMeanLayer(tf.keras.layers.Layer):
    def __init__(self, axis=0, **kwargs):
        super(ReduceMeanLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, input):
        return tf.math.reduce_mean(input, axis=self.axis)


saved_model_path = '../models/new_yamnet'

input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
embedding_extraction_layer = hub.KerasLayer(yamnet_model_handle, trainable=False, name='yamnet')
_, embeddings_output, _ = embedding_extraction_layer(input_segment)
serving_outputs = my_model(embeddings_output)
serving_outputs = ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)
serving_model = tf.keras.Model(input_segment, serving_outputs)
serving_model.save(saved_model_path, include_optimizer=False)

tf.keras.utils.plot_model(serving_model)

"""Load your saved model to verify that it works as expected."""

reloaded_model = tf.saved_model.load(saved_model_path)

"""And for the final test: given some sound data, does your model return the correct result?"""

reloaded_results = reloaded_model(testing_wav_data)
detected_sound = my_classes[tf.argmax(reloaded_results)]
print(f'The main sound is: {detected_sound}')

"""If you want to try your new model on a serving setup, you can use the 'serving_default' signature."""

serving_results = reloaded_model.signatures['serving_default'](testing_wav_data)
detected_sound = my_classes[tf.argmax(serving_results['classifier'])]
print(f'The main sound is: {detected_sound}')

"""## (Optional) Some more testing
The model is ready. Let's compare it to YAMNet on the test dataset.
"""

test_pd = filtered_pd.loc[filtered_pd['fold'] == fold_eval]
row = test_pd.sample(1)
filename = row['filename'].item()
print(filename)
waveform = load_wav_16k_mono(filename)
print(f'Waveform values: {waveform}')
_ = plt.plot(waveform)

display.Audio(waveform, rate=16000)

# Run the model, check the output.
scores, embeddings, spectrogram = yamnet_model(waveform)
class_scores = tf.reduce_mean(scores, axis=0)
top_class = tf.argmax(class_scores)
infered_class = class_names[top_class]
top_score = class_scores[top_class]
print(f'[YAMNet] The main sound is: {infered_class} ({top_score})')

reloaded_results = reloaded_model(waveform)
your_top_class = tf.argmax(reloaded_results)
your_infered_class = my_classes[your_top_class]
class_probabilities = tf.nn.softmax(reloaded_results, axis=-1)
your_top_score = class_probabilities[your_top_class]
print(f'[Your model] The main sound is: {your_infered_class} ({your_top_score})')

"""## Next steps
You just created a model that can classify sounds from dogs or cats. 
With the same idea and proper data you could, for example, build a bird recognizer based on their singing.
Let us know what you come up with! Share with on social media your project.
"""
