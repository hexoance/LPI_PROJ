import os
import csv
from pydub import AudioSegment
import random
from utils.silence_removal import trim_silence, remove_middle_silence

TRAIN_DS_PERCENTAGE = 0.8
VAL_DS_PERCENTAGE = 0.0
TEST_DS_PERCENTAGE = 0.2

if TRAIN_DS_PERCENTAGE + VAL_DS_PERCENTAGE + TEST_DS_PERCENTAGE != 1:
    raise Exception('Train/Val/Test split sum must be equal to 100%')

DATASETS_PATH = './datasets/'
# DATASETS_PATH = 'D:/datasets/'
DATASETS = ['CUSTOM-SOUNDS', 'FSD50k', 'ESC-50']
DATASET = DATASETS[1] + '/'
GENERATED_PATH = DATASETS_PATH + 'GENERATED-SOUNDS/'


def read_classes(file):
    classes_read = []
    with open(file, newline='') as f:
        reader = csv.reader(f)
        iterator = iter(reader)
        next(iterator)
        for row in iterator:
            classes_read.append(row[0])
    print(classes_read)
    return classes_read


def read_vocabulary(file):
    vocabulary_read = {}
    with open(DATASETS_PATH + DATASET + file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            vocabulary_read[row[1]] = row[0]
    return vocabulary_read


def extract_mappings(file, maps, vocab):
    with open(DATASETS_PATH + DATASET + file, newline='') as f:
        reader = csv.reader(f)
        iterreader = iter(reader)
        next(iterreader)
        for row in iterreader:
            values = list(filter(lambda x: x in row[1].split(','), classes))
            if values:
                folder = DATASETS_PATH + DATASET + "audio-" + file.split('.')[0]
                maps.append([folder + "/" + row[0] + ".wav", str(0), str(vocab[values[0]]), values[0]])
    return maps


def sound_filter(maps):
    filtered_files = {}
    with open(DATASETS_PATH + DATASET + "sound_filter.csv", newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            filtered_files[row[0]] = row
    filtered_maps = []
    for file in maps:
        if file[0].split("/")[-1] in filtered_files:
            filtered_maps.append(file)
    return filtered_maps


def balance_wav_files(maps):
    # Classe -> Num. total de ficheiros .wav dessa classe (count) -> Array os ficheiros .wav (maps)
    classes_maps = {}
    for mapping in maps:
        category = mapping[3]
        if category not in classes_maps:
            classes_maps[category] = {'count': 0, 'maps': []}
        classes_maps[category]['count'] += 1
        classes_maps[category]['maps'].append(mapping)

    print("\nClasses: \n", classes_maps)

    # Número de ficheiros da Classe com menos ficheiros
    min_count = min(val['count'] for key, val in classes_maps.items())
    print("\nClasses to Balance (MIN):", min_count)

    train_ds = int(min_count * TRAIN_DS_PERCENTAGE)
    val_ds = int(min_count * VAL_DS_PERCENTAGE)
    test_ds = int(min_count * TEST_DS_PERCENTAGE)

    dif = min_count - (train_ds + val_ds + test_ds)
    train_ds += dif

    print("Train DS size: " + str(train_ds))
    print("Val DS size: " + str(val_ds))
    print("Test DS size: " + str(test_ds))

    # Obter os MIN ficheiros .wav aleatórios de cada classe
    maps = []
    for category in classes_maps:

        train_ds_counter = train_ds
        val_ds_counter = val_ds
        test_ds_counter = test_ds

        for line in random.sample(classes_maps[category]['maps'], min_count):
            if train_ds_counter > 0:
                line[1] = 1
                train_ds_counter -= 1
            elif val_ds_counter > 0:
                line[1] = 2
                val_ds_counter -= 1
            elif test_ds_counter > 0:
                line[1] = 3
                test_ds_counter -= 1
            maps.append(line)

    print("\nReady to Write:\n", {'Balanced Classes:': maps})
    return maps


def save_mappings_to_csv(maps):
    with open(GENERATED_PATH + 'mappings.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "fold", "target", "category"])
        for mapping in maps:
            mapping[0] = mapping[0].split('/')[-1]
            writer.writerow(mapping)


def delete_all_files_from_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            os.remove(os.path.join(root, file))


def copy_matching_files(maps):
    audio_preprocessing_operations = [
        trim_silence,
        remove_middle_silence,
    ]

    for mapping in maps:
        audio = AudioSegment.from_wav(mapping[0])

        # Run preprocessing operations on the audio, only if its not silence
        if mappings[3] != 'silence':
            for operation in audio_preprocessing_operations:
                audio = operation(audio)

        audio.export(GENERATED_PATH + "audio/" + mapping[0].split('/')[-1], format="wav")


if __name__ == '__main__':
    classes = read_classes('classes_to_retrain.csv')
    vocabulary = read_vocabulary('vocabulary.csv')
    mappings = []
    mappings = extract_mappings('dev.csv', mappings, vocabulary)
    mappings = extract_mappings('eval.csv', mappings, vocabulary)
    mappings = sound_filter(mappings)
    mappings = balance_wav_files(mappings)
    delete_all_files_from_folder(GENERATED_PATH + "audio")
    copy_matching_files(mappings)
    save_mappings_to_csv(mappings)
