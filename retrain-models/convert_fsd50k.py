import csv
import shutil
from pydub import AudioSegment
import random

TRAIN_DS_PERCENTAGE = 0.8
VAL_DS_PERCENTAGE = 0.0
TEST_DS_PERCENTAGE = 0.2

if TRAIN_DS_PERCENTAGE + VAL_DS_PERCENTAGE + TEST_DS_PERCENTAGE != 1:
    raise Exception('Train/Val/Test split sum must be equal to 100%')

DATASETS_PATH = './datasets/'
#DATASETS_PATH = 'D:/datasets/'
DATASET = 'FSD50k/'
classes = []
ids = {}
mappings = []

def readClasses(file):
    with open(DATASETS_PATH + DATASET + file, newline='') as f:
        reader = csv.reader(f)
        iterreader = iter(reader)
        next(iterreader)
        for row in iterreader:
            classes.append(row[0])

        print(classes)

def readVocabulary(file):
    with open(DATASETS_PATH + DATASET + file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            ids[row[1]] = row[0]


def extract_mappings(file):
    with open(DATASETS_PATH + DATASET + file, newline='') as f:
        reader = csv.reader(f)
        iterreader = iter(reader)
        next(iterreader)
        for row in iterreader:
            values = list(filter(lambda x: x in row[1].split(','), classes))
            if values:
                fold = 3
                if file == "dev.csv":
                    # default split column in dev.csv is val which is fold 2
                    fold = 2
                    if row[3] == "train":
                        fold = 1
                mappings.append([row[0] + ".wav", str(fold), str(ids[values[0]]), values[0]])


def save_mappings_to_csv(maps):
    with open(DATASETS_PATH + DATASET + 'data_mapping.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "fold", "target", "category"])
        for mapping in maps:
            writer.writerow(mapping)


def balanceWavFiles(maps):
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


def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms


def copy_matching_files(maps):
    for mapping in maps:
        folder = "audio-dev/"

        try:
            with open(DATASETS_PATH + DATASET + folder + mapping[0]) as f:
                pass
                # Do something with the file
        except IOError:
            folder = "audio-eval/"

        shutil.copyfile(DATASETS_PATH + DATASET + folder + mapping[0], DATASETS_PATH + DATASET + "audio/" + mapping[0])

        audio = AudioSegment.from_wav(DATASETS_PATH + DATASET + "audio/" + mapping[0])

        # create x sec of silence audio segment
        #silence = AudioSegment.silent(duration=30000 - (audio.duration_seconds * 1000))  # silence in milliseconds
        #audio = audio + silence
        #audio = audio[:5000]

        start_trim = detect_leading_silence(audio)
        end_trim = detect_leading_silence(audio.reverse())

        duration = len(audio)
        trimmed_sound = audio[start_trim:duration - end_trim]

        trimmed_sound.export(DATASETS_PATH + DATASET + "audio/" + mapping[0], format="wav")
        #audio.export(DATASETS_PATH + DATASET + "audio/" + mapping[0], format="wav")


def sound_filter(maps):

    filtered_files = {}
    with open(DATASETS_PATH + DATASET + "sound_filter.csv", newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            filtered_files[row[0]] = row
    filtered_maps = []
    for file in maps:
        if file[0] in filtered_files:
            filtered_maps.append(file)
    #print({"Sounds Filtered" : filtered_maps})
    return filtered_maps


readClasses('classes.csv')
readVocabulary('vocabulary.csv')
extract_mappings('dev.csv')
extract_mappings('eval.csv')
#mappings = sound_filter(mappings)
mappings = balanceWavFiles(mappings)
save_mappings_to_csv(mappings)
copy_matching_files(mappings)
