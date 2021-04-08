import csv
import shutil
from pydub import AudioSegment
import random

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

    # Obter os MIN ficheiros .wav aleatórios de cada classe
    maps = []
    for category in classes_maps:
        for line in random.sample(classes_maps[category]['maps'], min_count):
            maps.append(line)

    print("\nReady to Write:\n", {'Balanced Classes:': maps})


def copy_matching_files(maps):
    for mapping in maps:
        folder = "audio-dev/"
        if mapping[1] == '3':
            folder = "audio-eval/"
        shutil.copyfile(DATASETS_PATH + DATASET + folder + mapping[0], DATASETS_PATH + DATASET + "audio/" + mapping[0])

        audio = AudioSegment.from_wav(DATASETS_PATH + DATASET + "audio/" + mapping[0])

        # create x sec of silence audio segment
        silence = AudioSegment.silent(duration=30000 - (audio.duration_seconds * 1000))  # silence in milliseconds
        audio = audio + silence
        audio = audio[:5000]

        audio.export(DATASETS_PATH + DATASET + "audio/" + mapping[0], format="wav")

readClasses('classes.csv')
readVocabulary('vocabulary.csv')
extract_mappings('dev.csv')
extract_mappings('eval.csv')
balanceWavFiles(mappings)
save_mappings_to_csv(mappings)
copy_matching_files(mappings)
