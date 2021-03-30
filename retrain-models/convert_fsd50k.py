import csv
import shutil
from pydub import AudioSegment
import random

DATASETS_PATH = './datasets/'
#DATASETS_PATH ='D:/datasets/'
DATASET = 'FSD50k/'
classes = ['Computer_keyboard', 'Knock', 'Scissors']
ids = {'Computer_keyboard': 42, 'Knock': 106, 'Scissors': 143}
filesInClasses = {}
mappings = []

for i in range(len(classes)):
    filesInClasses[classes[i]] = 0

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
                filesInClasses[values[0]] += 1
        print(filesInClasses)


def save_mappings_to_csv(maps):
    min = 9999
    for classNames in filesInClasses:
        if filesInClasses[classNames] < min:
            min = filesInClasses[classNames]

    countClasses = {}
    for i in range(len(classes)):
        countClasses[classes[i]] = 0

    classes_maps = {}
    for mapping in maps:
        category = mapping[3]
        if category not in classes_maps:
            classes_maps[category] = {'count': 0, 'maps': []}

        classes_maps[category]['count'] += 1
        classes_maps[category]['maps'].append(mapping)

    print(classes_maps)

    with open(DATASETS_PATH + DATASET + 'data_mapping.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "fold", "target", "category"])
        for mapping in maps:
            wav = mapping[0]
            className = mapping[3]
            if countClasses[className] < min and not checkNonRepeatableWav(wav):
                writer.writerow(mapping)
                countClasses[className] += 1
        print(countClasses)

def checkNonRepeatableWav(wav):
    with open(DATASETS_PATH + DATASET + 'data_mapping.csv', 'r', newline='') as file:
        reader = csv.reader(file)
        exist = False
        for row in reader:
            if row[0] == wav:
                print("Repeated:", row[0])
                exist = True
                break
    return exist

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


extract_mappings('dev.csv')
extract_mappings('eval.csv')
save_mappings_to_csv(mappings)
copy_matching_files(mappings)
