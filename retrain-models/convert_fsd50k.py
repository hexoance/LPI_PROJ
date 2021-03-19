import csv
import shutil
from pydub import AudioSegment

DATASETS_PATH = './datasets/'
DATASET = 'FSD50k/'
classes = ['Computer_keyboard', 'Knock', 'Scissors']
ids = {'Computer_keyboard': 42, 'Knock': 106, 'Scissors': 143}
mappings = []


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
        audio = audio[:2500]

        audio.export(DATASETS_PATH + DATASET + "audio/" + mapping[0], format="wav")


extract_mappings('dev.csv')
extract_mappings('eval.csv')
save_mappings_to_csv(mappings)
copy_matching_files(mappings)
