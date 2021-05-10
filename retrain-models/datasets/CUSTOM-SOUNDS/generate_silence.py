import os
import csv
from pydub import AudioSegment

SILENCE_DURATION = 5000
SILENCE_FILES_GENERATED = 150
AUDIO_FOLDER = "audio"

# DATASETS_PATH = './datasets/'
DATASETS_PATH = 'D:/datasets/'
CUSTOM_SOUNDS = 'CUSTOM-SOUNDS/'


def delete_all_files_from_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            os.remove(os.path.join(root, file))


def generate_silence_sounds(folder, n_files, duration):
    for i in range(0, n_files):
        # create x millisecond(s) of silence
        silence_sound = AudioSegment.silent(duration=duration)

        # Either save modified audio
        silence_sound.export(folder + "/" + str(i) + ".wav", format="wav")


def create_silence_mappings(n_files):
    with open(DATASETS_PATH + CUSTOM_SOUNDS + 'mappings.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["file_id", "category_id"])
        for i in range(0, n_files):
            writer.writerow([str(i), str(0)])


delete_all_files_from_folder(DATASETS_PATH + CUSTOM_SOUNDS + AUDIO_FOLDER)
generate_silence_sounds(DATASETS_PATH + CUSTOM_SOUNDS + AUDIO_FOLDER, SILENCE_FILES_GENERATED, SILENCE_DURATION)
create_silence_mappings(SILENCE_FILES_GENERATED)
