import numpy as np
import tensorflow as tf
import io
import csv
import os
import shutil

vocabulary_dict = {}

with open('../data/audio/vocabulary.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        vocabulary_dict[row[2]] = row[1]

#print(vocabulary_dict)

eval_dict = {}

with open('../data/audio/eval.csv', newline='') as f:
    reader = csv.reader(f)
    iterreader = iter(reader)
    next(iterreader)
    for row in iterreader:
        eval_dict[row[0]] = row[2].split(",")

print(eval_dict)

remove_labels = ["Accelerating_and_revving_and_vroom", "Accordion", "Acoustic_guitar", "Aircraft", "Bass_drum", "Bicycle", "Bicycle_bell", "Boat_and_Water_vehicle", "Bowed_string_instrument",
                 "Brass_instrument", "Bus", "Buzz", "Camera", "Car", "Car_passing_by", "Chicken_and_rooster", "Chime", "Chink_and_clink", "Church_bell", "Cowbell", "Crash_cymbal", "Cricket",
                 "Crow", "Crowd", "Cymbal", "Drum", "Drum_kit", "Electric_guitar", "Finger_snapping", "Fixed-wing_aircraft_and_airplane", "Fowl", "Frog", "Glockenspiel", "Gong", "Growling", "Guitar"
                 "Gull_and_seagull", "Gunshot_and_gunfire", "Harmonica", "Harp", "Hi-hat", "Hiss", "Idling", "Mallet_percussion", "Marimba_and_xylophone",
                 "Ocean", "Organ", "Percussion", "Piano", "Plucked_string_instrument", "Race_car_and_auto_racing", "Rail_transport", "Rattle_(instrument)", "Skateboard", "Snare_drum", "Speech_synthesizer",
                 "Subway_and_metro_and_underground", "Tambourine", "Traffic_noise_and_roadway_noise", "Train", "Truck", "Trumpet", "Vehicle", "Vehicle_horn_and_car_horn_and_honking", "Waves_and_surf",
                 "Wild_animals", "Wind_instrument_and_woodwind_instrument"]
remove_id = []

'''
for label in remove_labels:
    id_remove = ""
    for id_voc in vocabulary_dict:
        if vocabulary_dict[id_voc] == label:
            id_remove = id_voc
            break

    for id_eval in eval_dict:
        for id_voc in eval_dict[id_eval]:
            if id_voc == id_remove:
                remove_id.append(id_eval)

                break

for id in remove_id:
    if id in eval_dict:
        eval_dict.pop(id)

print(len(eval_dict))


for label_id in vocabulary_dict:
    path = "../data/audio/classes/" + vocabulary_dict[label_id]
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

for wavcode in eval_dict:
    for label_id in eval_dict[wavcode]:
        print(vocabulary_dict[label_id])
        shutil.copyfile('../data/audio/eval_audio/FSD50K.eval_audio/' + wavcode + ".wav", '../data/audio/classes/' + vocabulary_dict[label_id] + '/' + wavcode + ".wav") '''