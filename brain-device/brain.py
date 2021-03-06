import datetime
import json

from flask import Flask
from flask import request
from openhab import OpenHAB

app = Flask(__name__)

base_url = 'http://localhost:8080/rest'
openhab = OpenHAB(base_url)
item_label_brain = openhab.get_item('BrainDevice')

eventsAudio = []
eventsVideo = []
MAX_EVENTS = 5
last_event = ""
WINDOWSIZE_SECONDS = 10


@app.route('/test')
def action():
    return 'action received!'


@app.route('/action', methods=['POST'])
def getOpenHabLabel():
    action = request.get_json()
    json = eval(action['action'])
    print(json)

    if json['type'] == "video":
        eventsVideo.append(json)
    else:
        eventsAudio.append(json)

    # Função que trata a informação:
    handleEvents(json)
    return json


def mostFrequentEvent(events):
    if len(events) < MAX_EVENTS:
        return

    moda = {}
    for event in events:
        if event['prediction'] not in moda:
            moda[event['prediction']] = 1
        else:
            moda[event['prediction']] += 1

    mostFrequent = max(moda, key=moda.get)
    print("Moda: ", moda)
    events.clear()
    return mostFrequent


def handleEvents(current_event):
    global last_event
    timestamp = convertTimestamps(current_event['timestamp'])

    mostFrequentAudio = mostFrequentEvent(eventsAudio)
    # mostFrequentVideo = mostFrequentEvent(eventsVideo)

    if mostFrequentAudio is None:
        return

    item_label_brain.state = mostFrequentAudio

    if last_event != "" and (
            convertTimestamps(last_event['timestamp']) + datetime.timedelta(seconds=WINDOWSIZE_SECONDS)) < timestamp:
        print("CLASS:", last_event['prediction'], "LASTED", WINDOWSIZE_SECONDS, "SECONDS", current_event['timestamp'])
        last_event['timestamp'] = current_event['timestamp']

    if last_event == "" or last_event['prediction'] != mostFrequentAudio:
        last_event = {"prediction": mostFrequentAudio, "timestamp": current_event['timestamp']}


def convertTimestamps(timestamp_str):
    return datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")


if __name__ == '__main__':
    app.run()
