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
    print(moda)
    print(mostFrequent)
    events.clear()
    return mostFrequent


def handleEvents(lastEvent):
    #timestamp = datetime.datetime.strptime(lastEvent['timestamp'], "%Y-%m-%d %H:%M:%S.%f")

    mostFrequentAudio= mostFrequentEvent(eventsAudio)
    mostFrequentVideo= mostFrequentEvent(eventsVideo)
    
    if mostFrequentAudio is not None:
        item_label_brain.state = mostFrequentAudio


if __name__ == '__main__':
    app.run()
