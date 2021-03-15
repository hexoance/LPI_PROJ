from flask import Flask
from flask import request
from openhab import OpenHAB

app = Flask(__name__)

base_url = 'http://localhost:8080/rest'
openhab = OpenHAB(base_url)

@app.route('/action')
def action():
    return 'action received!'


@app.route('/test', methods=['POST'])
def test():
    req = request.get_json()
    print(req)
    return 'teste!'


if __name__ == '__main__':
    app.run()
