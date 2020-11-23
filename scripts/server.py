from extract_poses import extract_poses
from flask import Flask
from flask import request
from flask import Response

app = Flask(__name__)

@app.route('/')
def hello():
    return Response(extract_poses(request.args['video_path'],False,False), mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')