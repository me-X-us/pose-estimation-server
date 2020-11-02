from extract_poses import extract_poses
from flask import Flask
from flask import request
app = Flask(__name__)

@app.route('/')
def hello():
    return extract_poses(request.args['video_path'],False,False)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
