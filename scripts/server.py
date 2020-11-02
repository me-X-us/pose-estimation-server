from extract_poses import extract_poses
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return extract_poses("testvid.mp4",False,False)
