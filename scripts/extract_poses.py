import os
import sys
import ast
import cv2
import time
import torch
import json
sys.path.insert(1, os.getcwd())
from SimpleHRNet import SimpleHRNet
from misc.visualization import check_video_rotation
from saveonJson import makePoint
def extract_poses(filename, single_person, Notuse_tiny_yolo):
    hrnet_m = 'HRNet'
    hrnet_c = 32
    hrnet_j = 17
    hrnet_weights = "./weights/pose_hrnet_w32_256x192.pth"
    image_resolution = '(384, 288)'
    max_batch_size = 16
#    device = None
#    if device is not None:
#        device = torch.device(device)
#    else:
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # print(device)
    dict_frametoJson = {"frames":[]}
    image_resolution = ast.literal_eval(image_resolution)
    rotation_code = check_video_rotation(filename)
    video = cv2.VideoCapture(filename)
    assert video.isOpened()
#    nof_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    if Notuse_tiny_yolo:   # default 는 tiny yolo를 사용
        yolo_model_def="./models/detectors/yolo/config/yolov3.cfg"
        yolo_class_path="./models/detectors/yolo/data/coco.names"
        yolo_weights_path="./models/detectors/yolo/weights/yolov3.weights"
    else:
        yolo_model_def="./models/detectors/yolo/config/yolov3-tiny.cfg"
        yolo_class_path="./models/detectors/yolo/data/coco.names"
        yolo_weights_path="./models/detectors/yolo/weights/yolov3-tiny.weights"
    model = SimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        model_name=hrnet_m,
        resolution=image_resolution,
        multiperson=not single_person,
        max_batch_size=max_batch_size,
        yolo_model_def=yolo_model_def,
        yolo_class_path=yolo_class_path,
        yolo_weights_path=yolo_weights_path,
        device=device
    )
    index = 0
    while True:
#        t = time.time()
        ret, frame = video.read()
        if not ret:
            break
        if rotation_code is not None:
            frame = cv2.rotate(frame, rotation_code)
        pts = model.predict(frame)
        for j, pt in enumerate(pts):
            keypoint_byframe = {"frameNo": 0, "keyPoints": []}
            keyNum = 0
            row = [index, j] + pt.flatten().tolist()
            if j == 0:  # Person의 경우만
                keypoint_byframe["frameNo"] = index
                for idx in range(2, len(row), 3):
                    keypoint_byframe["keyPoints"].append(makePoint(row[idx], row[idx + 1], keyNum, row[idx + 2]))
                    keyNum += 1
                dict_frametoJson["frames"].append(keypoint_byframe)
#        fps = 1. / (time.time() - t)
#        print('\rframe: % 4d / %d - framerate: %f fps ' % (index, nof_frames - 1, fps), end='')
        index += 1
    return json.dumps(dict_frametoJson)
    # with open("output.json",'w',encoding="utf-8") as fd:
    #     json.dump(dict_frametoJson,fd,indent='\t') 
