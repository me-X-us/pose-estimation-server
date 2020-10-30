import os
import sys
import argparse
import ast
import csv
import cv2
import time
import torch
import json


sys.path.insert(1, os.getcwd())
from SimpleHRNet import SimpleHRNet
from misc.visualization import check_video_rotation
from saveonJson import makePoint


def main(filename, hrnet_m, hrnet_c, hrnet_j, hrnet_weights, image_resolution, single_person, Notuse_tiny_yolo,
         max_batch_size, csv_output_filename, csv_delimiter, device):
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    # print(device)
    dict_frametoJson = {"Frames":[]}
    image_resolution = ast.literal_eval(image_resolution)

    rotation_code = check_video_rotation(filename)
    video = cv2.VideoCapture(filename)
    assert video.isOpened()
    nof_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # assert csv_output_filename.endswith('.csv')
    # with open(csv_output_filename, 'wt', newline='') as fd:
        # csv_output = csv.writer(fd, delimiter=csv_delimiter)
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
        t = time.time()

        ret, frame = video.read()
        if not ret:
            break
        if rotation_code is not None:
            frame = cv2.rotate(frame, rotation_code)

        pts = model.predict(frame)

            # csv format is:
            #   frame_index,detection_index,<point 0>,<point 1>,...,<point hrnet_j>
            # where each <point N> corresponds to three elements:
            #   y_coordinate,x_coordinate,confidence

            # Json foramt
            # "Frames": [
		    # {
			# "Frame": 0,
			# "Score": 0,
			# "keypoint": [
			# 	{
			# 		"position": {
			# 			"y": 156.9791717529297,
			# 			"x": 955.0
			# 		},
			# 		"part": "nose",
			# 		"score": 0.9094806909561157
			# 	}, ...

        for j, pt in enumerate(pts):
            keypoint_byframe = {"Frame": 0, "keypoint": []}
            keyNum = 0
            row = [index, j] + pt.flatten().tolist()
            if j == 0:  # Person의 경우만
                keypoint_byframe["Frame"] = index
                for idx in range(2, len(row), 3):
                    keypoint_byframe["keypoint"].append(makePoint(row[idx], row[idx + 1], keyNum, row[idx + 2]))
                    keyNum += 1
                    
                dict_frametoJson["Frames"].append(keypoint_byframe)
                    # To use CSV_output
                    # csv_output.writerow(row)

        fps = 1. / (time.time() - t)
        print('\rframe: % 4d / %d - framerate: %f fps ' % (index, nof_frames - 1, fps), end='')

        index += 1
    
    return json.dumps(dict_frametoJson)
    # with open("output.json",'w',encoding="utf-8") as fd:
    #     json.dump(dict_frametoJson,fd,indent='\t') 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=' Json type:\n')
    parser.add_argument("--filename", "-f", help="open the specified video",
                        type=str, default=None)
    parser.add_argument("--hrnet_m", "-m", help="network model - HRNet or PoseResNet", type=str, default='HRNet')
    parser.add_argument("--hrnet_c", "-c", help="hrnet parameters - number of channels (if model is HRNet), "
                                                "resnet size (if model is PoseResNet)", type=int, default=32)
    parser.add_argument("--hrnet_j", "-j", help="hrnet parameters - number of joints", type=int, default=17)
    parser.add_argument("--hrnet_weights", "-w", help="hrnet parameters - path to the pretrained weights",
                        type=str, default="./weights/pose_hrnet_w32_256x192.pth")
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(384, 288)')
    parser.add_argument("--single_person",
                        help="disable the multiperson detection (YOLOv3 or an equivalen detector is required for"
                             "multiperson detection)",
                        action="store_true")
    parser.add_argument("--Notuse_tiny_yolo",
                        help="Use YOLOv3-tiny in place of YOLOv3 (faster person detection). Ignored if --single_person",
                        action="store_true")
    parser.add_argument("--max_batch_size", help="maximum batch size used for inference", type=int, default=16)
    parser.add_argument("--csv_output_filename", help="filename of the Json that will be written.", type=str,
                        default='output.csv')
    parser.add_argument("--csv_delimiter", help="csv delimiter", type=str, default=',')
    parser.add_argument("--device", help="device to be used (default: cuda, if available)."
                                         "Set to `cuda` to use all available GPUs (default); "
                                         "set to `cuda:IDS` to use one or more specific GPUs "
                                         "(e.g. `cuda:0` `cuda:1,2`); "
                                         "set to `cpu` to run on cpu.", type=str, default=None)
    args = parser.parse_args()
    main(**args.__dict__)
