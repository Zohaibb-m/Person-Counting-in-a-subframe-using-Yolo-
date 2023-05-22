import argparse
import random

import cv2
from ultralytics import YOLO
import numpy as np
import torch
import pytesseract
import os

vertices_left = [
    (64, 254),
    (747, 940),
    (1116, 624),
    (244, 175)
]
vertices_front = [
    (936, 822),
    (1247, 1009),
    (1613, 417),
    (1422, 347)
]
vertices_right = [
    (721, 17),
    (651, 184),
    (1354, 461),
    (1451, 210)
]


def process_frame_through_pre_trained(frame, model, vertices, iou_thresh):
    # Mask out the Region of Interest
    mask = np.zeros_like(frame)
    roi_corners_left = np.array([vertices], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners_left, (255, 255, 255))
    # Create the cropped frame
    cropped_frame = cv2.bitwise_and(frame, mask)
    # Predict
    result = model(cropped_frame, mode='predict', classes=[0], iou=iou_thresh)
    # Count of people detected
    num_people = len(result[0].boxes)
    # Draw bounding boxes on the original frame
    for i, det in enumerate(result[0].boxes):
        x1, y1, x2, y2 = det.xyxy[0].tolist()
        label = f"Person {i + 1}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # Display number of people detected
    cv2.polylines(frame, [roi_corners_left], True, (0, 255, 0), thickness=2)
    cv2.putText(frame, f"Left side: {num_people}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)
    cv2.imshow('Frame', frame)
    return frame


def process_frame_through_post_trained(frame, model, vertices):
    # Mask out Region of Interest
    mask = np.zeros_like(frame)
    roi_corners_left = np.array([vertices], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners_left, (255, 255, 255))
    # Create the cropped frame
    cropped_frame = cv2.bitwise_and(frame, mask)
    # Predict
    result = model(cropped_frame)
    # Count number of people detected
    num_people = len(result.xyxy[0])
    # Draw bounding boxes on the original frame
    for box in result.xyxy[0]:
        x1, y1, x2, y2, conf, cls = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # Display number of people detected
    cv2.polylines(frame, [roi_corners_left], True, (0, 255, 0), thickness=2)
    cv2.putText(frame, f"Front side: {num_people}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)
    return frame

def run(args):
    # # Load YOLO model
    # model = YOLO(args.weights)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.weights)
    # model.autoshape()
    # Open video file
    cap = cv2.VideoCapture(args.source)
    # print('Hello')
    # Get video dimensions
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # print(fps)
    # Create video writer object
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    i = 0
    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # cv2.resize(frame, (720, 1080))
        if i % (fps / 2) == 0:
            cv2.imshow('thr', frame)
            processed_frame = process_frame_through_post_trained(frame, model, vertices_left)
            # print(hour)
            # if not hour:
            #     continue
            # cv2.imwrite(f"SamplePics/sample{i}.jpg", processed_frame)
        i += 1
        # print(i)
        # Write processed frame to output video
        out.write(processed_frame)

        # Display processed frame
        cv2.imshow('Processed Frame', processed_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # filenames = ['ch03_20230330090000.mp4', 'ch03_20230330130820.mp4', 'ch03_20230330165844.mp4',
                 # 'ch03_20230330171942.mp4', 'ch03_20230330213502.mp4']
    filenames = [0]
    for file in filenames:
        parser = argparse.ArgumentParser()
        parser.add_argument('--source', type=str, default=f'../sample_videos/sample4.mp4', help='path to video file')
        # parser.add_argument('--cfg', type=str, default='yolov5l.yaml', help = 'model.yaml path')
        parser.add_argument('--weights', nargs='+', type=str,
                            default=r"/home/kamran/tkbees/Project_Person_Counting/yolov5/Yolov5l_Fine_Tuned.pt",
                            help='path to YOLO weights file')
        # parser.add_argument('--iou-thresh', type=float, default=0.5, help='IOU threshold for NMS')
        # # parser.add_argument('--view-img', action='store_true', help='display results')
        # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        # parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        # parser.add_argument('--is_save', type=bool, default=False, help='do not save images/videos')
        # parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        opt = parser.parse_args()
        os.environ['TESSDATA_PREFIX'] = '/home/kamran/miniconda3/envs/yolov8/share/'
        run(opt)
