import argparse
import time

import cv2
import torch
import numpy as np
import os
vertices_left = [
    (62, 239),
    (811, 1075),
    (1250, 700),
    (290, 150)
]
vertices_front = [
    (787,513),
    (1309,811),
    (1667,295),
    (1277,173)
]
vertices_right = [
    (335,127),
    (1011,399),
    (1283,147),
    (537,5)
]

hour = 8
mint = 59
sec = 55
def process_frame(frame, model, vertices):
    # Crop frame
    copy_frame = frame
    mask = np.zeros_like(frame)
    # print(frame.shape)
    roi_corners = np.array([vertices], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, (255, 255, 255))
    cropped_frame = cv2.bitwise_and(frame, mask)
    # Detect people using YOLO
    # result_left = model(cropped_frame_left, classes = [0])
    result = model(cropped_frame)
    # print(result_left)
    # Count number of people detected
    num_people = len(result.xyxy[0])
    # num_people_left = len(result_left[0].boxes)
    # for box in result.xyxy[0]:
    #     x1, y1, x2, y2, conf, cls = map(int, box)
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # print("Hello")
    # Draw bounding boxes on original frame
    # if draw_bounding_box:
    # for i, det in enumerate(result_left[0].boxes):
    #         x1, y1, x2, y2 = det.xyxy[0].tolist()
    # label = f"Person {i+1}"
    # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    # cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # # Display number of people detected
    # cv2.polylines(frame, [roi_corners], True, (0, 255, 0), thickness=2)
    # cv2.putText(frame, f"Left side: {num_people}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)
    # cv2.imshow('Frame',frame)
    return num_people


def run(args):
    # # Load YOLO model
    # model = YOLO(args.weights)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.weights)
    # model.autoshape()
    print("Hello")
    # Open video file
    cap = cv2.VideoCapture(args.source)

    # Get video dimensions
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # print(fps)
    # Create video writer object
    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    i = 0
    # Process each frame
    curr_people_left = 0
    curr_people_front = 0
    curr_people_right = 0
    csvfile_left = open('Logs_left_side.csv', 'a')
    csvfile_right = open('Logs_right_side.csv', 'a')
    csvfile_front = open('Logs_front_side.csv', 'a')
    global sec, mint, hour
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % fps == 0:
            sec += 1
            # print(i)
            if sec >= 60:
                sec = 0
                mint += 1
                if mint >= 60:
                    mint = 0
                    hour += 1
        if i % (fps/2) == 0:
            num_people_left = process_frame(frame, model,vertices_left)
            num_people_right = process_frame(frame, model,vertices_right)
            num_people_front = process_frame(frame, model,vertices_front)
            # cv2.imwrite(f"SamplePics/sample{i}.jpg", processed_frame)
            if num_people_left == 0:
                curr_people_left = 0
            if curr_people_left != num_people_left:
                csvfile_left.write(f'{hour}:{mint}:{sec},{num_people_left}\n')
                curr_people_left = num_people_left
                # print(f'{hour}:{mint}:{sec},{num_people_left} Left\n')
            if num_people_right == 0:
                curr_people_right = 0
            if curr_people_right != num_people_right:
                csvfile_right.write(f'{hour}:{mint}:{sec},{num_people_right}\n')
                curr_people_right = num_people_right
                # print(f'{hour}:{mint}:{sec},{num_people_right} Right\n')
            if num_people_front == 0:
                curr_people_front = 0
            if curr_people_front != num_people_front:
                csvfile_front.write(f'{hour}:{mint}:{sec},{num_people_front}\n')
                curr_people_front = num_people_front
                # print(f'{hour}:{mint}:{sec},{num_people_front} Front\n')
        i += 1
        if i%1200 == 0:
            print(f'{hour}:{mint}:{sec}')
        # Write processed frame to output video
        # out.write(processed_frame)

        # Display processed frame
        # cv2.imshow('Processed Frame', processed_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    fileTime = open('times.txt','a')
    # filenames = ['ch03_20230330090000.mp4' , 'ch03_20230330130820.mp4', 'ch03_20230330165844.mp4','ch03_20230330171942.mp4','ch03_20230330213502.mp4']
    filenames = ['ch03_20230330090000.mp4' ]
    for file in filenames:
        parser = argparse.ArgumentParser()
        parser.add_argument('--source', type=str, default=f'../Ch 2 and ch 3/{file}', help='path to video file')
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
        st = time.time()
        run(opt)
        fileTime.write(file)
        fileTime.write(str(time.time()-st))
        print(time.time() - st)
        print(f'{hour}:{mint}:{sec}')
