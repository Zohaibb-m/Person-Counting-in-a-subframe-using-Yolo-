import argparse
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
vertices_date = [
    (362, 73),
    (362, 105),
    (408, 105),
    (408, 73)
]
vertices_cashier_area = [
    (1344, 320),
    (1619, 401),
    (1656, 1077),
    (793, 1053),
    (874, 611),
    (1226, 685)
]

last_five_results = []
last_100_frame_results = []


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
    # for box in result.xyxy[0]:
    #     x1, y1, x2, y2, conf, cls = map(int, box)
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # # Display number of people detected
    # cv2.polylines(frame, [roi_corners_left], True, (0, 255, 0), thickness=2)
    # cv2.putText(frame, f"Front side: {num_people}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)
    # cv2.imwrite('frame.jpeg',frame)
    # frame = cv2.resize(frame,(1024,720))
    # cv2.imwrite('frame.jpeg', frame)
    # cv2.imshow('Frame', frame)
    return num_people


def find_date(frame):
    cv2.imshow('thr', frame)
    mask = np.zeros_like(frame)
    # print(frame.shape)
    roi_corners_left = np.array([vertices_date], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners_left, (255, 255, 255))
    cropped_frame = cv2.bitwise_and(frame, mask)
    hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
    lwr = np.array([0, 0, 0])
    upr = np.array([255, 255, 255])
    msk = cv2.inRange(hsv, lwr, upr)
    thr = 255 - cv2.bitwise_and(cropped_frame, cropped_frame, mask=msk)
    crp_img = cv2.cvtColor(thr, cv2.COLOR_HSV2BGR)
    crp_gry = cv2.cvtColor(crp_img, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(crp_gry, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 71, 100)
    # cv2.imshow('thr', thr)
    extracted_information = pytesseract.image_to_string(thr)[0:2]
    # print('-------------------')
    # print(extracted_information)
    # print('-------------------')
    if extracted_information.isnumeric() and 0 <= int(extracted_information) <= 23:
        if len(last_five_results) != 0 and extracted_information == max(last_five_results, key=last_five_results.count):
            return extracted_information
        if len(last_five_results) == 10:
            last_five_results.pop()
        last_five_results.append(extracted_information)
    elif len(last_five_results) == 0:
        return False
    else:
        return max(last_five_results, key=last_five_results.count)
    # extracted_information = extracted_information[:5] + '-' + extracted_information[7:]

    return extracted_information


sec = 39
mint = 59
hour = 8

previous_count = 0


def run(args):
    # # Load YOLO model
    # model = YOLO(args.weights)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.weights)
    # Open video file
    cap = cv2.VideoCapture(args.source)
    # Get video dimensions
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    i = 0
    csvfile_left = open(f'{args.source[args.source.find("r/") + 2:args.source.find(".mp4")]}_cashier_area.csv', 'w')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # cv2.resize(frame, (720, 1080))
        global hour, mint, sec
        if i % fps == 0:
            sec += 1
            if sec == 60:
                sec = 0
                mint += 1
                if mint == 60:
                    mint = 0
                    hour += 1

        if i % int(fps / int(fps / 2)) == 0:
            # cv2.imshow('thr', frame)
            num_people_left = process_frame_through_post_trained(frame, model, vertices_cashier_area)
            # hour = find_date(frame)
            # print(hour)
            # if not hour:
            #     continue
            last_100_frame_results.append(num_people_left)
            if len(last_100_frame_results) > 100:
                # print(last_100_frame_results)
                last_100_frame_results.pop(0)
                curr_count = max(last_100_frame_results, key=last_100_frame_results.count)
                global previous_count
                if previous_count < curr_count:
                    csvfile_left.write(f'{hour},{curr_count - previous_count},0\n')
                    previous_count = curr_count
                elif previous_count > curr_count:
                    csvfile_left.write(f'{hour},0,{previous_count - curr_count}\n')
                    previous_count = curr_count
        i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # print(i)

    # Release resources
    cap.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    filenames = ['ch02_20230330090000.mp4', 'ch02_20230330162911.mp4', 'ch02_20230330165845.mp4',
                 'ch02_20230330211521.mp4']
    # filenames = ['ch02_20230330165845.mp4']
    for file in filenames:
        parser = argparse.ArgumentParser()
        parser.add_argument('--source', type=str, default=f'../../cctv_ger/{file}', help='path to video file')
        # parser.add_argument('--source', type=str, default=f'../sample_videos/sample17.mp4', help='path to video file')
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
