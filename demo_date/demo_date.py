import argparse
import cv2
import numpy as np
import pytesseract
import os
from easyocr import Reader
import datetime

vertices = [
    [
        (38, 70),
        (83, 70),
        (83, 109),
        (38, 109)

    ], [
        (101, 70),
        (154, 70),
        (154,  107),
        (101, 107)
    ],
    [
        (170, 70),
        (255, 70),
        (255, 107),
        (170, 107)
    ],
    #[     (272, 70),
    #     (346, 70),
    #     (346, 107),
    #     (272, 107)
    # ],
    [
        (362, 73),
        (362, 105),
        (408, 105),
        (408, 73)
    ], [
        (424, 70),
        (472, 70),
        (472, 107),
        (424, 107)
    ], [
        (493, 70),
        (540, 70),
        (540, 107),
        (493, 107)
    ]
]

vertices_full_date = [
    (35, 70),
    (540, 70),
    (540, 107),
    (35, 107)
]

# coordinates = [[35, 70, 84, 107], [101, 70, 149, 107], [170, 70, 259, 107], [272, 70, 346, 107], [361, 70, 411, 107],
#                [424, 70, 472, 107], [493, 70, 540, 107]]

weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
reader = Reader(['en'], True)
last_five_results = []
sec = 56
mint = 38
hour = 21
date = False
day = False
month = False
year = False
date_found = False


def find_date(frame):
    global date, month, year, day, date_found
    if not date:
        cropped_frame = crop_frame(frame, vertices[1])
        result = reader.readtext(cropped_frame, allowlist='0123456789')
        for (bbox, text, prob) in result:
            if 31 >= int(text) > 0:
                date = int(text)
    if not month:
        cropped_frame = crop_frame(frame, vertices[0])
        # cropped_frame = pre_process(cropped_frame)
        # cv2.imshow('fr', cropped_frame)
        result = reader.readtext(cropped_frame, allowlist='0123456789')
        for (bbox, text, prob) in result:
            print(f'Result{text}')
            if 12 >= int(text) > 0:
                month = int(text)
    if not year:
        cropped_frame = crop_frame(frame, vertices[2])
        result = reader.readtext(cropped_frame, allowlist='0123456789')
        for (bbox, text, prob) in result:
            if len(text) == 4:
                year = int(text)
    if not day:
        if date and month and year:
            day = weekdays[datetime.datetime(year, month, date).weekday()]

    if date and month and year and day:
        date_found = True

    if date_found:
        return f'{month}-{date}-{year}-{day}'
    return f'{month}-{date}-{year}-{day}RERER'


def get_hour(frame):
    cropped_frame = crop_frame(frame, vertices[3])
    cv2.imshow('fr', cropped_frame)
    result = reader.readtext(cropped_frame, allowlist='0123456789')
    for (bbox, text, prob) in result:
        if 23 >= int(text) >= 0:
            return int(text)
    return False

def get_cropped_image(frame, coordinate):
    print(coordinate)
    return frame[coordinate[1]:coordinate[3], coordinate[0]: coordinate[2]]


def pre_process(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lwr = np.array([0, 0, 0])
    upr = np.array([255, 255, 255])
    msk = cv2.inRange(hsv, lwr, upr)
    thr = 255 - cv2.bitwise_and(frame, frame, mask=msk)
    crp_img = cv2.cvtColor(thr, cv2.COLOR_HSV2BGR)
    crp_gry = cv2.cvtColor(crp_img, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(crp_gry, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 71, 100)
    return thr


def crop_frame(frame, vertices):
    mask = np.zeros_like(frame)
    roi_corners = np.array([vertices], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, (255, 255, 255))
    return cv2.bitwise_and(frame, mask)


def find_date_through_easyocr(frame):
    print(find_date(frame))


def find_date_thorugh_tessaract(frame):
    predictions = []
    images = []
    for i in range(0, 6):
        cropped_frame = crop_frame(frame, vertices[i])
        pre_processed_img = pre_process(cropped_frame)
        images.append(pre_processed_img)
        predictions.append(pytesseract.image_to_string(pre_processed_img, config="digits"))
    print(predictions)
    for i in range(6):
        cv2.namedWindow(f'thr{i}', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'thr{i}', (1024, 720))
        cv2.imshow(f'thr{i}', images[i])


def run(args):
    # Open video file
    cap = cv2.VideoCapture(args.source)
    # Get video dimensions
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))q
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    i = 0
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
        if i % int(int(fps) / int(fps / 2)) == 0:
            find_date_through_easyocr(frame)
            print(get_hour(frame))
        i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release resources
    cap.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # filenames = ['ch02_20230330090000.mp4', 'ch02_20230330113656.mp4', 'ch02_20230330162911.mp4' ,
    #               'ch02_20230330165845.mp4', 'ch02_20230330211521.mp4']
    filenames = ['ch02_20230330165845.mp4']
    for file in filenames:
        parser = argparse.ArgumentParser()
        # parser.add_argument('--source', type=str, default=f'../../cctv_ger/{file}', help='path to video file')
        parser.add_argument('--source', type=str, default=f'../sample_videos/video2.mp4', help='path to video file')
        parser.add_argument('--weights', nargs='+', type=str,
                            default=r"/home/kamran/tkbees/Project_Person_Counting/yolov5/Yolov5l_Fine_Tuned.pt",
                            help='path to YOLO weights file')
        opt = parser.parse_args()
        os.environ['TESSDATA_PREFIX'] = '/home/kamran/miniconda3/envs/yolov8/share/'
        run(opt)
