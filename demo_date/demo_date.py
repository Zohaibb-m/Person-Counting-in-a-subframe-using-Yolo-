import argparse
import time

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
        (103, 70),
        (148, 70),
        (148, 107),
        (103, 107)
    ],
    [
        (170, 70),
        (255, 70),
        (255, 107),
        (170, 107)
    ],
    # [     (272, 70),
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
        (428, 70),
        (478, 70),
        (478, 107),
        (428, 107)
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

coordinates = [[35, 70, 84, 107], [101, 70, 149, 103], [170, 70, 259, 107], [272, 70, 346, 107], [361, 70, 411, 107],
               [427, 70, 478, 107], [493, 70, 544, 103]]

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
save_counter = 0

import copy


def detect_easy_ocr(frame):
    global save_counter
    result = reader.readtext(frame, allowlist='0123456789')
    for (bbox, text, prob) in result:
        if text is not None:
            print(text)
            return int(text)
    print('Preprocessing')
    frame_new = preprocess_image(copy.deepcopy(frame))
    result = reader.readtext(frame_new, allowlist='0123456789')
    for (bbox, text, prob) in result:
        if text is not None:
            return int(text)
    print("Preprocessing 2")
    frame_mix = preprocess_image_mix(copy.deepcopy(frame))
    result = reader.readtext(frame_mix, allowlist='0123456789')
    for (bbox, text, prob) in result:
        if text is not None:
            return int(text)
    cv2.imwrite(f'./failed_images/{save_counter}_failed_image.jpg', frame)
    cv2.imwrite(f'./failed_images/{save_counter}_failed_image_mix.jpg', frame_mix)
    cv2.imwrite(f'./failed_images/{save_counter}_failed_image_new.jpg', frame_new)
    save_counter += 1
    return False


def find_date(frame, crop, show):
    global date, month, year, day, date_found
    if not date:
        if crop:
            cropped_frame = get_cropped_image(frame, coordinates[1])
        else:
            cropped_frame = crop_frame(frame, vertices[1])
        if show:
            cv2.imshow('date', cropped_frame)
        result = detect_easy_ocr(cropped_frame)
        if 31 >= result > 0:
            date = result
    if not month:
        if crop:
            cropped_frame = get_cropped_image(frame, coordinates[0])
        else:
            cropped_frame = crop_frame(frame, vertices[0])
        if show:
            cv2.imshow('month', cropped_frame)
        result = detect_easy_ocr(cropped_frame)
        if 12 >= result > 0:
            month = result
    if not year:
        if crop:
            cropped_frame = get_cropped_image(frame, coordinates[2])
        else:
            cropped_frame = crop_frame(frame, vertices[2])
        if show:
            cv2.imshow('year', cropped_frame)
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
    return month, date, year, day


def get_hour(frame, crop, show):
    if crop:
        cropped_frame = get_cropped_image(frame, coordinates[4])
    else:
        cropped_frame = crop_frame(frame, vertices[3])
    if show:
        cv2.imshow('hour', cropped_frame)
    result = detect_easy_ocr(cropped_frame)
    if 23 >= result >= 0:
        return result
    return False


def get_min(frame, crop, show):
    if crop:
        cropped_frame = get_cropped_image(frame, coordinates[5])
    else:
        cropped_frame = crop_frame(frame, vertices[4])
    if show:
        cv2.imshow('min', cropped_frame)
    result = detect_easy_ocr(cropped_frame)
    if result:
        if 60 >= result >= 0:
            return result
    return False


def get_second(frame, crop, show):
    if crop:
        cropped_frame = get_cropped_image(frame, coordinates[6])
    else:
        cropped_frame = crop_frame(frame, vertices[5])
    if show:
        cv2.imshow('sec', cropped_frame)
    result = detect_easy_ocr(cropped_frame)
    if result:
        if 60 >= result >= 0:
            return result
    return False


def preprocess_image_mix(image):
    val = 230
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Create a black canvas according to the image size
    canvas = np.zeros_like(gray)

    # Invert the image
    inverted = cv2.bitwise_not(gray)

    # Find white pixels with intensity value greater than 245 in the original image
    original_white_pixels = np.where(gray > val)
    # Find white pixels with intensity value greater than 245 in the inverted image
    inverted_white_pixels = np.where(inverted > val)

    # Place white pixels onto the black canvas
    canvas[original_white_pixels] = 255
    canvas[inverted_white_pixels] = 255

    return canvas


def get_cropped_image(frame, coordinate):
    # print(coordinate)
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


def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Invert the image
    inverted = cv2.bitwise_not(gray)

    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(inverted)

    # cv2.imshow("inverted", inverted)
    # cv2.imshow("equalized", equalized)

    return equalized


def crop_frame(frame, vertices):
    mask = np.zeros_like(frame)
    roi_corners = np.array([vertices], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, (255, 255, 255))
    return cv2.bitwise_and(frame, mask)


def find_date_through_easyocr(frame, crop, show):
    print(find_date(frame, crop, show))


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


def run(crop, show):
    # Open video file
    file = open('Logs.csv', 'w')
    for inn in range(1):
        print('Hi')
        # cap = cv2.VideoCapture(f'../../cctv_ger/ch03_0{inn + 1}.mp4')
        cap = cv2.VideoCapture(f'../sample_videos/sample4.mp4')
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
                find_date_through_easyocr(frame, crop, show)
                # file.write(f'{month},{date},{year},{day},')
                # file.write(f'{get_hour(frame)},{get_min(frame)},{get_second(frame)}\n')
                print(f'Hour: {get_hour(frame, crop, show)}')
                print(f'Min: {get_min(frame, crop, show)}')
                print(f'Sec: {get_second(frame, crop, show)}')
            i += 1
            if i % 1000 == 0:
                print(i)
            # cv2.waitKey(0)
            # time.sleep(4)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Release resources
        cap.release()
        # out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    os.environ['TESSDATA_PREFIX'] = '/home/kamran/miniconda3/envs/yolov8/share/'
    run(crop=True, show=True)
