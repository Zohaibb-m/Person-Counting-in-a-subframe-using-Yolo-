import argparse
import cv2
import numpy as np
import pytesseract
import os

vertices = [
    [
    (35, 70),
    (84, 70),
    (84, 107),
    (35, 107)

], [
    (101, 70),
    (149, 70),
    (149, 107),
    (101, 107)
], [
    (170, 70),
    (259, 70),
    (259, 107),
    (170, 107)
], [
    (272, 70),
    (346, 70),
    (346, 107),
    (272, 107)
], [
    (362, 73),
    (362, 105),
    (408, 105),
    (408, 73)
], [
    (424, 70),
    (472, 70),
    (472, 107),
    (424, 107)
],  [
    (493, 70),
    (540, 70),
    (540, 107),
    (493, 107)
]
]

# coordinates = [[35, 70, 84, 107], [101, 70, 149, 107], [170, 70, 259, 107], [272, 70, 346, 107], [361, 70, 411, 107],
#                [424, 70, 472, 107], [493, 70, 540, 107]]

last_five_results = []
sec = 56
mint = 38
hour = 21


def get_cropped_image(frame, coordinate):
    # print(frame.shape)
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


# def find_date(frame):
#     predictions = []
#     images = []
#     for i in range(0, 7):
#         cropped_img = get_cropped_image(frame, coordinates[i])
#         # pre_proceesed_img = pre_process(cropped_img)
#         images.append(cropped_img)
#         # cv2.imshow('thr', cropped_img)
#         predictions.append(pytesseract.image_to_string(cropped_img))
#     print(predictions)
#     # cv2.imshow('thr', images[0])
#     # cv2.imshow('thr1', images[1])
#     # cv2.imshow('thr2', images[2])
#     # cv2.imshow('thr3', images[3])
#     # cv2.imshow('thr4', images[4])
#     # cv2.imshow('thr5', images[5])
#     # cv2.imshow('thr6', images[6])


def crop_frame(frame, vertices):
    mask = np.zeros_like(frame)
    roi_corners = np.array([vertices], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, (255, 255, 255))
    return cv2.bitwise_and(frame, mask)


def find_date(frame):
    predictions = []
    images = []
    for i in range(0, 7):
        cropped_frame = crop_frame(frame, vertices[i])
        pre_processed_img = pre_process(cropped_frame)
        images.append(pre_processed_img)
        predictions.append(pytesseract.image_to_string(pre_processed_img))
    print(predictions)
    for i in range(7):
        cv2.namedWindow(f'thr{i}', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'thr{i}', (1024,720))
        cv2.imshow(f'thr{i}', images[i])

def run(args):
    # Open video file
    cap = cv2.VideoCapture(args.source)
    # Get video dimensions
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
        if i % int(fps) / int(fps / 2) == 0:
            find_date(frame)
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
        parser.add_argument('--source', type=str, default=f'../sample_videos/sample4.mp4', help='path to video file')
        parser.add_argument('--weights', nargs='+', type=str,
                            default=r"/home/kamran/tkbees/Project_Person_Counting/yolov5/Yolov5l_Fine_Tuned.pt",
                            help='path to YOLO weights file')
        opt = parser.parse_args()
        os.environ['TESSDATA_PREFIX'] = '/home/kamran/miniconda3/envs/yolov8/share/'
        run(opt)
