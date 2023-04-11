import argparse
import cv2
from ultralytics import YOLO
import numpy as np

vertices = [
    (62,239),
    (811,1075),
    (1213,691),
    (254,149)
]

def process_frame(frame, model, iou_thresh, draw_bounding_box):
    # Crop frame
    # height, width, color = frame.shape
    # X1, Y1, X2, Y2 = [width//4,height//4, 3*width//4, 3*height//4]
    # cropped_frame = frame[Y1:Y2,X1:X2,:]
    mask = np.zeros_like(frame)
    # print(frame.shape)
    roi_corners = np.array([vertices], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, (255, 255, 255))
    cropped_frame = cv2.bitwise_and(frame, mask)
    # Detect people using YOLO
    result = model(cropped_frame, mode='predict', classes=[0], iou=iou_thresh, hide_conf=True, save=True, conf = 0.1 )

    # Count number of people detected
    num_people = len(result[0].boxes)

    # Draw bounding boxes on original frame
    if draw_bounding_box:
        for i, det in enumerate(result[0].boxes):
            res = det.xyxy
            x1, y1, x2, y2 = det.xyxy[0].tolist()
            # x1 += X1
            # y1 += Y1
            # x2 += X1
            # y2 += Y1
            label = f"Person {i+1}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            # cv2.rectangle(frame, (int(X1), int(Y1)), (int(X2), int(Y2)), (255, 0, 255), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display number of people detected
    cv2.polylines(frame, [roi_corners], True, (0, 255, 0), thickness=2)
    cv2.putText(frame, f"No. of people detected at the left side: {num_people}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)

    return frame


def run(args, draw_bounding_box):
    # Load YOLO model
    model = YOLO(args.weights)

    # Open video file
    cap = cv2.VideoCapture(args.source)

    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create video writer object
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        processed_frame = process_frame(frame, model, args.iou_thresh, draw_bounding_box)
        cv2.imwrite("sample.jpg", processed_frame)
        # Write processed frame to output video
        out.write(processed_frame)

        # Display processed frame
        # cv2.imshow('Processed Frame', processed_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='sample_videos/sample4.mp4', help='path to video file')
    parser.add_argument('--weights', nargs='+', type=str, default=r"models/detection/yolov5l.pt",
                        help='path to YOLO weights file')
    parser.add_argument('--iou-thresh', type=float, default=0.5, help='IOU threshold for NMS')
    # parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--is_save', type=bool, default=True, help='do not save images/videos')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    opt = parser.parse_args()

    run(opt, draw_bounding_box = True)
