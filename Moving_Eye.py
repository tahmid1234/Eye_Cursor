import sys
import cv2
import numpy as np
import dlib
import array
import time
from math import hypot

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def get_colored_eye(eye_points, facial_landmarks, frame):
    eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part((eye_points[0])).y),
                           (facial_landmarks.part((eye_points[1])).x, facial_landmarks.part((eye_points[1])).y),
                           (facial_landmarks.part((eye_points[2])).x, facial_landmarks.part((eye_points[2])).y),
                           (facial_landmarks.part((eye_points[3])).x, facial_landmarks.part((eye_points[3])).y),
                           (facial_landmarks.part((eye_points[4])).x, facial_landmarks.part((eye_points[4])).y),
                           (facial_landmarks.part((eye_points[5])).x, facial_landmarks.part((eye_points[5])).y)],
                          np.int32)
    cv2.polylines(frame, [eye_region], True, 255, 2)
    cv2.fillPoly(frame, [eye_region], (100, 0, 0))


def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midPoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midPoint(facial_landmarks.part(eye_points[4]), facial_landmarks.part(eye_points[5]))
    hor_line_len = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_len = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    return hor_line_len / ver_line_len


def midPoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def eyeDetection(p1, p2, p3, p4, p5, p6, frame):
    left_point = (p1.x, p1.y)
    right_point = (p2.x, p2.y)
    center_top = midPoint(p3, p4)
    center_bottom = midPoint(p5, p6)
    hor_line = cv2.line(frame, left_point, right_point, (255, 0, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (255, 0, 0), 2)


x = 0


# def getUpDownGAze(landmarks ):


def get_gaze_ration(eye_points, facial_landmarks, frame, gray):
    eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part((eye_points[0])).y),
                           (facial_landmarks.part((eye_points[1])).x, facial_landmarks.part((eye_points[1])).y),
                           (facial_landmarks.part((eye_points[2])).x, facial_landmarks.part((eye_points[2])).y),
                           (facial_landmarks.part((eye_points[3])).x, facial_landmarks.part((eye_points[3])).y),
                           (facial_landmarks.part((eye_points[4])).x, facial_landmarks.part((eye_points[4])).y),
                           (facial_landmarks.part((eye_points[5])).x, facial_landmarks.part((eye_points[5])).y)],
                          np.int32)
    # cv2.polylines(frame,[left_eye_region],True,(50,255,90),2)
    # print(len(left_eye_region))
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    left_eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])
    # print(min_x, max_x, min_y, max_y)

    gray_eye = left_eye[min_y:max_y, min_x:max_x]
    # gray_eye=cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)
    # the second argument depends on device camera .
    _, threshold_eye = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape

    left_side_threshold = threshold_eye[0:height, 0:int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[0:height, int(width / 2):width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    up_side_threshold = threshold_eye[0:int(height / 2), 0:width]
    up_side_white = cv2.countNonZero(up_side_threshold)
    down_side_threshold = threshold_eye[int(height / 2):height, 0:width]
    down_side_white = cv2.countNonZero(down_side_threshold)

    if (right_side_white == 0):
        gaze_side_ratio = 5
    elif (left_side_white == 0):
        gaze_side_ratio = 1
    else:
        gaze_side_ratio = left_side_white / right_side_white

    # cv2.putText(frame, str(gaze_up_down), (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    return gaze_side_ratio, down_side_white


x0 = 0
y0 = 0

x1 = 0
y1 = 0
x2 = 0
y2 = 0
x3 = 0
y3 = 0
x4 = 0
y4 = 0
x5 = 0
y5 = 0
c = 9


def true():
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame,(x,y),(x1,y1),(255,0,0),5)
        landmarks = predictor(gray, face)
        # eyeDetection(landmarks.part(36),landmarks.part(39),landmarks.part(37),landmarks.part(38),landmarks.part(40),landmarks.part(41))
        # eyeDetection(landmarks.part(42),landmarks.part(45),landmarks.part(43),landmarks.part(44),landmarks.part(46),landmarks.part(47))

        # gaze detection
        gaze_left_eye_side_ratio, gaze_left_eye_topDown_ratio = get_gaze_ration([36, 37, 38, 39, 40, 41], landmarks,
                                                                                frame, gray)
        gaze_right_eye_side_ratio, gaze_right_eye_topDown_ratio = get_gaze_ration([42, 43, 44, 45, 46, 47], landmarks,
                                                                                  frame, gray)

        side_gaze_ratio = (gaze_left_eye_side_ratio + gaze_right_eye_side_ratio) / 2
        upDOwn_gaze_ratio = (gaze_left_eye_topDown_ratio + gaze_right_eye_topDown_ratio) / 2

        if upDOwn_gaze_ratio <= 44:
            cv2.putText(frame, "down", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 6)
        elif 44 < upDOwn_gaze_ratio < 109:
            cv2.putText(frame, "center", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 6)
        else:
            cv2.putText(frame, "up", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 6)

        # cv2.putText(frame, str(upDOwn_gaze_ratio), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)

        cv2.putText(frame, "X", (270, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)

    # if side_gaze_ratio<=1:
    #    cv2.putText(frame,"right", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 6)
    # elif 1<side_gaze_ratio<=1.85:
    #   cv2.putText(frame, "CENTER", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 6)
    # else:
    #   cv2.putText(frame, "left", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 6)

    # eye=cv2.resize(gray_eye,None,fx=5,fy=5)
    # threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)

    # cv2.imshow("left_eye", left_eye)
    # cv2.imshow("left",left_side_threshold)
    # cv2.imshow("right", right_side_threshold)
    # cv2.imshow("up", up_side_threshold)
    # cv2.imshow("down", down_side_threshold)
    # cv2.imshow("Thereshol EYE",threshold_eye)

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(10)
    if key == 27:
        return
        # break

cap.release()
cv2.destroyAllWindows()
