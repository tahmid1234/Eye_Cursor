import Moving_Eye as g
import sys
import cv2
import numpy as np
import dlib
import array
import time
import tkinter as TK
import turtle
from pynput.mouse import Button,Controller

import random
mouse_x=0
mouse_y=0
click=1
cap = cv2.VideoCapture(0)
wn = turtle.Screen()
wn.bgcolor("white")
wn.title("Bouncing Ball")
wn.tracer(0)
ball = turtle.Turtle()
ball.shape("circle")
ball.penup()
ball.speed(5)
ball.goto(0, 200)
ball.dy = 0
ball.dx = 2
gravity = -.3


mouse=Controller()
mouse.position=(1500,200)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
state = "state"
shapes = ["circle", "triangle", "square"]


while True:
    _, frame = cap.read()
    wn.update()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    ball.dy += gravity
    ball.sety(ball.ycor() + ball.dy)
    ball.setx(ball.xcor() + ball.dx)
    mouse.move(mouse_x,mouse_y)
    mouse_x = 0
    mouse_y = 0
    right_side_upperBound = 1.2
    center_upperBound = 3.2
    # print(ball.ycor())
    # print(gravity)
    # if ball.ycor()<-300:
    # ball.dy*=-1
    # print("break")
    print("xCOR + xy")
    print(ball.xcor(), ball.dx)

    print("Outter face")
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame,(x,y),(x1,y1),(255,0,0),5)
        landmarks = predictor(gray, face)
        # eyeDetection(landmarks.part(36),landmarks.part(39),landmarks.part(37),landmarks.part(38),landmarks.part(40),landmarks.part(41))
        # eyeDetection(landmarks.part(42),landmarks.part(45),landmarks.part(43),landmarks.part(44),landmarks.part(46),landmarks.part(47))

        # gaze detection
        g.get_colored_eye([36, 37, 38, 39, 40, 41], landmarks, frame)
        g.get_colored_eye([42, 43, 44, 45, 46, 47], landmarks, frame)
        gaze_left_eye_side_ratio, gaze_left_eye_topDown_ratio = g.get_gaze_ration([36, 37, 38, 39, 40, 41], landmarks,
                                                                                  frame, gray)
        gaze_right_eye_side_ratio, gaze_right_eye_topDown_ratio = g.get_gaze_ration([42, 43, 44, 45, 46, 47], landmarks,
                                                                                    frame, gray)
        blinking_left_eye_side_ratio = g.get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        blinking_right_eye_side_ratio = g.get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (blinking_left_eye_side_ratio + blinking_right_eye_side_ratio) / 2

        side_gaze_ratio = (gaze_left_eye_side_ratio + gaze_right_eye_side_ratio) / 2
        upDOwn_gaze_ratio = (gaze_left_eye_topDown_ratio + gaze_right_eye_topDown_ratio) / 2
        print("Face")
        if upDOwn_gaze_ratio <= 99.5:
            if state != "DOWN":
                mouse_y=10
                ball.dy = 0
                gravity = -.3
                state = "DOWN"
                wn.bgcolor("white")
                ball.color("black")
                ball.shape(random.choice(shapes))
            cv2.putText(frame, "down", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 6)
            right_side_upperBound = 3.1
            center_upperBound = 5
        # elif 42 < upDOwn_gaze_ratio < 99.5:
        #
        #     cv2.putText(frame, "center", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 6)
        else:
            if state != "UP":
                mouse_y = -10
                ball.dy = 0
                gravity = .3
                state = "UP"
            print(ball.dy)
            print("UP")
            wn.bgcolor("black")
            ball.color("white")
            cv2.putText(frame, "up", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 6)

        if (side_gaze_ratio <= right_side_upperBound):
            ball.dx = 2
            mouse_x=-10
            cv2.putText(frame, "Right", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 6)
        elif right_side_upperBound < side_gaze_ratio < center_upperBound:
            cv2.putText(frame, "side", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 6)
        else:
            mouse_x = 10
            ball.dx = -2
            cv2.putText(frame, "left", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 6)

        if blinking_ratio > 5.85:
            mouse.click(Button.left,click)
            if(click==2):
                click=1
            else:
                click=2
            ball.shape(random.choice(shapes))
            cv2.putText(frame, "blinking", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 6)

        cv2.putText(frame, str(side_gaze_ratio), (270, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

wn.mainloop()
cap.release()
cv2.destroyAllWindows()
print("***************************")
# print(g.check(1,2))
