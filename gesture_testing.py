from imutils import face_utils
#  from helper import *
import numpy as np
import pyautogui as pag
import imutils
import dlib
import cv2
import math

MOUTH_AR_THRESH = 0.3
MOUTH_AR_CONSECUTIVE_FRAMES = 15
EYE_AR_THRESH = 0.19
EYE_AR_CONSECUTIVE_FRAMES = 10
WINK_AR_DIFF_THRESH = 0.04
WINK_AR_CLOSE_THRESH = 0.19
WINK_CONSECUTIVE_FRAMES = 10

DEGREE_COUNTER = 0
MOUTH_COUNTER = 0
EYE_COUNTER = 0
WINK_COUNTER = 0
INPUT_MODE = False
EYE_CLICK = False
LEFT_WINK = False
RIGHT_WINK = False
SCROLL_MODE = False
ANCHOR_POINT = (0, 0)
WHITE_COLOR = (255, 255, 255)
YELLOW_COLOR = (0, 255, 255)
RED_COLOR = (200, 10, 10)
GREEN_COLOR = (0, 200, 200)
BLUE_COLOR = (255, 0, 0)
BLACK_COLOR = (0, 0, 0)

shape_predictor = "model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Video capture
vid = cv2.VideoCapture(0)
resolution_w = 1366
resolution_h = 768
cam_w = 640
cam_h = 480
unit_w = resolution_w / cam_w
unit_h = resolution_h / cam_h

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # Return the eye aspect ratio
    return ear

def mouth_aspect_ratio(mouth):
    # Compute the euclidean distances between the three sets
    # of vertical mouth landmarks (x, y)-coordinates
    A = np.linalg.norm(mouth[13] - mouth[19])
    B = np.linalg.norm(mouth[14] - mouth[18])
    C = np.linalg.norm(mouth[15] - mouth[17])
    # Compute the euclidean distance between the horizontal
    # mouth landmarks (x, y)-coordinates
    D = np.linalg.norm(mouth[12] - mouth[16])
    # Compute the mouth aspect ratio
    mar = (A + B + C) / (2 * D)
    # Return the mouth aspect ratio
    return mar


# Return direction given the nose and anchor points.
def direction(nose_point, anchor_point, w, h, multiple=1):
    nx, ny = nose_point
    x, y = anchor_point
    if nx > x + multiple * w:
        return 'right'
    elif nx < x - multiple * w:
        return 'left'
    if ny > y + multiple * h:
        return 'down'
    elif ny < y - multiple * h:
        return 'up'
    return '-'


def setAnchor(nx,ny):
    global ANCHOR_POINT
    ANCHOR_POINT = ( nx, ny )

while True:
    
    _, frame = vid.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=cam_w, height=cam_h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Loop over the face detections
    if len(rects) > 0:
        rect = rects[0]
    else:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        continue

    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # Find left and right eye coordinate
    mouth = shape[mStart:mEnd]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    nose = shape[nStart:nEnd]
    nose_line = shape[27:31]

    # left is right, right is left.
    temp = leftEye
    leftEye = rightEye
    rightEye = temp
    mar = mouth_aspect_ratio(mouth)
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    diff_ear = np.abs(leftEAR - rightEAR)

    nose_point = (nose[3, 0], nose[3, 1])

    
    mouthHull = cv2.convexHull(mouth)
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [mouthHull], -1, (0,0,255), 1)
    cv2.drawContours(frame, [leftEyeHull], -1, (0,0,255), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0,0,255), 1)

    for (x, y) in np.concatenate((mouth, leftEye, rightEye,nose_line), axis=0):
        cv2.circle(frame, (x, y), 2, GREEN_COLOR, -1)
 
    radians = math.atan2(nose_line[0][1]-nose_line[3][1], nose_line[0][0]-nose_line[3][0] )
    degrees = -1*math.degrees(radians) 
    # print(degrees)

    # Right click or left click
    # done !!!!!!!!!!
    if (not INPUT_MODE ) or (not SCROLL_MODE) :
        if degrees<=66:
            if DEGREE_COUNTER>=5 :
                pag.click(button='right')
                print("Right Winking!")
                DEGREE_COUNTER = 0
            else :
                DEGREE_COUNTER+=1
        elif degrees>=96:
            if DEGREE_COUNTER>=5 :
                pag.click(button='left')
                print("left Winking!")
                DEGREE_COUNTER = 0
            else :
                DEGREE_COUNTER+=1
        else:
            DEGREE_COUNTER=0

       # if diff_ear > WINK_AR_DIFF_THRESH:       
    #     if leftEAR < rightEAR:
    #         if leftEAR < EYE_AR_THRESH:
    #             WINK_COUNTER += 1
    #             if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
    #                 pag.click(button='left')
    #                 WINK_COUNTER = 0
    #     elif leftEAR > rightEAR:
    #         if rightEAR < EYE_AR_THRESH:
    #             WINK_COUNTER += 1
    #             if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
    #                 pag.click(button='right')
    #                 WINK_COUNTER = 0
    #     else:
    #         WINK_COUNTER = 0
    # else:

    
    # Scrolling Mode !!!!
    # done!!!
    if not INPUT_MODE:
        if ear <= EYE_AR_THRESH:
            EYE_COUNTER += 1
            print('Eyes_closed_once')
            if EYE_COUNTER > EYE_AR_CONSECUTIVE_FRAMES:
                SCROLL_MODE = not SCROLL_MODE
                EYE_COUNTER = 0
                print('Scrolling_mode_Achieved!!!')                
        else:
            EYE_COUNTER = 0
        # WINK_COUNTER = 0


    # mouth opening
    # done!!!!!
    if not SCROLL_MODE:
        if mar > MOUTH_AR_THRESH:
            MOUTH_COUNTER += 1

            if MOUTH_COUNTER >= MOUTH_AR_CONSECUTIVE_FRAMES:
                # if the alarm is not on, turn it on
                INPUT_MODE = not INPUT_MODE
                MOUTH_COUNTER = 0
                ANCHOR_POINT = nose_point
        else:
            MOUTH_COUNTER = 0


    # movement for mouse
    if INPUT_MODE:
        cv2.putText(frame, "TAKING INPUT", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED_COLOR, 2)
        x, y =ANCHOR_POINT
        nx, ny = nose_point
        w, h = 60, 35
        multiple = 1
        cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), GREEN_COLOR, 2)
        cv2.line(frame, ANCHOR_POINT, nose_point, BLUE_COLOR, 2)

        dir = direction(nose_point, ANCHOR_POINT, w, h)
        cv2.putText(frame, dir.upper(), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
        
        drag = 25
        if dir == 'right':
            pag.moveRel(drag, 0)
        elif dir == 'left':
            pag.moveRel(-drag, 0)
        elif dir == 'up':
            if SCROLL_MODE:
                pag.scroll(40)
            else:
                pag.moveRel(0, -drag)
        elif dir == 'down':
            if SCROLL_MODE:
                pag.scroll(-40)
            else:
                pag.moveRel(0, drag)
        # setAnchor(nx, ny)
        print(ANCHOR_POINT ," ", nose_point)
        
    # for scrolling
    if SCROLL_MODE:
        cv2.putText(frame, 'SCROLLING', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the `Esc` key was pressed, break from the loop
    if key == 27:
        break


cv2.destroyAllWindows()
vid.release()
