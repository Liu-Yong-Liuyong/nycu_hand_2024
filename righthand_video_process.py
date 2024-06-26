# mediapipe只能辨識各個手指頭但不能辨識左右手
import cv2
import csv
import mediapipe as mp
import pyautogui
import time
import numpy as np
import threading
import pygame.mixer
import pygame.time
import pafy
import youtube_dl
import numpy as np
import math
import matplotlib.pyplot as plt
import os

outer_joint_list = [[8, 7, 6], [12, 11, 10], [16, 15, 14], [20, 19, 18]]
middle_joint_list = [[4, 3, 2], [7, 6, 5], [11, 10, 9], [15, 14, 13], [19, 18, 17]]
inner_joint_list = [[3, 2, 1], [6, 5, 0], [10, 9, 0], [14, 13, 0], [18, 17, 0]]
hands = mp.solutions.hands
outer_finger_angles = {}
inner_finger_angles = {}
middle_finger_angles = {}
def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    cosine_theta = dot_product / (norm_v1 * norm_v2)
    angle_in_radians = np.arccos(np.clip(cosine_theta, -1.0, 1.0))
    angle_in_degrees = np.degrees(angle_in_radians)

    return angle_in_degrees

def draw_finger_angles(image, results):
    # Loop through hands
    for hand in results.multi_hand_landmarks:
        # Loop through joint sets
        for finger_index, joint in enumerate(outer_joint_list):
            temp = 0
            finger_name = f"Finger_{finger_index + 2}"
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y, hand.landmark[joint[0]].z])  # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y, hand.landmark[joint[1]].z])  # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y, hand.landmark[joint[2]].z])  # Third coord
            v1 = a - b
            v2 = b - c
            angle = calculate_angle(v1, v2)
            radians = abs(np.arctan2(c[1] - b[1], c[2] - b[2])) + abs(np.arctan2(a[1] - b[1], a[2] - b[2]))
            #####
            if np.arctan2(c[1] - b[1], c[2] - b[2])<0 and np.arctan2(a[1] - b[1], a[2] - b[2])<0:
                radians = abs(abs(np.arctan2(c[1] - b[1], c[2] - b[2])) - abs(np.arctan2(a[1] - b[1], a[2] - b[2])))
                temp = 1
            #####
          
            x_y_angle = np.abs(radians * 180.0 / np.pi)
           
            if c[1] > a[1] and temp == 0: 
                x_y_angle = x_y_angle-180.0
            else:
                x_y_angle = 180.0-x_y_angle
            #####
            if x_y_angle < 0 and abs(abs(x_y_angle)-angle) < 1:
                angle = -angle
            if finger_name not in outer_finger_angles:
                outer_finger_angles[finger_name] = []
            outer_finger_angles[finger_name].append(angle)

            x, y = int(b[0] * image.shape[1]), int(b[1] * image.shape[0])
            cv2.putText(image, str(round(angle, 2)), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        for finger_index, joint in enumerate(middle_joint_list):
            #####
            temp = 0
            #####
            finger_name = f"Finger_{finger_index + 1}"
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y, hand.landmark[joint[0]].z])  # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y, hand.landmark[joint[1]].z])  # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y, hand.landmark[joint[2]].z])  # Third coord
            v1 = a - b
            v2 = b - c
            angle = calculate_angle(v1, v2)
            if finger_name == "Finger_1":
                radians = abs(np.arctan2(c[1] - b[1], c[0] - b[0])) + abs(np.arctan2(a[1] - b[1], a[0] - b[0]))
                #####
                if np.arctan2(c[1] - b[1], c[0] - b[0])<0 and np.arctan2(a[1] - b[1], a[0] - b[0])<0:
                    radians = abs(abs(np.arctan2(c[1] - b[1], c[0] - b[0])) - abs(np.arctan2(a[1] - b[1], a[0] - b[0])))
                    temp = 1
                #####
            else:
                radians = abs(np.arctan2(c[1] - b[1], c[2] - b[2])) + abs(np.arctan2(a[1] - b[1], a[2] - b[2]))
                #####
                if np.arctan2(c[1] - b[1], c[2] - b[2])<0 and np.arctan2(a[1] - b[1], a[2] - b[2])<0:
                    radians = abs(abs(np.arctan2(c[1] - b[1], c[2] - b[2])) - abs(np.arctan2(a[1] - b[1], a[2] - b[2])))
                    temp = 1
                #####
            x_y_angle = np.abs(radians * 180.0 / np.pi)
            
            #都是負號/\
            
            if finger_name == "Finger_1":
                if c[1] < a[1] and temp == 0: 
                    x_y_angle = 180.0-x_y_angle
                else:
                    x_y_angle = x_y_angle-180.0
            else:
                if c[1] > a[1] and temp == 0: 
                    x_y_angle = x_y_angle-180.0
                else:
                    x_y_angle = 180.0-x_y_angle
         
            if x_y_angle < 0 and abs(abs(x_y_angle) - angle) < 1:
                angle = -angle
            if finger_name not in middle_finger_angles:
                middle_finger_angles[finger_name] = []
            middle_finger_angles[finger_name].append(angle)

            x, y = int(b[0] * image.shape[1]), int(b[1] * image.shape[0])
            cv2.putText(image, str(round(x_y_angle, 2)), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        for finger_index, joint in enumerate(inner_joint_list):
            temp = 0
            finger_name = f"Finger_{finger_index + 1}"
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y, hand.landmark[joint[0]].z])  # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y, hand.landmark[joint[1]].z])  # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y, hand.landmark[joint[2]].z])  # Third coord
            v1 = a - b
            v2 = b - c
            angle = calculate_angle(v1, v2)
            if finger_name == "Finger_1":
                radians = abs(np.arctan2(c[1] - b[1], c[0] - b[0])) + abs(np.arctan2(a[1] - b[1], a[0] - b[0]))
                #####
                if np.arctan2(c[1] - b[1], c[0] - b[0])<0 and np.arctan2(a[1] - b[1], a[0] - b[0])<0:
                    radians = abs(abs(np.arctan2(c[1] - b[1], c[0] - b[0])) - abs(np.arctan2(a[1] - b[1], a[0] - b[0])))
                    temp = 1
                #####
              
            else:
                radians = abs(np.arctan2(c[1] - b[1], c[2] - b[2])) + abs(np.arctan2(a[1] - b[1], a[2] - b[2]))
                #####
                if np.arctan2(c[1] - b[1], c[2] - b[2])<0 and np.arctan2(a[1] - b[1], a[2] - b[2])<0:
                    radians = abs(abs(np.arctan2(c[1] - b[1], c[2] - b[2])) - abs(np.arctan2(a[1] - b[1], a[2] - b[2])))
                    temp = 1
                #####
          
            x_y_angle = np.abs(radians * 180.0 / np.pi)
            if finger_name == "Finger_1":
                if c[1] < a[1] and temp == 0: 
                    x_y_angle = 180.0-x_y_angle
                else:
                    x_y_angle = x_y_angle-180.0
            else:
                if c[1] > a[1] and temp == 0: 
                    x_y_angle = x_y_angle-180.0
                else:
                    x_y_angle = 180.0-x_y_angle
        
            if x_y_angle < 0 and abs(abs(x_y_angle) - angle) < 1:
                angle = -angle
            if finger_name not in inner_finger_angles:
                inner_finger_angles[finger_name] = []
            inner_finger_angles[finger_name].append(angle)

            x, y = int(b[0] * image.shape[1]), int(b[1] * image.shape[0])
            cv2.putText(image, str(round(angle, 2)), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def save_to_csv(filename, file):
    # Initialize a list to store the data
    data = []

    # Iterate over each finger
    for finger_name, finger_values in file.items():
        # Calculate minimum and maximum values for the finger
        min_value = round(min(finger_values),3)
        max_value = round(max(finger_values),3)
        
        # Append the finger name, minimum, and maximum values to the data list
        data.append([finger_name, min_value, max_value])

    # Write the data to the CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['Finger', 'Minimum', 'Maximum'])
        # Write data rows
        writer.writerows(data)



# Get the directory where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the video file relative to the base directory
video_path = os.path.join(base_dir, 'static', 'user_upload_video.mp4')
drawing = mp.solutions.drawing_utils

# Now you can use video_path to open the video file
cap = cv2.VideoCapture(video_path)
drawing = mp.solutions.drawing_utils

hand_obj = hands.Hands(max_num_hands=1)

start_init = False
prev = -1
mp_drawing = mp.solutions.drawing_utils
left_hand = [0, 0, 0]
right_hand = [0, 0, 0]
draw = 0
while True:
    end_time = time.time()
    _, frm = cap.read()
    if frm is None:
        break
    #frm = cv2.flip(frm, 1)
    res = hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    if res.multi_hand_landmarks:
        change = 0
        for num, hand_keyPoints in enumerate(res.multi_hand_landmarks):
            #cnt = count_fingers(hand_keyPoints)
            mp_drawing.draw_landmarks(frm, hand_keyPoints, hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                      )
        if draw == 0: 
            draw_finger_angles(frm, res) 
            draw = draw + 1
        elif draw == 7:
            draw = 0
        else:
            draw = draw + 1
        #draw_finger_angles(frm, res)

    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window", 1000, 1000)
    cv2.imshow("Window", frm)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break
    np.save('outer_finger_angles_website.npy', np.array(outer_finger_angles))
    np.save('middle_finger_angles_website.npy', np.array(middle_finger_angles))
    np.save('inner_finger_angles_website.npy', np.array(inner_finger_angles))

    save_to_csv('o1.csv', outer_finger_angles)
    save_to_csv('m1.csv', middle_finger_angles)
    save_to_csv('i1.csv', inner_finger_angles)


