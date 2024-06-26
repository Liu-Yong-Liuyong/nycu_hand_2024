import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from matplotlib.animation import FuncAnimation,PillowWriter
from threading import Thread
import time
import matplotlib
from datetime import timedelta
import imageio
from itertools import cycle,islice


def init():
    line.set_data([], [])

    return line,

def update(frame,finger_n):

    
    x_values = np.arange(frame+1)
    y_values = outer_finger_angles[finger_n][:frame + 1]
    line.set_data(x_values, y_values)

    line.set_label(f'{finger_n} Angle')

    return line,

def video_thread():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    


def animation_thread(gif_path):
    
    gif_frames = imageio.mimread(gif_path)

    for frame in gif_frames:
        cv2.imshow('GIF Player', frame)
        if cv2.waitKey(50) & 0xFF == 27:  # ESC键退出
            break




def combined_video_thread(gif_path,output_path):
    print("combined video thread")
    gif_frames = imageio.mimread(gif_path)
    gif_frame_count = len(gif_frames)

    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    new_width = int(video_width*450/video_height)

    interval=total_frames//gif_frame_count

    video_frames = islice(iter(cap.read, (False, None)), 0, None, interval)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    file_name = os.path.splitext(gif_path)[0]
    file_name = file_name.replace('/', '_') 
    output_video_name = os.path.basename(file_name) + '_combined_output.avi'
    output_video_path = os.path.join(output_path, output_video_name)
    out = cv2.VideoWriter(output_video_path, fourcc, fps/interval, (new_width+750, 450))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    for frame, gif_frame in zip(range(total_frames), gif_frames):

        ret, video_frame = next(video_frames)
        if not ret:
            break
        
        #new_width = int(video_width*450/video_height)
        video_frame = cv2.resize(video_frame, (new_width,450))


        if gif_frame.shape[2] > video_frame.shape[2]:
            gif_frame = gif_frame[:, :, :video_frame.shape[2]]
        elif gif_frame.shape[2] < video_frame.shape[2]:
            zeros = np.zeros((gif_frame.shape[0], gif_frame.shape[1], video_frame.shape[2] - gif_frame.shape[2]), dtype=gif_frame.dtype)
            gif_frame = np.concatenate((gif_frame, zeros), axis=2)

        # 將兩個影片合併顯示在同一個視窗中
        combined_frame = np.concatenate((video_frame, gif_frame), axis=1)

        #cv2.imshow('Combined Video', combined_frame)

        out.write(combined_frame)
        if cv2.waitKey(50) & 0xFF == 27:  # ESC键退出
            break

        
        

    cv2.destroyAllWindows()
    #cap.release()
    out.release()
    print("leave combined video thread")
    

# Load the saved NumPy .npy file
outer_finger_angles = np.load('outer_finger_angles_website.npy', allow_pickle=True).item()
middle_finger_angles = np.load('middle_finger_angles_website.npy', allow_pickle=True).item()
inner_finger_angles = np.load('inner_finger_angles_website.npy', allow_pickle=True).item()
outer_save_directory = 'dipj_gif/'
middle_save_directory = 'pipj_gif/'
inner_save_directory = 'mcpj_gif/'
outer_combine='outer_combine/'
middle_combine='middle_combine/'
inner_combine='inner_combine/'

os.makedirs(outer_save_directory, exist_ok=True)
os.makedirs(middle_save_directory, exist_ok=True)
os.makedirs(inner_save_directory, exist_ok=True)


# Get the video frame count and frame rate
base_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(base_dir, 'static', 'user_upload_video.mp4')
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(total_frames)
fps = cap.get(cv2.CAP_PROP_FPS)


# Calculate total time of animation
total_time = total_frames / fps
print(total_time)

files = os.listdir(inner_save_directory)
for file_name in files:
    combined_video_thread(os.path.join(inner_save_directory, file_name),inner_combine)

files = os.listdir(middle_save_directory)
for file_name in files:
    combined_video_thread(os.path.join(middle_save_directory, file_name),middle_combine)

files = os.listdir(outer_save_directory)
for file_name in files:
    combined_video_thread(os.path.join(outer_save_directory, file_name),outer_combine)

cap.release()


    
