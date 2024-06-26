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


def init():
    line.set_data([], [])

    return line,

def update(frame, finger_n, angles):
    time_values = np.arange(frame + 1)*8 / 30
    y_values = angles[finger_n][:frame + 1]
    line.set_data(time_values, y_values)
    # Set x-axis ticks to represent time intervals
    ax.set_xticks(np.arange(0, time_values[-1]+1, 1))  # Assuming one tick per second
    ax.set_xlabel('Time (seconds)')

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





    

# Load the saved NumPy .npy file
outer_finger_angles = np.load('outer_finger_angles_website.npy', allow_pickle=True).item()
middle_finger_angles = np.load('middle_finger_angles_website.npy', allow_pickle=True).item()
inner_finger_angles = np.load('inner_finger_angles_website.npy', allow_pickle=True).item()
outer_save_directory = 'dipj_gif/'
middle_save_directory = 'pipj_gif/'
inner_save_directory = 'mcpj_gif/'
os.makedirs(outer_save_directory, exist_ok=True)
os.makedirs(middle_save_directory, exist_ok=True)
os.makedirs(inner_save_directory, exist_ok=True)


# Get the video frame count and frame rate
base_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(base_dir, 'static', 'user_upload_video.mp4')
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

total_time=0
# Calculate total time of animation
if fps!=0:
    total_time = total_frames / fps


# Plot the finger angles
for finger_name, angles in outer_finger_angles.items():
    #plt.plot(angles, label=finger_name)
    # Add labels and legend
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_xlabel('Frame')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('DIPJ Angles over Time')
    
    
    plt.plot(angles,color='white')
    line, = ax.plot([], [], label=f'{finger_name} Angle')
    max_angle = np.max(angles)
    min_angle = np.min(angles)
    plt.text(total_frames /2 , max_angle, f'Max Angle: {max_angle:.2f}', ha='center')
    plt.text(total_frames /2, min_angle, f'Min Angle: {min_angle:.2f}', ha='center')
    ax.legend()
    #plt.savefig(os.path.join(outer_save_directory, f'{finger_name}_angle_plot.png'))
    num_frames = len(outer_finger_angles[finger_name])
    video_interval = total_time / len(outer_finger_angles[finger_name])
    print("total time=",total_time)
    print(len(outer_finger_angles[finger_name]))
    # Set x-axis limits to cover the entire range of time values
    ax.set_xlim(0, total_time)
    ani = FuncAnimation(fig, update, frames=num_frames,fargs=(finger_name,outer_finger_angles), init_func=init, blit=True, interval=total_time*1000/num_frames)
    # Show the plot
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


    
    writer = PillowWriter(fps=num_frames/total_time)
    ani.save(os.path.join(outer_save_directory, f'{finger_name}_angle_animation.gif'), writer=writer)

    #plt.show()

    
    

for finger_name, angles in middle_finger_angles.items():
    #plt.plot(angles, label=finger_name)
    # Add labels and legend
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_xlabel('Frame')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('PIPJ over Time')
    
    
    plt.plot(angles, color='white')
    line, = ax.plot([], [], label=f'{finger_name} Angle')
    max_angle = np.max(angles)
    min_angle = np.min(angles)
    plt.text(total_time *2 , max_angle, f'Max Angle: {max_angle:.2f}', ha='center')
    plt.text(total_time *2, min_angle, f'Min Angle: {min_angle:.2f}', ha='center')
    ax.legend()
    #plt.savefig(os.path.join(middle_save_directory, f'{finger_name}_angle_plot.png'))
    num_frames = len(middle_finger_angles[finger_name])
    video_interval = total_time / len(middle_finger_angles[finger_name])
    print("total time=",total_time)
    print(len(middle_finger_angles[finger_name]))
    ax.set_xlim(0, total_time)
    ani = FuncAnimation(fig, update, frames=num_frames,fargs=(finger_name,middle_finger_angles), init_func=init, blit=True, interval=total_time*1000/num_frames)
    # Show the plot
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)



    writer = PillowWriter(fps=num_frames/total_time)
    ani.save(os.path.join(middle_save_directory, f'{finger_name}_angle_animation.gif'), writer=writer)

    
    #plt.show()

for finger_name, angles in inner_finger_angles.items():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_xlabel('Frame')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('MCPJ Angles over Time')
    
    
    plt.plot(angles, color='white')
    line, = ax.plot([], [], label=f'{finger_name} Angle')
    max_angle = np.max(angles)
    min_angle = np.min(angles)
    plt.text(total_time , max_angle, f'Max Angle: {max_angle:.2f}', ha='center')
    plt.text(total_time , min_angle, f'Min Angle: {min_angle:.2f}', ha='center')
    ax.legend()

    #plt.savefig(os.path.join(inner_save_directory, f'{finger_name}_angle_plot.png'))
    num_frames = len(inner_finger_angles[finger_name])
    video_interval = total_time / len(inner_finger_angles[finger_name])
    print("total time=",total_time)
    print(len(inner_finger_angles[finger_name]))
    ax.set_xlim(0, total_time)
    ani = FuncAnimation(fig, update, frames=num_frames,fargs=(finger_name,inner_finger_angles), init_func=init, blit=True, interval=total_time*1000/num_frames)
    # Show the plot
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)



    writer = PillowWriter(fps=num_frames/total_time)
    ani.save(os.path.join(inner_save_directory, f'{finger_name}_angle_animation.gif'), writer=writer)

    #plt.show()
    
