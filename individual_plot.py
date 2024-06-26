import numpy as np
import matplotlib.pyplot as plt
import os

# Load the saved NumPy .npy file
outer_finger_angles = np.load('outer_finger_angles_website.npy', allow_pickle=True).item()
middle_finger_angles = np.load('middle_finger_angles_website.npy', allow_pickle=True).item()
inner_finger_angles = np.load('inner_finger_angles_website.npy', allow_pickle=True).item()
outer_save_directory = 'outer_angle_images_indi/'
middle_save_directory = 'middle_angle_images_indi/'
inner_save_directory = 'inner_angle_images_indi/'
os.makedirs(outer_save_directory, exist_ok=True)
os.makedirs(middle_save_directory, exist_ok=True)
os.makedirs(inner_save_directory, exist_ok=True)
fingers = {
  "Finger_1": "Thumb",
  "Finger_2": "Index finger",
  "Finger_3": "Middle finger",
  "Finger_4": "Ring finger",
  "Finger_5": "Little finger"
}
# Assuming frame rate is known (replace 30 with your actual frame rate)
frame_rate = 30

# Plot the finger angles
for finger_name, angles in outer_finger_angles.items():
    # Convert frame numbers to time in seconds
    time_seconds = np.arange(len(angles))*8 / frame_rate
    plt.plot(time_seconds, angles, label=fingers[finger_name])
    # Add labels and legend
    plt.xlabel('Time (seconds)')
    plt.ylabel('Angle (degrees)')
    plt.title('DIPJ over Time')
    plt.legend()
    # Set x-axis ticks every 0.5 seconds
    plt.xticks(np.arange(0, time_seconds[-1], step= 1))
    plt.savefig(os.path.join(outer_save_directory, f'{finger_name}_angle_plot.png'))
    # Show the plot
    plt.close()

for finger_name, angles in middle_finger_angles.items():
    # Convert frame numbers to time in seconds
    time_seconds = np.arange(len(angles))*8 / frame_rate
    plt.plot(time_seconds, angles, label=fingers[finger_name])
    # Add labels and legend
    plt.xlabel('Time (seconds)')
    plt.ylabel('Angle (degrees)')
    plt.title('PIPJ over Time')
    plt.legend()
    # Set x-axis ticks every 0.5 seconds
    plt.xticks(np.arange(0, time_seconds[-1], step= 1))
    plt.savefig(os.path.join(middle_save_directory, f'{finger_name}_angle_plot.png'))
    # Show the plot
    plt.close()

for finger_name, angles in inner_finger_angles.items():
    # Convert frame numbers to time in seconds
    time_seconds = np.arange(len(angles))*8 / frame_rate
    plt.plot(time_seconds, angles, label=fingers[finger_name])
    # Add labels and legend
    plt.xlabel('Time (seconds)')
    plt.ylabel('Angle (degrees)')
    plt.title('MCPJ over Time')
    plt.legend()
    # Set x-axis ticks every 0.5 seconds
    plt.xticks(np.arange(0, time_seconds[-1], step= 1))
    plt.savefig(os.path.join(inner_save_directory, f'{finger_name}_angle_plot.png'))
    # Show the plot
    plt.close()
