from flask import Flask, render_template, send_file, send_from_directory, request, jsonify
import subprocess,os,csv
import uuid
from moviepy.editor import VideoFileClip
from zipfile import ZipFile

app = Flask(__name__)

# Set the path where videos will be stored
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
VIDEO_NAME = 'user_upload_video'  # Change this to your desired video name

def get_video_list(path):
    videos = []
    for filename in os.listdir(path):
        if filename.endswith('.mp4') or filename.endswith('.avi'):
            videos.append(filename)
    return videos

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/right')
def right_page():
    # Logic for the right page
    return render_template('righthand.html')

@app.route('/left')
def left_page():
    # Logic for the left page
    return render_template('lefthand.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'})

    video_file = request.files['video']

    if video_file.filename == '':
        return jsonify({'error': 'No selected video file'})

    if video_file:
        # Save the video file with the fixed name
        video_extension = os.path.splitext(video_file.filename)[1]
        filename = VIDEO_NAME + video_extension
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # Check if a file with the same name exists and delete it
        if os.path.exists(video_path):
            os.remove(video_path)
        video_file.save(video_path)

        # Convert the saved .webm file to .mp4
        video_path_mp4 = os.path.join(app.config['UPLOAD_FOLDER'], VIDEO_NAME + '.mp4')
        if os.path.exists(video_path_mp4):
            os.remove(video_path_mp4)
        
        subprocess.run(['ffmpeg', '-i', video_path, video_path_mp4])

        return jsonify({'success': 'Video uploaded successfully', 'filename': filename})
    else:
        return jsonify({'error': 'Video not provided'})

@app.route('/process-righthand-video')
def process_righthand_video():
    try:
        subprocess.run(['python', 'righthand_video_process.py'], check=True)
        subprocess.run(['python', 'individual_plot.py'], check=True)
        subprocess.run(['python', 'video_plot.py'], check=True)
        subprocess.run(['python', 'combine.py'], check=True)

        inner_files = []
        directory = 'inner_angle_images_indi'
        files = os.listdir(directory)
        files_with_path = [os.path.join(directory, file).replace('\\', '/') for file in files]
        inner_files.extend(files_with_path)
        
        middle_files = []
        directory = 'middle_angle_images_indi'
        files = os.listdir(directory)
        files_with_path = [os.path.join(directory, file).replace('\\', '/') for file in files]
        middle_files.extend(files_with_path)

        outer_files = []
        directory = 'outer_angle_images_indi'
        files = os.listdir(directory)
        files_with_path = [os.path.join(directory, file).replace('\\', '/') for file in files]
        outer_files.extend(files_with_path)
        
        outer_videos = get_video_list('outer_combine')
        middle_videos = get_video_list('middle_combine')
        inner_videos = get_video_list('inner_combine')

        # Read data from CSV files
        data1 = []
        # Assuming your CSV files are named i1.csv, i2.csv, and i3.csv
        filenames = ['i1.csv', 'm1.csv', 'o1.csv']
        for filename in filenames:
            with open(filename, 'r') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip the first row
                column = []
                for row in csv_reader:
                    column.append(row[1])
                data1.append(column)

        # Read data from CSV files
        data2 = []
        # Assuming your CSV files are named i1.csv, i2.csv, and i3.csv
        for filename in filenames:
            with open(filename, 'r') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip the first row
                column = []
                for row in csv_reader:
                    column.append(row[2])
                data2.append(column)
        return render_template('video_replay.html', inner_image_list=inner_files, middle_image_list=middle_files, outer_image_list=outer_files, 
                                video_path='/static/' + VIDEO_NAME + '.mp4',outerV=outer_videos,middleV=middle_videos,innerV=inner_videos,
                                data1 = data1,data2 = data2)
    except subprocess.CalledProcessError as e:
        return f'Error occurred: {e}'

@app.route('/process-lefthand-video')
def process_lefthand_video():
    try:
        subprocess.run(['python', 'lefthand_video_process.py'], check=True)
        subprocess.run(['python', 'individual_plot.py'], check=True)
        subprocess.run(['python', 'video_plot.py'], check=True)
        subprocess.run(['python', 'combine.py'], check=True)

        inner_files = []
        directory = 'inner_angle_images_indi'
        files = os.listdir(directory)
        files_with_path = [os.path.join(directory, file).replace('\\', '/') for file in files]
        inner_files.extend(files_with_path)
        
        middle_files = []
        directory = 'middle_angle_images_indi'
        files = os.listdir(directory)
        files_with_path = [os.path.join(directory, file).replace('\\', '/') for file in files]
        middle_files.extend(files_with_path)

        outer_files = []
        directory = 'outer_angle_images_indi'
        files = os.listdir(directory)
        files_with_path = [os.path.join(directory, file).replace('\\', '/') for file in files]
        outer_files.extend(files_with_path)
        
        outer_videos = get_video_list('outer_combine')
        middle_videos = get_video_list('middle_combine')
        inner_videos = get_video_list('inner_combine')

        # Read data from CSV files
        data1 = []
        # Assuming your CSV files are named i1.csv, i2.csv, and i3.csv
        filenames = ['i1.csv', 'm1.csv', 'o1.csv']
        for filename in filenames:
            with open(filename, 'r') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip the first row
                column = []
                for row in csv_reader:
                    column.append(row[1])
                data1.append(column)
        # Read data from CSV files
        data2 = []
        # Assuming your CSV files are named i1.csv, i2.csv, and i3.csv
        for filename in filenames:
            with open(filename, 'r') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip the first row
                column = []
                for row in csv_reader:
                    column.append(row[2])
                data2.append(column)
        return render_template('video_replay.html', inner_image_list=inner_files, middle_image_list=middle_files, outer_image_list=outer_files, 
                                video_path='/static/' + VIDEO_NAME + '.mp4',outerV=outer_videos,middleV=middle_videos,innerV=inner_videos
                                ,data1 = data1,data2 = data2)
    except subprocess.CalledProcessError as e:
        return f'Error occurred: {e}'

@app.route('/video')
def video():
    # Specify the path to your video file
    video_path = 'static/user_upload_video.mp4'
    return send_file(video_path, mimetype='video/mp4')

@app.route('/images/<path:filename>')
def images(filename):
    # Determine the directory based on the filename
    directory, filename = filename.split('/', 1)
    # Print the content before and after '/'
    return send_from_directory(f'{directory}', filename)

@app.route('/outer_combine/<filename>')
def outer_combine(filename):
    return send_from_directory('outer_combine', filename)

@app.route('/middle_combine/<filename>')
def middle_combine(filename):
    return send_from_directory('middle_combine', filename)

@app.route('/inner_combine/<filename>')
def inner_combine(filename):
    return send_from_directory('inner_combine', filename)

@app.route("/download-folder")
def download_folder():
    folder_paths = {"middle_angle_images_indi": "PIPJ", "inner_angle_images_indi": "MCPJ", "outer_angle_images_indi": "DIPJ"}
    zip_file_path = "images.zip"  # 壓縮文件將被臨時存儲的路徑

    # 創建包含文件夾內容的zip文件
    with ZipFile(zip_file_path, "w") as zipf:
        for folder_path, prefix in folder_paths.items():
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    # 使用文件夹的前缀来构建新的文件名
                    new_file_name = f"{prefix}_{file}"
                    # 将文件添加到 zip 文件，新文件名包含文件夹的前缀
                    zipf.write(os.path.join(root, file), new_file_name)

    # 將zip文件發送給客戶端
    return send_file(zip_file_path, as_attachment=True)

@app.route("/download-combine")
def download_combine():
    folder_paths = {"middle_combine": "PIPJ", "inner_combine": "MCPJ", "outer_combine": "DIPJ"}
    zip_file_path = "combine_videos.zip"  # 壓縮文件將被臨時存儲的路徑

    # 創建包含文件夾內容的zip文件
    with ZipFile(zip_file_path, "w") as zipf:
        for folder_path, _ in folder_paths.items():  
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    # 將文件添加到 zip 文件，使用原始文件名
                    zipf.write(os.path.join(root, file))

    # 將zip文件發送給客戶端
    return send_file(zip_file_path, as_attachment=True)




if __name__ == "__main__":
    app.run('0.0.0.0', debug=True)