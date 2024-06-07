import mimetypes
import subprocess
from threading import Semaphore
import datetime
import os
import pathlib
import cv2
import time
import ffmpeg
import requests
import torch
from flask import Blueprint, Response, request, send_file, stream_with_context
from PIL import Image
from pathlib import Path
from threading import Thread, Lock
from api.config import Config
from api.funtionals import send_push_notification
from athu.auth_midde import token_required
from database.camera_db import Camera
from database.notification_db import Notification
from database.record_db import Record
from yolov5.detect import detect_img
from yolov5.models.common import DetectMultiBackend
from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import select_device
from nhap2 import pose_recognition
from flask_cors import CORS

from flask_cors import cross_origin

live = Blueprint('live', __name__, url_prefix="/api")
CORS(live)
# Thay đổi PosixPath thành WindowsPath tạm thời
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


def run_ffmpeg(width, height, fps, id_camera):
    HLS_OUTPUT = f'stream/hls/{id_camera}/stream.m3u8'
    os.makedirs(f'stream/hls/{id_camera}', exist_ok=True)
    output_stream = ffmpeg.output(
    ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s='720x640'),  # thay thế '640x480' bằng kích thước thực tế của video của bạn
    HLS_OUTPUT,
    **{
        'c:v': 'libx264',
        'preset': 'superfast',  # Tăng tốc độ mã hóa
        'crf': '23',
        'b:v': '900k',
        'f': 'hls',
        'hls_time': '10',  # Giảm thời gian mỗi phân đoạn
        'hls_list_size': '5',  # Tăng số lượng phân đoạn lưu trữ
        'hls_flags': 'delete_segments+append_list',
        'hls_delete_threshold': '10',  # Tăng ngưỡng xóa
        'g': '30',  # I-frame interval
        'r': '15'
    }
    )
    return ffmpeg.run_async(output_stream, pipe_stdin=True)

# Load model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = select_device(device)

# Định nghĩa đường dẫn cho việc lưu trữ video
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_root_path = os.path.join(root_path, 'record')
if not os.path.exists(save_root_path):
    os.makedirs(save_root_path)

# Khởi tạo lock cho phân luồng
thread_lock = Lock()
def save_record(frame_falling, username, cap):
    content = "Có người ngã trong video"
    h, w, _ = frame_falling[0].shape
    save_record_path = os.path.join(save_root_path, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    save_path = str(Path(save_record_path).with_suffix(".mp4"))  
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"X264"), 17, (w, h))
    id_record = Record.create_record(save_path, username, str(cap.id), content)
    for frame in frame_falling:
        print(len(frame_falling))
        if len(frame.shape) == 3:
            vid_writer.write(frame)
            cv2.waitKey(1)
    Notification.create_notification(cap.token_expo, username, id_record, f"Have person falling in  {cap.name}", str(cap.id))
# Hàm thực hiện nhận dạng và streaming video
def detect(step=1, username="", camera_id=0, ai=True):
    # Đường dẫn của video hoặc camera
    model = attempt_load(Config.WEIGHTS, device=device)
    cap = Camera.get_camera_by_id(camera_id)
    camera_url = cap.rtsp_url
    if len(camera_url) < 4:
        camera_url = int(camera_url)
    camera = cv2.VideoCapture(camera_url)
    count = 0
    if camera:  
        fps = camera.get(cv2.CAP_PROP_FPS)
        w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:  
        fps, w, h = 20, im0.shape[1], im0.shape[0]
    ffmpeg_process = run_ffmpeg( w,h, fps, camera_id)
    save_record_path = os.path.join(save_root_path, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    save_path = str(Path(save_record_path).with_suffix(".mp4"))  
    print(save_path)
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"X264"), fps, (w, h))
    flag_create_new_record = True
    count = 0
    frame_falling = []
    count_frame_fall = 0
    extend = 0
    while True:
        success, frame = camera.read()
        if not success:
            break
        # im0 = frame
        if count % step == 0:   
            im0, _, clss = detect_img(frame, model, True, True)
            if 0 not in clss and flag_create_new_record:
                content = "Có người trong video"
                save_record_path = os.path.join(save_root_path, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
                save_path = str(Path(save_record_path).with_suffix(".mp4")) 
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"X264"), fps, (w, h))
                id_record = Record.create_record(save_path, username, camera_id, content)
                print("save record in {savepath}")
                flag_create_new_record = False
            else:
                if 0 in clss:   
                    frame0, action_labels = pose_recognition(frame)
                    if 'falling' in action_labels:
                        count_frame_fall +=1
                        if count_frame_fall > 20:
                            save_record(frame_falling, username, cap)
                            # save_record(frame_falling, username, cap)
                            count_frame_fall = 0
                            send_push_notification(cap.token_expo, f'Thông báo từ camera {cap.name}', f"Have person falling in  camera {cap.name}")
                        # Notification.create_notification('token', username, id_record, f"Have person falling in  {cap.name}", camera_id)
                    # cv2.imshow("pose", frame0)
                    flag_create_new_record = True
            im0 = cv2.resize(im0, dsize=(frame.shape[1], frame.shape[0]))
            frame_falling.append(frame)
            if  len(frame_falling) > 200:
                frame_falling.pop(0)
        if not ai:
            im0 = frame
        _, buffer = cv2.imencode('.jpg', im0)
        frame_data = buffer.tobytes()
        count += 1
        vid_writer.write(im0)
        img = cv2.resize(im0, (720, 640))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ffmpeg_process.stdin.write(img.tobytes())
        # cv2.imshow("view", im0)
        # cv2.waitKey(1)
        # yield (b'--frame\r\n'
        #        b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
number_thead = 0
# Route để stream video
semaphore = Semaphore(2)
# @live.route('/live/<user_name>/<camera_id>')
# def video_stream(user_name, camera_id):
#     ai = request.args.get("ai")
#     # Kiểm tra Semaphore trước khi tạo luồng mới
#     with thread_lock:
#         if semaphore.acquire(blocking=False):
#             thread = Thread(
#                 daemon=True,
#                 target=detect,
#                 args=(1, user_name, camera_id, ai)
#             )
#             thread.start()
#         else:
#             print("Hàng đợi đầy. Không thể tạo thêm luồng.")

#     return Response(stream_with_context(detect(1, user_name, camera_id, ai)), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Hàm để stream các frame đã được nhận dạng
# def stream_frames(detect_func, username, camera_id, ai):
#     with thread_lock:
#         return Response(detect_func(1, username, camera_id, ai), mimetype='multipart/x-mixed-replace; boundary=frame') 

@live.route("/live/<id_camera>/stream.m3u8", methods = ["GET"])
@cross_origin()
def live_m3u8(id_camera):
    ai = True
    cap = Camera.get_camera_by_id(id_camera)
    username = ""
    if cap:
        username = cap.created_by
    else:
        return "Not found", 404
    with thread_lock:
        if semaphore.acquire(blocking=False):
            thread = Thread(
                daemon=True,
                target=detect,
                args=(1, username, id_camera, ai)
            )
            thread.start()
        else:
            print("Hàng đợi đầy. Không thể tạo thêm luồng.")
    time.sleep(5)
    m3u8_path = f"stream/hls/{id_camera}/stream.m3u8"
    file_type, _ = mimetypes.guess_type(m3u8_path)
    if file_type is None:
        file_type = 'application/vnd.apple.mpegurl'
    
    def generate():
        with open(m3u8_path, 'rb') as f:
            while True:
                data = f.read(4096)
                if not data:
                    break
                yield data

    try:
        return Response(generate(), mimetype=file_type, direct_passthrough=True)
    except FileNotFoundError:
        return "Not found", 404

@live.route('/live/<id_camera>/<segment>')
def video_segment(id_camera, segment):
    # print("segment", segment, end= "\n\n\n\n\n")
    # segment_number = int(segment)
    path = f"stream/hls/{id_camera}/{segment}"
    with open(path, 'rb') as segment_file:
        data = segment_file.read()
    yield data