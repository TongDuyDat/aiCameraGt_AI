from io import BytesIO
import os
from flask import Blueprint, jsonify, request,Response, send_file, url_for
from api.config import Config
from  PIL import Image
from pathlib import Path
import pathlib
from database.record_db import Record
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from subprocess import Popen, PIPE
records = Blueprint('records', __name__, url_prefix="/api")
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_root_path = os.path.join(root_path, Config.RECORD_PATH)
def check_file_size(file_path, threshold=1.5):
    """
    Kiểm tra kích thước của file và in ra thông báo nếu kích thước lớn hơn ngưỡng được chỉ định.
    
    Args:
    - file_path (str): Đường dẫn tới file cần kiểm tra.
    - threshold (float): Ngưỡng kích thước file để kiểm tra (đơn vị: MB).
    """
    # Chuyển đổi ngưỡng từ MB sang bytes
    threshold_bytes = threshold * 1024 * 1024
    
    # Kiểm tra kích thước file
    file_size = os.path.getsize(file_path)
    
    if file_size > threshold_bytes:
        return True
    else:
        return False
    
@records.route('/download_video/<filename>', methods=['GET'])
def download_video(filename):
    # Assuming your video files are stored in a folder named 'videos'
    # Adjust the folder path accordingly
    # Construct the full path to the video file
    video_path = os.path.join(save_root_path, filename)
    # Return the file as a response
    return send_file(video_path, as_attachment=False)

@records.route('/load_records', methods=['GET'])
def load_records():
    """
    Load records for a given username.

    Returns:
        200 - If records are loaded successfully.
        404 - If no records are found for the given username.
        500 - For any other exception that occurs during processing.
    """
    try:
        # Query records for the given username
        username = request.args["username"]
        records = Record.objects(username=username).order_by('-created_at')
        # records = sorted(records, key=lambda x:x.created_at, reverse=True)
        url = f"http://{Config.host}:{Config.port}/"
        # If no records found, return 404
        if not records:
            return jsonify({'error': 'No records found for the given username.'}), 200

        # Serialize records to JSON
        serialized_records = []
        
        for record in records:
            if not check_file_size(record.file_path, 2):
                continue
            tmp = {
            'url': url + url_for('api.records.download_video', filename=f"{Path(record.file_path).name}"),
            'created_at': record.created_at,
            'id_camera': record.id_camera,
            'content': record.content
            }
            serialized_records.append(tmp)
        if len(records) >10:
            records = records[0:10]
        else:
            records = records
        # Return records
        return jsonify(serialized_records), 200
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
