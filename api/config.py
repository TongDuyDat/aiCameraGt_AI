class Config:
    DEBUG = True
    SECRET_KEY = 'mysecretkey' 
    DATABASE_URI = 'sqlite:///db.sqlite'
    UPLOAD_FOLDER = "video_upload"
    # WEIGHTS = "D:/freelance/detect_door_open/door1.pt"
    WEIGHTS = "D:/NCKH/NCKH2024/CBBT/backend_cbbt/yolov5/weights/yolov5l.pt"
    DATASET = "yolov5/data/data.yaml"
    YOLOv5 = "D:/freelance/detect_door_open/yolov5"
    IOU_cof = 0.45
    SCORE_threshold = 0.25
    NUMBER_class = 2
    RECORD_PATH = 'record'
    host = '192.168.0.101'  # Adjust this to your actual IP address
    port = '5000'  # Adjust this to your actual port number