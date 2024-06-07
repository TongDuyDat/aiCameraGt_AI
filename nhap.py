# import cv2
# import ffmpeg
# import subprocess
# from flask import Flask, Response

# app = Flask(__name__)

# # Initialize video capture from webcam (change index if you have multiple cameras)
# cap = cv2.VideoCapture(0)

# # FFmpeg command for HLS output
# ffmpeg_cmd = (
#     ffmpeg
#     .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='640x480')
#     .output('stream.m3u8',
#             format='hls',
#             hls_time=5,
#             hls_list_size=10,
#             hls_flags='delete_segments',
#             hls_allow_cache=5)
#     .compile()
# )

# @app.route('/video.m3u8')
# def video():
#     m3u8_content = '''#EXTM3U
#     #EXT-X-VERSION:3
#     #EXT-X-TARGETDURATION:5
#     #EXT-X-MEDIA-SEQUENCE:0
#     #EXTINF:5,
#     /video/0\n'''
#     return Response(m3u8_content, content_type='application/vnd.apple.mpegurl')

# @app.route('/video/<int:segment>')
# def video_segment(segment):
#     segment_filename = f'stream{segment}.ts'
#     try:
#         with open(segment_filename, 'rb') as segment_file:
#             data = segment_file.read()
#         return Response(data, content_type='video/MP2T')
#     except FileNotFoundError:
#         return 'Segment not found', 404

# def generate_frames():
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         yield frame.tobytes()

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    
#     ffmpeg_process = subprocess.Popen(
#         ffmpeg_cmd,
#         stdin=subprocess.PIPE
#     )
    
#     for frame in generate_frames():
#         ffmpeg_process.stdin.write(frame)
    
#     ffmpeg_process.stdin.close()
#     ffmpeg_process.wait()

a = [0 ,1]

print(a[-20:])