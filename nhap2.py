# import cv2
import argparse
from collections import defaultdict
import warnings
import numpy as np
# from ultralytics import YOLO
import sys
sys.path.append("D:/NCKH/NCKH2024/CBBT/backend_cbbt")
from pyskl.apis.inference import inference_recognizer, init_recognizer
import torch
from ultralytics import YOLO
# from mmcv import load, dump
import mmcv
from time import time
from tqdm import tqdm
import cv2
from yolov5.utils.dataloaders import LoadStreams
from scipy.optimize import linear_sum_assignment

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# print(device)

checkpoint = 'epoch_4.pth'
config_test = "custom_config.py"
# config_test = "config_test.py"
config = mmcv.Config.fromfile(config_test)
# print(config)
# config.test_dataloader.dataset.pipeline = [x for x in config.test_dataloader.dataset.pipeline if x['type'] != 'DecompressPose']
# print(config)
config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']

model_stgcn = init_recognizer(config, checkpoint, device)
# print(model_stgcn)
label_map = [x.strip() for x in open("pyskl/tools/data/label_map/custom_label.txt").readlines()]

def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    kptThres = 0.1
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < kptThres:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
            # print((int(x_coord), int(y_coord)))
    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<kptThres or conf2<kptThres:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
    return im

def dist_ske(ske1, ske2):
    dist = np.linalg.norm(ske1[:, :2] - ske2[:, :2], axis=1) * 2
    diff = np.abs(ske1[:, 2] - ske2[:, 2])
    return np.sum(np.maximum(dist, diff))

def pose_tracking(pose_results, max_tracks=2, thre=30):
    # print(f'pose_results {pose_results.shape}')
    tracks, num_tracks = [], 0
    num_joints = None
    for idx, poses in enumerate(pose_results):
        if len(poses) == 0:
            continue
        if num_joints is None:
            num_joints = poses[0].shape[0]
        track_proposals = [t for t in tracks if t['data'][-1][0] > idx - thre]
        n, m = len(track_proposals), len(poses)
        scores = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                scores[i][j] = dist_ske(track_proposals[i]['data'][-1][1],
                                        poses[j])

        row, col = linear_sum_assignment(scores)
        for r, c in zip(row, col):
            track_proposals[r]['data'].append((idx, poses[c]))
        if m > n:
            for j in range(m):
                if j not in col:
                    num_tracks += 1
                    new_track = dict(data=[])
                    new_track['track_id'] = num_tracks
                    new_track['data'] = [(idx, poses[j])]
                    tracks.append(new_track)
    tracks.sort(key=lambda x: -len(x['data']))
    result = np.zeros((max_tracks, len(pose_results), num_joints, 3),
                      dtype=np.float16)
    for i, track in enumerate(tracks[:max_tracks]):
        for item in track['data']:
            idx, pose = item
            result[i, idx] = pose
    return result[..., :2], result[..., 2]

# source = 0
# cap = cv2.VideoCapture(source)  # 0
# print(model.device)
# dataset = LoadStreams(str(source), img_size=640, stride=64, auto=1, vid_stride=1)
# dataset = iter(dataset)

track_history=defaultdict(lambda: [])
drop_counting=defaultdict(lambda: 0)
max_miss = 10
kptSeqNum = 40


# while cap.isOpened():
#     # _, im, im0s, _, s = next(dataset)
#     # frame = im0s[0]
#     _, frame = cap.read()
#     result = model.track(frame,
#                         persist=True,
#                         verbose = False
#                         #   show=True,
#                         #   conf=0.7,
#                         #    iou=0.5,
#                         )[0]
#     # t2 = time.time()
#     # print(f"Time cost : {t2-t1} sec")
#     boxes = result.boxes.xywh.cpu()
#     keypoints = result.keypoints.data#.cpu().numpy()

#     track_ids = result.boxes.id#.int().cpu().tolist()
#     if track_ids is None:
#         track_ids = []
#     else:
#         track_ids = track_ids.int().cpu().tolist()

#     # print("track_ids : ",track_ids)
#     diff = list(set(list(set(track_history.keys()))).difference(track_ids))
#     for d in diff:
#         if drop_counting[d] > max_miss:
#             del drop_counting[d]
#             del track_history[d]
#         else:
#             drop_counting[d]+=1

#     track_ids_conform_frame_num = [] ; poseTrackResult = []
#     boxess = []
#     for box, track_id,keypoint in zip(boxes, track_ids,keypoints):
#         track = track_history[track_id]
#         track.append(keypoint.unsqueeze(0))

#         if kptSeqNum is not None:
#             if len(track) > kptSeqNum:  
#                 track.pop(0)
#             if len(track) == kptSeqNum:
#                 poseTrackResult.append(torch.cat(track).cpu().unsqueeze(0))
#                 track_ids_conform_frame_num.append(track_id)  
#         else:
#             poseTrackResult.append(torch.cat(track).cpu().unsqueeze(0))
#             track_ids_conform_frame_num.append(track_id)  
#         boxess.append(box)
#     # print(track_ids_conform_frame_num)
    
#     h, w, _ = frame.shape
#     for resultIdx, track_id in enumerate(track_ids_conform_frame_num):
#         number_of_seq = poseTrackResult[resultIdx].numpy().shape
#         pose_result = poseTrackResult[resultIdx].numpy()
#         pose_result = np.expand_dims(pose_result, 0)
#         fake_anno = dict(
#         frame_dir='',
#         label=-1,
#         img_shape=(int(h) , int(w)),
#         original_shape=(int(h), int(w)),
#         start_index=0,
#         modality='Pose',
#         total_frames=kptSeqNum)

#         num_person = 1
#         # Current PoseC3D models are trained on COCO-keypoints (17 keypoints)
#         num_keypoint = 17
#         keypoint = np.zeros((num_person, kptSeqNum, num_keypoint, 2),
#                             dtype=np.float16)
#         keypoint_score = np.zeros((num_person, kptSeqNum, num_keypoint),
#                                   dtype=np.float16)
#         # print( number_of_seq)
#         if len(number_of_seq) >=3:
#              keypoint = pose_result[0, :, :, :, :2]
#              keypoint_score = pose_result[0, :, :, :, 2]
#         action_label = "...."
#         fake_anno['keypoint'] = keypoint
#         fake_anno['keypoint_score'] = keypoint_score
        
#         results = inference_recognizer(model_stgcn, fake_anno)
#         print(results)
#         if results[0][1] >= 0.2:
#             action_label = label_map[results[0][0]]
#         else:
#             action_label = 'unknown'
#         print(f'{track_id} {action_label} {results[0][1]}')
#         current_kpt = poseTrackResult[resultIdx][0,-1,:,:].numpy().flatten()
#         x,y,w,h = boxess[resultIdx]
#         x1,y1,x2,y2 = int(x-(w/2)),int(y-(h/2)),int(x+(w/2)),int(y+(h/2))
#         # print(x1,y1,x2,y2)
#         text = f"tid:{track_id} with seq : {number_of_seq} action: {action_label}"
#         # print("number_of_seq :",number_of_seq)
#         # print("current kpt : ",current_kpt)
#         frame = plot_skeleton_kpts(frame,current_kpt,3)
#         cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
#         cv2.putText(frame, text, (x1, y1+15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 1, cv2.LINE_AA)
#     cv2.imshow('Result', frame)
#     cv2.waitKey(1)
    
def pose_recognition(frame):
    model = YOLO("yolov8n-pose.pt")
    model.to(device)
    result = model.track(frame,
                        persist=True,
                        verbose = False
                        #   show=True,
                        #   conf=0.7,
                        #    iou=0.5,
                        )[0]
    # t2 = time.time()
    # print(f"Time cost : {t2-t1} sec")
    boxes = result.boxes.xywh.cpu()
    keypoints = result.keypoints.data#.cpu().numpy()

    track_ids = result.boxes.id#.int().cpu().tolist()
    if track_ids is None:
        track_ids = []
    else:
        track_ids = track_ids.int().cpu().tolist()

    # print("track_ids : ",track_ids)
    diff = list(set(list(set(track_history.keys()))).difference(track_ids))
    for d in diff:
        if drop_counting[d] > max_miss:
            del drop_counting[d]
            del track_history[d]
        else:
            drop_counting[d]+=1

    track_ids_conform_frame_num = [] ; poseTrackResult = []
    boxess = []
    for box, track_id,keypoint in zip(boxes, track_ids,keypoints):
        track = track_history[track_id]
        track.append(keypoint.unsqueeze(0))

        if kptSeqNum is not None:
            if len(track) > kptSeqNum:  
                track.pop(0)
            if len(track) == kptSeqNum:
                poseTrackResult.append(torch.cat(track).cpu().unsqueeze(0))
                track_ids_conform_frame_num.append(track_id)  
        else:
            poseTrackResult.append(torch.cat(track).cpu().unsqueeze(0))
            track_ids_conform_frame_num.append(track_id)  
        boxess.append(box)
    # print(track_ids_conform_frame_num)
    
    h, w, _ = frame.shape
    action_labels = []
    for resultIdx, track_id in enumerate(track_ids_conform_frame_num):
        number_of_seq = poseTrackResult[resultIdx].numpy().shape
        pose_result = poseTrackResult[resultIdx].numpy()
        pose_result = np.expand_dims(pose_result, 0)
        fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(int(h) , int(w)),
        original_shape=(int(h), int(w)),
        start_index=0,
        modality='Pose',
        total_frames=kptSeqNum)

        num_person = 1
        # Current PoseC3D models are trained on COCO-keypoints (17 keypoints)
        num_keypoint = 17
        keypoint = np.zeros((num_person, kptSeqNum, num_keypoint, 2),
                            dtype=np.float16)
        keypoint_score = np.zeros((num_person, kptSeqNum, num_keypoint),
                                  dtype=np.float16)
        # print( number_of_seq)
        if len(number_of_seq) >=3:
             keypoint = pose_result[0, :, :, :, :2]
             keypoint_score = pose_result[0, :, :, :, 2]
        action_label = "...."
        fake_anno['keypoint'] = keypoint
        fake_anno['keypoint_score'] = keypoint_score
        
        results = inference_recognizer(model_stgcn, fake_anno)
        if results[0][1] >= 0.7:
            action_label = label_map[results[0][0]]
        else:
            action_label = 'unknown'
        action_labels.append(action_label)
        # print(f'{track_id} {action_label} {results[0][1]}')
        current_kpt = poseTrackResult[resultIdx][0,-1,:,:].numpy().flatten()
        x,y,w,h = boxess[resultIdx]
        x1,y1,x2,y2 = int(x-(w/2)),int(y-(h/2)),int(x+(w/2)),int(y+(h/2))
        # print(x1,y1,x2,y2)
        text = f"tid:{track_id} action: {action_label}"
        # print("number_of_seq :",number_of_seq)
        # print("current kpt : ",current_kpt)
        frame = plot_skeleton_kpts(frame,current_kpt,3)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
        cv2.putText(frame, text, (x1, y1+15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return  frame, action_labels