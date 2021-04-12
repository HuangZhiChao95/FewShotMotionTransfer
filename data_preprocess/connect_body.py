import cv2
import os
import json
import numpy as np
from tqdm import tqdm
from os.path import join as osp
import glob
import argparse

def create_pose(root_path):

    def drawline(label_map, p1, p2, threshold, label, thickness):
        if p1[2] > threshold and p2[2] > threshold:
            cv2.line(label_map, tuple(p1[:2]), tuple(p2[:2]), label, thickness)
        elif p1[2] > threshold:
            cv2.line(label_map, tuple(p1[:2]), tuple(p1[:2]), label, thickness)
        elif p2[2] > threshold:
            cv2.line(label_map, tuple(p2[:2]), tuple(p2[:2]), label, thickness)

    def connect_body(label_map, connect, threshold, point, thickness, label):
        for i in range(len(connect)):
            drawline(label_map, point[connect[i][0] * 3: connect[i][0] * 3 + 3],
                     point[connect[i][1] * 3:connect[i][1] * 3 + 3], threshold, label, thickness)
            label += 1

    os.system("mkdir -p {}".format(os.path.join(root_path, "body")))
    result = {}
    threshold = 0.2
    body_connect = [(0, 15), (0, 16), (15, 17), (16, 18), (0, 1), (1, 2), (1, 5), (1, 8), (2, 9), (5, 12), (8, 9), (8, 12),
                    (2, 3), (3, 4), (5, 6), (6, 7), (9, 10), (10, 11), (11, 24), (11, 22), (22, 23), (12, 13), (13, 14),
                    (14, 21), (14, 19), (19, 20)]

    head_index = [0, 15, 16, 17, 18]
    foot_index = [11, 14, 19, 20, 21, 22, 23, 24]
    images_path = os.listdir(osp(root_path, "image"))
    images_path.sort()
    frame = cv2.imread(osp(root_path, "image", images_path[0]))
    height = frame.shape[0]
    width = frame.shape[1]
    for j, image_path in tqdm(enumerate(images_path), total=len(images_path)):
        if image_path.find("jpg") == -1:
            continue
        name = image_path[:-4]

        with open(os.path.join(root_path, "json", "{}_keypoints.json".format(name))) as f:
            item = json.load(f)
        people = item["people"]
        if len(people) == 0:
            continue

        scale = np.zeros(len(people), dtype=np.float32)
        up = np.zeros(len(people), dtype=np.float32)
        bottom = np.zeros(len(people), dtype=np.float32)

        for k, points in enumerate(people):
            body_point = points["pose_keypoints_2d"]
            head_center = np.zeros(2, dtype=np.float32)
            count = 0
            for i in head_index:
                if body_point[2 + i * 3] >= threshold:
                    count += 1
                    head_center[0] += body_point[i * 3]
                    head_center[1] += body_point[i * 3 + 1]
            if count > 0:
                head_center = head_center / count
            else:
                continue

            foot_center = np.zeros(2, dtype=np.float32)
            count = 0
            for i in foot_index:
                if body_point[2 + i * 3] >= threshold:
                    count += 1
                    foot_center[0] += body_point[i * 3]
                    foot_center[1] += body_point[i * 3 + 1]
            if count > 0:
                foot_center = foot_center / count
            else:
                continue

            scale[k] = foot_center[1] - head_center[1]
            up[k] = head_center[1]
            bottom[k] = foot_center[1]

        k = np.argmax(scale)
        points = people[k]

        body_label = np.zeros((height, width), dtype=np.uint8)

        body_point = points["pose_keypoints_2d"]
        for i in range(len(body_point) // 3):
            body_point[3 * i] = int(max(body_point[3 * i], 0))
            body_point[3 * i + 1] = int(body_point[3 * i + 1])

        connect_body(body_label, body_connect, threshold, body_point, thickness=3, label=1)

        result["{}.jpg".format(name)] = body_point
        cv2.imwrite(os.path.join(root_path, "body", "{}.png".format(name)), body_label)


parser = argparse.ArgumentParser()
parser.add_argument("root")
args = parser.parse_args()

root_paths = glob.glob(osp(args.root, "*"))

for root_path in root_paths:
    print(root_path)
    create_pose(root_path)