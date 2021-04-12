import os
import glob
import argparse
from os.path import join as osp
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("root")
args = parser.parse_args()
root = args.root

if not os.path.exists(root):
    os.system("mkdir -p "+root)

with open("video_list.txt") as f:
    lines = f.readlines()

for line in lines:
    items = line.strip().split(" ")
    video_id = items[0]
    url = "https://www.youtube.com/watch?v="+video_id
    os.system("youtube-dl --id "+url)
    filename = glob.glob(video_id+"*")[0]
    os.system("mv {} {}".format(filename, osp(root, filename)))
    if not os.path.exists(osp(root, video_id, "origin_image")):
        os.system("mkdir -p " + osp(root, video_id, "origin_image"))
    os.system("ffmpeg -i {} {} {}/%06d.jpg".format(osp(root, filename), " ".join(items[1:]), osp(root, video_id, "origin_image")))
    os.system("mv {} {}".format(osp(root, filename), osp(root, video_id)))

folders = glob.glob(osp(root, "*"))
for folder in folders:
    images = glob.glob(osp(folder, "origin_image", "*.jpg"))
    if not os.path.exists(osp(folder, "image")):
        os.system("mkdir -p "+osp(folder, "image"))
    print(folder)
    for img_path in tqdm(images, total=len(images)):
        in_path = img_path
        out_path = img_path.replace("origin_image", "image")
        image = cv2.imread(in_path)
        h, w, _ = image.shape
        image = image[:, (w-h)//2:(w+h)//2]
        image = cv2.resize(image, (512, 512))
        cv2.imwrite(out_path, image)