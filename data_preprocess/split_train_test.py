import os
import glob
import argparse
from os.path import join as osp
import cv2
from tqdm import tqdm
import random

parser = argparse.ArgumentParser()
parser.add_argument("--root")
parser.add_argument("--train")
parser.add_argument("--test")
parser.add_argument("--n_train", type=int, default=50)

args = parser.parse_args()
root = args.root

if not os.path.exists(args.train):
    os.system("mkdir -p "+args.train)

if not os.path.exists(args.test):
    os.system("mkdir -p "+args.test)

folders = glob.glob(osp(root, "*"))
random.shuffle(folders)
train_folders = folders[:args.n_train]
test_folders = folders[args.n_train:]

for folder in train_folders:
    basename = os.path.basename(folder)
    os.system("ln -sf {} {}".format(osp(root, folder), osp(args.train, folder)))

for folder in test_folders:
    basename = os.path.basename(folder)
    os.system("ln -sf {} {}".format(osp(root, folder), osp(args.test, folder)))