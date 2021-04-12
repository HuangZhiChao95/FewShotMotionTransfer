from PIL import Image
import glob
import numpy as np
from os.path import join as osp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("root")
args = parser.parse_args()

folders = glob.glob(osp(args.root, "*"))

for folder in folders:
    with open(osp(folder, "image_list.txt")) as f:
        files = f.readlines()
    backgrounds = []
    masks = []
    for file in files:
        image = np.asarray(Image.open(osp(folder, "image/{}.jpg".format(file.strip())))).astype(np.float32) / 255
        mask = Image.open(osp(folder, "segmentation/{}.png".format(file.strip()))).convert("RGB")
        mask = np.asarray(mask).astype(np.float32) / 255
        background = (1 - mask) * image
        backgrounds.append(background)
        masks.append(1 - mask)

    b = backgrounds[0]
    m = masks[0]
    n = len(masks) // 5
    for i in range(1, len(masks)):
        index = np.logical_and(m < 0.5, masks[i] > 0.5)
        b[index] = backgrounds[i][index]
        m = np.clip(masks[i] + m, 0, 1)

    b = Image.fromarray((b * 255).astype(np.uint8))
    b.save(osp(folder, "background.png"))
