import cv2
import numpy as np
import os
from os.path import join as osp
from tqdm import tqdm
from glob import glob
import argparse

def GetTexture(im, IUV):
    U = IUV[:, :, 1]
    V = IUV[:, :, 2]
    Texture = np.zeros((24, 128, 128, 3), dtype=np.uint8)
    for PartInd in range(1, 25):
        tex = Texture[PartInd - 1, :, :, :].squeeze()
        x, y = np.where(IUV[:, :, 0] == PartInd)
        u = U[x, y] // 2
        v = V[x, y] // 2
        tex[u, v] = im[x, y]
        Texture[PartInd - 1] = tex
    TextureIm = np.zeros((128 * 4, 128 * 6, 3), dtype=np.uint8)
    for i in range(len(Texture)):
        x = i // 6 * 128
        y = i % 6 * 128
        TextureIm[x:x + 128, y:y + 128] = Texture[i]
    return TextureIm


parser = argparse.ArgumentParser()
parser.add_argument("root")
args = parser.parse_args()

folders = glob(osp(args.root, "*"))
for root in folders:
    print(root)
    image_path_list = [x for x in os.listdir(osp(root, "image")) if x.endswith("jpg")]
    mask = np.zeros((128*4, 128*6), dtype=np.float32) + 1e-8
    Textures = np.zeros((128*4, 128*6, 3), dtype=np.float32)
    if not os.path.exists(os.path.join(root, "texture")):
        os.mkdir(os.path.join(root, "texture"))
    for i, image_path in tqdm(enumerate(image_path_list), total=len(image_path_list)):
        IUV_path = osp(root, "densepose", image_path[:-4] + "_IUV.png")
        if not os.path.exists(IUV_path):
            continue
        im = cv2.imread(osp(root, "image", image_path))
        IUV = cv2.imread(IUV_path)
        texture = GetTexture(im, IUV,)
        out_path = osp(root, "texture", image_path[:-4] + ".png")
        cv2.imwrite(out_path, texture)
        Textures += texture
        mask += (texture.sum(2) != 0)

    texture_target = Textures / np.expand_dims(mask, 2)

    cv2.imwrite(osp(root, "texture_target.png"), texture_target)
