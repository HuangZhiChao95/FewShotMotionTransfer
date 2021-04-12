from DataSet import ReconstructDataSet, TransferDataSet
from torch.utils.data.dataloader import DataLoader
from PIL import Image
from tqdm import trange
from models.model import Model
import argparse
from torch.nn.parallel import DataParallel
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import os, cv2, traceback, shutil
import utils
import yaml

def create_finetune_set(root, samples=20):
    '''Sample images from the source person and create a small dataset for finetuning'''
    root = os.path.normpath(root)
    newroot = root+"_finetune"
    if os.path.exists(newroot):
        os.system("rm -r "+newroot)
    folder = root
    newfolder = os.path.join(folder.replace(root, newroot), "001")

    os.system("mkdir -p {}/image".format(newfolder))
    os.system("mkdir -p {}/body".format(newfolder))
    os.system("mkdir -p {}/densepose".format(newfolder))
    os.system("mkdir -p {}/texture".format(newfolder))
    os.system("mkdir -p {}/segmentation".format(newfolder))
    filename = "finetune_samples.txt"

    with open(os.path.join(folder, filename)) as f:
        filelist = f.readlines()
        filelist = filelist[:samples]

    for name in filelist:
        name = name.strip()
        shutil.copyfile(os.path.join(folder, "image", name + ".jpg"), os.path.join(newfolder, "image", name + ".jpg"))
        shutil.copyfile(os.path.join(folder, "body", name + ".png"), os.path.join(newfolder, "body", name + ".png"))
        shutil.copyfile(os.path.join(folder, "densepose", name + "_IUV.png"), os.path.join(newfolder, "densepose", name + "_IUV.png"))
        shutil.copyfile(os.path.join(folder, "texture", name+".png"), os.path.join(newfolder, "texture", name+".png"))
        shutil.copyfile(os.path.join(folder, "segmentation", name+".png"), os.path.join(newfolder, "segmentation", name+".png"))

    with open(os.path.join(newfolder, "image_list.txt"), "w") as f:
        for name in filelist:
            f.write(name.strip()+"\n")

    return newroot


def finetune(config, writer, device_idxs=[0]):

    config['phase'] = 'train'
    newroot = create_finetune_set(config['source_root'], config['finetune_sample'])
    dataset = ReconstructDataSet(newroot, config, list_name="image_list.txt")
    data_loader = DataLoader(dataset, batch_size=config['batchsize'], num_workers=16, pin_memory=True, shuffle=True, drop_last=False)

    background = Image.open(os.path.join(config['source_root'], config['background']))

    image_size = config['resize']
    background = background.resize((image_size, image_size))
    background = torch.from_numpy(np.asarray(background).transpose((2, 0, 1)).astype(np.float32)/255).unsqueeze(0)

    model = Model(config, "finetune")
    iter_loader = iter(data_loader)
    model.prepare_for_finetune(next(iter_loader), background)
    device = torch.device("cuda:" + str(device_idxs[0]))
    model = model.to(device)
    model.background_start = model.background_start.to(device)
    model = DataParallel(model, device_idxs)
    model.train()

    totol_step = 0
    for epoch in trange(config['epochs']):
        iterator = tqdm(enumerate(data_loader), total=len(data_loader))
        for i, data in iterator:

            data_gpu = {key: item.to(device) for key, item in data.items()}
            mask, fake_image, textures, body, cordinate, background, losses = model(data_gpu, "finetune")

            for key, item in losses.items():
                losses[key] = item.mean()
                writer.add_scalar("Loss/"+key, losses[key], totol_step)

            if totol_step < config['finetune_coor_step']:
                model.module.optimizer_G.zero_grad()
            model.module.optimizer_texture_stack.zero_grad()
            loss_G = losses.get("loss_G_L1", 0) + losses.get("loss_G_GAN", 0) + losses.get("loss_G_GAN_Feat", 0) + losses.get("loss_G_mask", 0) + losses.get("loss_texture", 0) * config['l_texture'] + losses.get("loss_coordinate", 0) * config['l_coordinate'] + losses.get("loss_mask", 0) * config['l_mask'] + losses.get("loss_background", 0) * config['l_background']
            loss_G.backward()

            if totol_step < config['finetune_coor_step']:
                model.module.optimizer_G.step()

            model.module.optimizer_texture_stack.step()
            writer.add_scalar("Loss/G", loss_G, totol_step)
            totol_step+=1

        if config['display'] and epoch % config['display_frep'] == 0:
            body_sum = body.sum(dim=1, keepdim=True)
            B, _, H, W = cordinate.size()
            cordinate_zero = torch.zeros((B, 1, H, W), dtype=torch.float32, device=cordinate.device)
            mask_label = torch.argmax(mask, dim=1, keepdim=True)

            cordinate_u = torch.gather(dim=1, index=mask_label, input=torch.cat((torch.zeros_like(cordinate_zero), cordinate[:, :24]), dim=1))
            cordinate_v = torch.gather(dim=1, index=mask_label, input=torch.cat((torch.zeros_like(cordinate_zero), cordinate[:, 24:]), dim=1))
            writer.add_images("Cordinate/U", utils.colorize(cordinate_u)*data_gpu["foreground"].expand_as(data["image"]).to(torch.float32), totol_step, dataformats="NCHW")
            writer.add_images("Cordinate/V", utils.colorize(cordinate_v)*data_gpu["foreground"].expand_as(data["image"]).to(torch.float32), totol_step, dataformats="NCHW")
            b, _, h, w = textures.size()
            writer.add_images("Texture", torch.clamp(textures[0].view(24, 3, h, w), 0, 1), totol_step, dataformats="NCHW")
            b, c, h, w = data_gpu["texture"][0].size()
            writer.add_images("Texture_Input", data_gpu["texture"][0].view(b, 24, 3, h, w).view(b * 24, 3, h, w), totol_step, dataformats="NCHW")
            writer.add_images("Mask/Generate", (1 - mask[:,0]).unsqueeze(1), totol_step, dataformats='NCHW')
            writer.add_images("Mask/Individual", utils.d_colorize(mask_label), totol_step, dataformats="NCHW")
            writer.add_images("Mask/Target", data["foreground"], totol_step, dataformats="NCHW")
            writer.add_images("Image/Fake", torch.clamp(fake_image, 0, 1), totol_step, dataformats="NCHW")
            writer.add_images("Image/True", data["image"], totol_step, dataformats="NCHW")

            writer.add_images("Input/body", body_sum, totol_step, dataformats="NCHW")
            writer.add_images("Background", torch.clamp(background, 0, 1), totol_step, dataformats="NCHW")

    torch.cuda.empty_cache()
    os.system("rm -r "+newroot)
    return model


def inference(model, config, device_idxs=[0]):
    config['phase'] = 'inference'
    config['hflip'] = False
    dataset = TransferDataSet(config['target_root'], config['source_root'], config)
    data_loader = DataLoader(dataset, batch_size=config['batchsize'], num_workers=4, pin_memory=True, shuffle=False)

    device = torch.device("cuda:" + str(device_idxs[0]))
    image_size = config['resize']

    fourcc = cv2.VideoWriter_fourcc(*'PIM1')
    folder = os.path.join(config["output_folder"], config["name"])
    if not os.path.exists(folder):
        os.system("mkdir -p "+folder)
    writer = cv2.VideoWriter(os.path.join(folder, config['output_name']), fourcc, 24, (image_size*3, image_size))

    with torch.no_grad():
        try:
            iterator = tqdm(enumerate(data_loader), total=len(data_loader))
            for i, data in iterator:
                data_gpu = {key: item.to(device) for key, item in data.items()}
                mask, fake_image, real_image, body, coordinate, texture = model(data_gpu, "inference")

                label = utils.d_colorize(data_gpu["body"]).cpu().numpy()
                B, _, H, W = coordinate.size()

                real_image = data['image'].cpu().numpy()
                fake_image = np.clip(fake_image.cpu().numpy(), 0, 1)

                outputs = np.concatenate((real_image, label, fake_image), axis=3)
                for output in outputs:
                    write_image = (output[::-1].transpose((1, 2, 0)) * 255).astype(np.uint8)
                    writer.write(write_image)

        except Exception as e:
            print(traceback.format_exc())
            writer.release()

        writer.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--device", type=int, nargs='+')
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--source_root", default=None)
    parser.add_argument("--target_root", default=None)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.source_root is not None:
        config['source_root'] = args.source_root
    if args.target_root is not None:
        config['target_root'] = args.target_root

    if config['output_name'] is None:
        config['output_name'] = "src_{}_to_{}.mp4".format(os.path.normpath(config['source_root']), os.path.normpath(config['target_root']))

    writer = SummaryWriter(log_dir=os.path.join(config['checkpoint_path'], config["name"], "finetune", config["output_name"][:-4]), comment=config['name'])
    model = finetune(config, writer, args.device)
    inference(model, config, args.device)