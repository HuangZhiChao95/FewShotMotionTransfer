from DataSet import ReconstructDataSet
from torch.utils.data.dataloader import DataLoader
from tqdm import trange
from models.model import Model
import argparse
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import yaml

def pretrain(config, writer, device_idxs=[0]):

    data_loader = DataLoader(ReconstructDataSet(config['dataroot'], config), batch_size=config['batchsize'], num_workers=8, pin_memory=True, shuffle=True)

    device = torch.device("cuda:" + str(device_idxs[0]))
    model = Model(config, "pretrain_texture")
    model.prepare_for_texture()

    model = model.to(device)
    model = DataParallel(model, device_idxs)
    model.train()
    totol_step = 0
    for epoch in trange(config['epochs']):
        iterator = tqdm(enumerate(data_loader), total=len(data_loader))
        for i, data in iterator:

            data_gpu = {key: item.to(device) for key, item in data.items()}

            loss, texture_input, texture_output, texture_target = model(data_gpu, "pretrain_texture")

            loss = loss.mean()
            model.module.optimizer_T.zero_grad()
            loss.backward()
            model.module.optimizer_T.step()
            writer.add_scalar("Loss/Texture", loss, totol_step)

            if totol_step % config['display_freq'] == 0:
                b, n, c, h, w = texture_target.size()

                writer.add_images("Texture/output", torch.clamp(texture_output[0].view(24, 3, h, w), 0, 1), totol_step, dataformats="NCHW")
                writer.add_images("Texture/Target", texture_target[0].view(24*n, 3, h, w), totol_step, dataformats="NCHW")
                writer.add_images("Texture/input", texture_input[0].view(24*n, 3, h, w), totol_step, dataformats="NCHW")

            totol_step+=1

        model.module.save('latest')
        model.module.save(str(epoch+1))

        model.module.scheduler_T.step()

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--device", type=int, nargs='+')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    writer = SummaryWriter(log_dir=os.path.join(config['checkpoint_path'], config["name"], "pretrain_texture"), comment=config['name'])
    pretrain(config, writer, args.device)
