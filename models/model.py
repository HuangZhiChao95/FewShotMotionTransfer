import torch
import torch.nn as nn

from models.networks import Geometry_Generator, Texture_Generator, VGGLoss
import os
import random
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun


class Model(nn.Module):

    def __init__(self, config, phase):
        super(Model, self).__init__()

        self.config = config
        self.phase = phase
        self.save_dir = os.path.join(config['checkpoint_path'], config['name'])
        if not os.path.exists(self.save_dir):
            os.system("mkdir -p "+self.save_dir)

        if self.phase != 'pretrain_texture':
            self.generator = Geometry_Generator(config['G'])
        if self.phase != 'pretrain':
            self.texture_generator = Texture_Generator(config['Texture_G'])

        self.apply(weights_init('kaiming'))
        self.L1Loss = nn.L1Loss(reduction='none')
        self.L2Loss = nn.MSELoss(reduction='none')
        self.PerceptualLoss = VGGLoss()

    def restore_network(self):
        self.load_network(self.generator, 'G', self.config['pretrain_name'])
        self.load_network(self.texture_generator, 'Texture_G', self.config['pretrain_name'])

    def prepare_for_pretrain(self):
        self.lr = self.config['lr']

        self.EntropyLoss = nn.CrossEntropyLoss()
        G_params = list(self.generator.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr=self.lr, betas=(0.5, 0.999))
        self.scheduler_G = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_G, self.config["lr_milestone"], gamma=0.5)

    def prepare_for_texture(self):
        self.lr_T = self.config['lr_T']
        T_params = self.texture_generator.parameters()
        self.optimizer_T = torch.optim.Adam(T_params, lr=self.lr_T, betas=(0.5, 0.999))
        self.scheduler_T = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_T, self.config["lr_milestone"],  gamma=0.5)

    def prepare_for_train(self, n_class=10):
        self.prepare_for_pretrain()
        self.prepare_for_texture()
        self.restore_network()
        texture_stack = torch.zeros((n_class, 72, self.config['texture_size'], self.config['texture_size']))
        self.texture_stack = nn.Parameter(texture_stack)
        self.texture_list = [False for i in range(n_class)]
        self.optimizer_texture_stack = torch.optim.Adam([self.texture_stack], lr=self.lr_T, betas=(0.5, 0.999))

    def prepare_for_finetune(self, data, background):
        self.prepare_for_train()
        with torch.no_grad():
            part_texture = data["texture"][:self.config['num_texture']]
            b, n, c, h, w = part_texture.size()
            part_texture = part_texture.view(b * n, c, h, w)
            part_texture = torch.nn.functional.interpolate(part_texture, (self.config['texture_size'], self.config['texture_size']))
            part_texture = part_texture.view(1, b*n, c, self.config['texture_size'], self.config['texture_size'])
            texture = self.texture_generator.get_feat(part_texture)
        self.texture_stack = nn.Parameter(texture)
        self.texture_feature = []
        self.background = nn.Parameter(background)
        self.background_start = background
        self.optimizer_texture_stack = torch.optim.Adam([self.texture_stack, self.background], lr=self.lr_T, betas=(0.5, 0.999))

    def _encode_label(self, label_map, nc, dropout, dropout_all):
        size = label_map.size()
        oneHot_size = (size[0], nc+1, size[2], size[3])
        input_label = torch.zeros(torch.Size(oneHot_size), device=label_map.device, dtype=torch.float32)
        input_label = input_label.scatter_(1, label_map.long(), 1.0)
        prob = torch.rand(nc, dtype=torch.float32, device=label_map.device) < dropout
        prob_all = random.random()<dropout_all
        prob = prob.view(1, len(prob), 1, 1).expand_as(input_label[:,1:]).to(torch.float32)
        encode_label = input_label[:,1:]*prob*prob_all
        return encode_label

    def _get_input_pose(self, data):
        body = self._encode_label(data["body"], self.config["body_nc"], 1.0, 1.0)
        class_body = self._encode_label(data['class_body'], self.config["body_nc"], 1.0, 1.0)
        return body, class_body

    def get_image(self, mask, cordinate, texture):

        prob_mask = F.softmax(mask, dim=1)
        B, _, H, W= cordinate.size()
        u = cordinate[:,:24]*2-1
        v = cordinate[:,24:]*2-1

        grids = torch.stack((v,u), dim=4)
        textures = []
        b, c, h, w = texture.size()
        texture = texture.view(b, 24, c//24, h, w)
        for i in range(B):
            textures.append(nn.functional.grid_sample(texture[i], grids[i]))

        textures = torch.stack(textures, dim=0)
        fake_image = (prob_mask[:,1:].unsqueeze(2).expand_as(textures)*textures).sum(dim=1)

        return prob_mask, fake_image

    def pretrain(self, data):
        input, class_input = self._get_input_pose(data)

        pose_code = self.generator.enc_content(input)
        label_codes, label_features = self.generator.enc_class_model(data["class_image"]*data["class_foreground"].expand(-1,3,-1,-1))
        weight_codes, weight_features = self.generator.weight_model(input)
        class_weight_codes, class_weight_features = self.generator.weight_model(class_input)
        label_code, label_feature = self.generator.att(weight_codes, weight_features, class_weight_codes, class_weight_features, label_codes, label_features)
        mask, coordinate = self.generator.dec(pose_code, label_code, label_feature)

        mask_loss = self.EntropyLoss(mask, data["mask"])
        target_U = data["U"].expand(-1,24,-1,-1)
        target_V = data["V"].expand(-1,24,-1,-1)
        target_coordinate = torch.cat((target_U, target_V), dim=1)
        coordinate_per_pixel_loss = self.L1Loss(coordinate, target_coordinate)
        label_mask = self._encode_label(data["mask"].unsqueeze(1), 24, 1.0, 1.0).repeat(1,2,1,1).to(torch.float32)
        coordinate_diff = (coordinate_per_pixel_loss*label_mask).sum(dim=1)
        coordinate_loss = (coordinate_diff).mean(dim=0).sum()

        return mask_loss, coordinate_loss, F.softmax(mask, dim=1), coordinate, input, label_mask

    def pretrain_texture(self, data):
        texture = data['texture']
        b, n, c, h, w = texture.size()
        texture = texture.view(b*n, c, h, w)
        texture = torch.nn.functional.interpolate(texture, (self.config['texture_size'], self.config['texture_size']))
        texture = texture.view(b, n, c, self.config['texture_size'], self.config['texture_size'])

        target = data['texture'].view(b*n, c, h, w)
        target = torch.nn.functional.interpolate(target, (self.config['texture_size'], self.config['texture_size']))
        target = target.view(b, n, c, self.config['texture_size'], self.config['texture_size'])

        texture_output = self.texture_generator(texture)
        b, n, c, h, w = target.size()
        mask = target.view(b, n, 24, 3, h, w).sum(dim=3, keepdim=True) > 1e-3
        mask = mask.repeat(1, 1, 1, 3, 1, 1).view(b, n, c, h, w).to(torch.float32)
        per_pixel_loss = self.L1Loss(texture_output.unsqueeze(1).expand(-1, n, -1, -1, -1), target)
        loss_pixel = (per_pixel_loss*mask).mean(dim=0).sum()

        return loss_pixel, texture, texture_output, target

    def train_network(self, data, mode='UV'):
        input, class_input = self._get_input_pose(data)

        pose_code = self.generator.enc_content(input)
        label_codes, label_features = self.generator.enc_class_model(data["class_image"]*data["class_foreground"].expand(-1,3,-1,-1))
        weight_codes, weight_features = self.generator.weight_model(input)
        class_weight_codes, class_weight_features = self.generator.weight_model(class_input)
        label_code, label_feature = self.generator.att(weight_codes, weight_features, class_weight_codes, class_weight_features, label_codes, label_features)
        mask, coordinate = self.generator.dec(pose_code, label_code, label_feature)

        if mode == 'UV':
            label = int(data['class'][0])
            if self.texture_list[label]:
                texture = self.texture_stack[label].unsqueeze(0)
            else:
                with torch.no_grad():
                    part_texture = data["texture"][:self.config['num_texture']]
                    b, n, c, h, w = part_texture.size()
                    part_texture = part_texture.view(b * n, c, h, w)
                    part_texture = torch.nn.functional.interpolate(part_texture, (self.config['texture_size'], self.config['texture_size']))
                    part_texture = part_texture.view(1, b*n, c, self.config['texture_size'], self.config['texture_size'])
                    self.texture_stack.data[label] = self.texture_generator(part_texture)[0].data
                texture = self.texture_stack[label].unsqueeze(0)
                self.texture_list[label] = True
        else:
            part_texture = data["texture"][:self.config['num_texture']]
            b, n, c, h, w = part_texture.size()
            part_texture = part_texture.view(b * n, c, h, w)
            part_texture = torch.nn.functional.interpolate(part_texture, (self.config['texture_size'], self.config['texture_size']))
            part_texture = part_texture.view(1, b * n, c, self.config['texture_size'], self.config['texture_size'])
            texture = self.texture_generator(part_texture)

        texture = texture.repeat(self.config['batchsize'], 1, 1, 1)
        image = data["image"] * data["foreground"].expand_as(data["image"]).to(torch.float32)
        prob_mask, fake_image = self.get_image(mask, coordinate, texture)

        losses = self.create_loss_train(mask, data["foreground"], fake_image, image)
        losses['loss_coordinate'] = self.coordinate_constraint(coordinate, data)
        losses['loss_mask'] = self.mask_constraint(mask, data)
        if mode != 'UV':
            losses['loss_texture'] = self.texture_constraint(texture[0].unsqueeze(0), part_texture)

        return prob_mask.detach(), fake_image.detach(), texture.detach(), input, coordinate.detach(), losses

    def finetune(self, data):
        input, class_input = self._get_input_pose(data)

        pose_code = self.generator.enc_content(input)
        label_codes, label_features = self.generator.enc_class_model(data["class_image"]*data["class_foreground"].expand(-1,3,-1,-1))
        weight_codes, weight_features = self.generator.weight_model(input)
        class_weight_codes, class_weight_features = self.generator.weight_model(class_input)
        label_code, label_feature = self.generator.att(weight_codes, weight_features, class_weight_codes, class_weight_features, label_codes, label_features)
        mask, coordinate = self.generator.dec(pose_code, label_code, label_feature)

        texture = self.texture_generator.forward_feat(self.texture_stack).repeat(self.config['batchsize'], 1, 1, 1)
        prob_mask, fake_image = self.get_image(mask, coordinate, texture)
        mask_foreground = 1 - prob_mask[:, 0].unsqueeze(1).expand(-1, 3, -1, -1).to(torch.float32)

        pre_image = mask_foreground * fake_image + (1 - mask_foreground) * self.background.expand_as(fake_image)

        losses = self.create_loss_finetune(mask, data["foreground"], pre_image, data['image'])
        losses['loss_coordinate'] = self.coordinate_constraint(coordinate, data)
        losses['loss_mask'] = self.mask_constraint(mask, data)
        losses['loss_background'] = self.L1Loss(self.background, self.background_start).mean()
        return prob_mask.detach(), pre_image.detach(), texture.detach(), input, coordinate.detach(), self.background, losses

    def inference(self, data):
        input, class_input = self._get_input_pose(data)

        pose_code = self.generator.enc_content(input)
        label_codes, label_features = self.generator.enc_class_model(data["class_image"]*data["class_foreground"].expand(-1,3,-1,-1))
        weight_codes, weight_features = self.generator.weight_model(input)
        class_weight_codes, class_weight_features = self.generator.weight_model(class_input)
        label_code, label_feature = self.generator.att(weight_codes, weight_features, class_weight_codes, class_weight_features, label_codes, label_features)
        mask, coordinate = self.generator.dec(pose_code, label_code, label_feature)

        texture = self.texture_generator.forward_feat(self.texture_stack).repeat(self.config['batchsize'], 1, 1, 1)

        image = data["image"] * data["foreground"].expand_as(data["image"]).to(torch.float32)

        prob_mask, fake_image = self.get_image(mask, coordinate, texture)
        mask_foreground = 1 - prob_mask[:, 0].unsqueeze(1).expand(-1, 3, -1, -1).to(torch.float32)

        pre_image = mask_foreground * fake_image + (1 - mask_foreground) * self.background.expand_as(fake_image)

        return prob_mask.detach(), pre_image.detach(), image, input, coordinate.detach(), texture

    def create_loss_train(self, mask, target_binary, fake_image, real_image):
        loss = {}

        mask_prob = F.log_softmax(mask, dim=1)
        binary_prob = torch.stack((mask_prob[:, 0].unsqueeze(1), ((1 - mask_prob[:, 0].exp().unsqueeze(1)) + 1e-8).log()), dim=1)
        loss["loss_G_mask"] = F.nll_loss(binary_prob, target_binary)

        loss_G_L1 = self.L1Loss(fake_image, real_image)*(target_binary.expand(-1,3,-1,-1))
        loss["loss_G_L1"] = loss_G_L1.sum()/target_binary.expand(-1,3,-1,-1).sum() + self.PerceptualLoss(fake_image, real_image) * self.config['l_vgg']

        return loss

    def create_loss_finetune(self, mask, target_binary, fake_image, real_image):
        loss = {}

        mask_prob = F.log_softmax(mask, dim=1)
        binary_prob = torch.stack((mask_prob[:, 0].unsqueeze(1), ((1 - mask_prob[:, 0].exp().unsqueeze(1)) + 1e-8).log()), dim=1)
        loss["loss_G_mask"] = F.nll_loss(binary_prob, target_binary)

        loss_G_L1 = self.L1Loss(fake_image, real_image)
        loss["loss_G_L1"] = loss_G_L1.mean() + self.PerceptualLoss(fake_image, real_image) * self.config['l_vgg']

        return loss

    def texture_constraint(self, texture, part_texture):
        b, n, c, h, w = part_texture.size()
        texture = texture.unsqueeze(1).expand(-1,n,-1,-1,-1)
        mask = (part_texture.view(b,n,24,3,h,w).sum(dim=3, keepdim=True)>1e-3).repeat(1,1,1,3,1,1).view(b,n,c,h,w)
        loss = (self.L1Loss(texture, part_texture) * mask).mean()

        return loss

    def coordinate_constraint(self, coordinate, data):
        target_U = data["U"].expand(-1,24,-1,-1)
        target_V = data["V"].expand(-1,24,-1,-1)
        target_coordinate = torch.cat((target_U, target_V), dim=1)
        coordinate_per_pixel_loss = self.L1Loss(coordinate, target_coordinate)
        label_mask = self._encode_label(data["mask"].unsqueeze(1), 24, 1.0, 1.0).repeat(1,2,1,1).to(torch.float32)
        coordinate_diff = (coordinate_per_pixel_loss*label_mask).sum(dim=1)
        coordinate_loss = coordinate_diff.mean()

        return coordinate_loss

    def mask_constraint(self, mask, data):
        mask_loss = F.cross_entropy(mask, data["mask"], reduction='none')
        mask_loss = mask_loss * (data['mask'] > 0)

        return mask_loss.mean()

    def forward(self, data, phase):
        if phase == 'pretrain':
            return self.pretrain(data)
        if phase == 'pretrain_texture':
            return self.pretrain_texture(data)
        if phase == 'train_UV':
            return self.train_network(data, "UV")
        if phase == 'train_texture':
            return self.train_network(data, "texture")
        if phase == 'inference':
            return self.inference(data)
        if phase == 'finetune':
            return self.finetune(data)

    def save(self, name):
        if hasattr(self, 'generator'):
            self.save_network(self.generator, 'G', name)
        if hasattr(self, 'texture_generator'):
            self.save_network(self.texture_generator, 'Texture_G', name)
        if hasattr(self, 'texture_stack'):
            save_filename = '%s_net_%s.npy' % ("texture_stack", name)
            save_path = os.path.join(self.save_dir, save_filename)
            np.save(save_path, self.texture_stack.detach().cpu().numpy())

    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        print("load "+save_filename)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)

        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise ('Generator must exist!')
        else:
            try:
                network.load_state_dict(torch.load(save_path))
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        print(
                            'Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    not_initialized = set()

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])

                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

