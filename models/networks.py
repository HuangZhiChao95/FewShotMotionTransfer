import torch
from torch import nn

from models.blocks import Conv2dBlock, ResBlocks
import torchvision.models as models


class Geometry_Generator(nn.Module):
    def __init__(self, hp):
        super(Geometry_Generator, self).__init__()
        nf = hp['nf']
        down_class = hp['n_downs_class']
        down_content = hp['n_downs_content']
        n_res_blks = hp['n_res_blks']
        latent_dim = hp['latent_dim']
        content_input_dim = hp['input_dim']
        self.enc_class_model = ClassModelEncoder(down_class, 3, nf, latent_dim, norm='none', activ='relu', pad_type='reflect')
        self.enc_content = ContentEncoder(down_content, n_res_blks, content_input_dim, nf, 'in', activ='relu', pad_type='reflect')
        self.dec = Decoder(down_content, n_res_blks, self.enc_content.output_dim, latent_dim, nf*4, res_norm='in', activ='relu', pad_type='reflect')
        self.weight_model = WeightEncoder(down_class, content_input_dim, nf, latent_dim, norm='none', activ='relu', pad_type='reflect')
        self.att = SelfAttention()


class ClassModelEncoder(nn.Module):
    def __init__(self, downs, ind_im, dim, latent_dim, norm, activ, pad_type):
        super(ClassModelEncoder, self).__init__()
        self.model = nn.ModuleList()
        self.model.append(Conv2dBlock(ind_im, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type))
        for i in range(2):
            self.model.append(Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type))
            dim *= 2
        for i in range(downs - 2):
            self.model.append(Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type))
        self.model.append(nn.AdaptiveAvgPool2d(1))
        self.model.append(nn.Conv2d(dim, latent_dim, 1, 1, 0))
        self.output_dim = dim
        self.downs = downs

    def forward(self, x):
        features = []
        for i, m in enumerate(self.model):
            x = m(x)
            if 1 <= i <= self.downs:
                features.append(x)
        features.reverse()
        return x, features


class WeightEncoder(nn.Module):
    def __init__(self, downs, input_dim, dim, latent_dim, norm, activ, pad_type):
        super(WeightEncoder, self).__init__()
        self.model = nn.ModuleList()
        self.model.append(Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type))
        for i in range(2):
            self.model.append(Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type))
            dim *= 2
        for i in range(downs - 2):
            self.model.append(Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type))
        self.model.append(nn.AdaptiveAvgPool2d(1))
        self.model.append(nn.Conv2d(dim, latent_dim, 1, 1, 0))
        self.output_dim = dim
        self.downs = downs

    def forward(self, x):
        features = []
        for i, m in enumerate(self.model):
            x = m(x)
            if 1 <= i <= self.downs:
                features.append(x)
        features.reverse()
        return x, features


class ContentEncoder(nn.Module):
    def __init__(self, downs, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(downs):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def forward(self, weight_codes, weight_features, class_weight_codes, class_weight_features, label_codes, label_features):
        n_in, H, W = list(weight_codes.size())[1:]

        if n_in > H*W:
            f = torch.einsum('niab,nicd->nabcd', class_weight_codes, weight_codes)
            label_code = torch.einsum('nabcd,nicd->niab', f, label_codes)
        else:
            f = torch.einsum('nihw,njhw->nij', weight_codes, label_codes)
            label_code = torch.einsum('nij,nihw->njhw', f, class_weight_codes)

        features = []
        for i in range(len(weight_features)):
            n_in, H, W = list(weight_features[i].size())[1:]

            if n_in > H * W:
                f = torch.einsum('niab,nicd->nabcd', class_weight_features[i], weight_features[i])
                label_feature = torch.einsum('nabcd,nicd->niab', f, label_features[i])
            else:
                f = torch.einsum('nihw,njhw->nij', weight_features[i], label_features[i])
                label_feature = torch.einsum('nij,nihw->njhw', f, class_weight_features[i])
            features.append(label_feature)

        return label_code, features


class Decoder(nn.Module):
    def __init__(self, ups, n_res, dim, latent_dim, class_dim, res_norm, activ, pad_type):
        super(Decoder, self).__init__()
        self.model = []
        self.model += [nn.Conv2d(dim + latent_dim, dim, 1, 1, 0)]
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]

        self.mask = []
        self.coordinate = []

        for i in range(ups):
            self.mask += [nn.Upsample(scale_factor=2),
                          Conv2dBlock(dim+class_dim, dim // 2, 5, 1, 2, norm='in', activation=activ, pad_type=pad_type)]
            self.coordinate += [nn.Upsample(scale_factor=2),
                                Conv2dBlock(dim+class_dim, dim // 2, 5, 1, 2, norm='in', activation=activ, pad_type=pad_type)]
            dim //= 2
            if ups - i <= 2:
                class_dim //= 2

        self.mask += [Conv2dBlock(dim, 25, 7, 1, 3, norm='none', activation='none', pad_type=pad_type)]
        self.coordinate += [Conv2dBlock(dim, 48, 7, 1, 3, norm='none', activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.mask = nn.ModuleList(self.mask)
        self.coordinate = nn.ModuleList(self.coordinate)

    def forward(self, x, class_codes, class_features):
        _, _, h, w = x.size()
        class_codes = class_codes.expand(-1, -1, h, w)
        latent = self.model(torch.cat((x, class_codes), dim=1))
        mask = latent
        coordinate = latent
        for i in range(len(self.mask)):
            if i % 2 == 0 and i//2 < len(class_features):
                mask = torch.cat((mask, class_features[i//2]), dim=1)
                coordinate = torch.cat((coordinate, class_features[i//2]), dim=1)
            mask = self.mask[i](mask)
            coordinate = self.coordinate[i](coordinate)
        coordinate = torch.clamp(coordinate, 0, 1)

        return mask, coordinate


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):

        super(GatedConv2d, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.sigmoid = torch.nn.Sigmoid()
        self.activation = activation

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        x = self.activation(x) * self.gated(mask)
        return x


class Texture_Generator(nn.Module):
    def __init__(self, config, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super().__init__()
        ngf = config['nf']
        input_nc = config['input_dim']
        output_nc = config['output_dim']
        n_downsampling = config["n_downs"]
        n_blocks =config['n_res_blks']
        activation = nn.ReLU(True)

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            GatedConv2d(input_nc, ngf, kernel_size=7, padding=0, activation=activation),
            norm_layer(ngf)
        )

        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Sequential())
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            block = nn.Sequential(
                GatedConv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, activation=activation),
                norm_layer(ngf * mult * 2)
            )
            self.encoder.append(block)

        resblks = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            resblks += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        self.resblks = nn.Sequential(*resblks)

        self.decoder = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            block = nn.Sequential(
                nn.Upsample(scale_factor=2),
                GatedConv2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=1, padding=1, activation=activation),
                norm_layer(ngf * mult // 2)
            )
            self.decoder.append(block)

        self.conv_last = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)
        )

    def forward(self, input):
        b, n, c, h, w = input.size()
        input = input.view(b*n, c, h, w)
        x = self.conv1(input)

        for block in self.encoder:
            x = block(x)
        x = self.resblks(x)
        _, c, h, w = x.size()
        x = x.view(b, n, c, h, w)
        x = x.mean(dim=1)

        for i, block in enumerate(self.decoder):
            x = block(x)

        x = self.conv_last(x)
        return torch.clamp(x, 0, 1)

    def get_feat(self, input):
        b, n, c, h, w = input.size()
        input = input.view(b*n, c, h, w)
        x = self.conv1(input)

        for block in self.encoder:
            x = block(x)
            _, c, h, w = x.size()
        x = self.resblks(x)
        _, c, h, w = x.size()
        x = x.view(b, n, c, h, w)
        x = x.mean(dim=1).detach()

        return x

    def forward_feat(self, x):
        for i, block in enumerate(self.decoder):
            x = block(x)

        x = self.conv_last(x)
        return torch.clamp(x, 0, 1)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
