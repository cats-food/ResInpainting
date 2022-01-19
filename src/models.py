import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .networks import InpaintGenerator1, InpaintGenerator2,  Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name  # name有两种值，'InpaintingModel1' 和 'InpaintingModel2'。取决于是继承该BaseModel的是 InpaintingModel1类 还是InpaintingModel2类（见下）
        self.config = config
        self.iteration = 0

        if name == 'InpaintingModel1':
            self.gen_weights_path = config.InpaintingModel1_G_LOAD_PATH
            if config.ENABLE_D1:
                self.dis_weights_path = config.InpaintingModel1_D_LOAD_PATH
        elif name == 'InpaintingModel2':
            self.gen_weights_path = config.InpaintingModel2_G_LOAD_PATH
            self.dis_weights_path = config.InpaintingModel2_D_LOAD_PATH
        else:
            raise Exception('ysy: bug!')

    def load(self):  # 加载预训练模型的参数
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator from path %s ...' % (self.name, self.gen_weights_path))

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)  # 此处data是个字典，只有俩键： 键'iteration'的值是迭代次数；键'generator'的值是生成器的参数
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'],
                                           strict=False)
            self.iteration = data['iteration']
        else:
            raise Exception('ysy: no model found at: ' + self.gen_weights_path)

        # load discriminator only when training
        if self.config.MODE == 1:
            try:
                if not os.path.exists(self.dis_weights_path):
                    raise Exception('ysy: no model found at: ' + self.dis_weights_path)

                print('Loading %s discriminator from path %s ...' % (self.name, self.dis_weights_path))
                if torch.cuda.is_available():
                    data = torch.load(self.dis_weights_path)
                else:
                    data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

                self.discriminator.load_state_dict(data['discriminator'])
            except AttributeError:
                print('ysy prompt: %s has no attribute \'dis_weights_path\' thus it will not be loaded' % self.name)

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, os.path.join(self.config.G_SAVE_PATH,
                        self.name + '_G_' + ('%08d' % self.iteration) + '.pth'))
        try:
            torch.save({
                'discriminator': self.discriminator.state_dict()
            }, os.path.join(self.config.D_SAVE_PATH,
                            self.name + '_D_' + ('%08d' % self.iteration) + '.pth'))
        except AttributeError:
            print('ysy prompt: %s has no attribute \'discriminator\' thus it will not be saved' % self.name)


class InpaintingModel1(BaseModel):
    def __init__(self, config):
        super(InpaintingModel1, self).__init__('InpaintingModel1', config)
        generator = InpaintGenerator1()
        if config.ENABLE_D1:
            discriminator = Discriminator(in_channels=3, use_sigmoid= not (config.GAN_LOSS == 'hinge' or config.GAN_LOSS == 'wgan'))   # hinge 和 wgan 不用sigmoid
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            if config.ENABLE_D1:
                discriminator = nn.DataParallel(discriminator, config.GPU)

        l1_loss = nn.L1Loss()
        # perceptual_loss = PerceptualLoss()
        # style_loss = StyleLoss()


        self.add_module('generator', generator)
        self.add_module('l1_loss', l1_loss)
        self.gen_optimizer = optim.Adam(params=generator.parameters(), lr=float(config.LR),betas=(config.BETA1, config.BETA2))

        if config.ENABLE_D1:
            self.add_module('discriminator', discriminator)
            self.adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
            self.dis_optimizer = optim.Adam(params=discriminator.parameters(), lr=float(config.LR) * float(config.D2G_LR), betas=(config.BETA1, config.BETA2))


    def process(self, images, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        # self.dis_optimizer.zero_grad()  # this line has been moved downward

        # process outputs
        outputs = self(images, masks)  # 喂给generator
        gen_loss = 0
        dis_loss = torch.tensor(float('nan'))
        gen_gan_loss = torch.tensor(float('nan'))

        if self.config.ENABLE_D1:
            self.dis_optimizer.zero_grad()
            dis_loss = 0
            # discriminator loss
            dis_input_real = images
            dis_input_fake = outputs.detach()
            dis_real, _ = self.discriminator(dis_input_real)                    # in: [rgb(3)]
            dis_fake, _ = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
            dis_real_loss = self.adversarial_loss(dis_real, True, True)
            dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2

            # generator adversarial loss
            gen_input_fake = outputs
            gen_fake, _ = self.discriminator(gen_input_fake)                    # in: [rgb(3)]
            gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
            gen_loss += gen_gan_loss

        gen_l1_loss = 6 * torch.mean(torch.abs(outputs - images) * masks) + torch.mean(torch.abs(outputs - images) * (1 - masks))  # 6 hole_loss + 1 valid_loss
        gen_loss += gen_l1_loss * self.config.L1_LOSS_WEIGHT

        logs = [
            ("l_d1", dis_loss.item()),
            ("l_g1", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            # ("l_per", gen_content_loss.item()),
            # ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs


    def forward(self, images, masks):
        images_masked = (images * (1 - masks).float()) + masks
        outputs = self.generator(images_masked, 1 - masks)
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        if self.config.ENABLE_D1:
            dis_loss.backward()
            self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()


class InpaintingModel2(BaseModel):
    def __init__(self, config):
        super(InpaintingModel2, self).__init__('InpaintingModel2', config)

        generator = InpaintGenerator2()
        discriminator = Discriminator(in_channels=3, use_sigmoid=not (
                    config.GAN_LOSS == 'hinge' or config.GAN_LOSS == 'wgan'))  # hinge 和 wgan 不用sigmoid
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        if self.config.GAN_LOSS == 'wgan':  # wgan使用rmsprop
            self.gen_optimizer = optim.RMSprop(params=generator.parameters(),
                                               lr=float(config.LR))  # betas=(config.BETA1, config.BETA2))
            self.dis_optimizer = optim.RMSprop(params=discriminator.parameters(), lr=float(config.LR) * float(
                config.D2G_LR))
        else:  # 默认使用adam
            self.gen_optimizer = optim.Adam(params=generator.parameters(), lr=float(config.LR),
                                            betas=(config.BETA1, config.BETA2))
            self.dis_optimizer = optim.Adam(params=discriminator.parameters(),
                                            lr=float(config.LR) * float(config.D2G_LR),
                                            betas=(config.BETA1, config.BETA2))


    def process(self, images_gt, images_in, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs = self(images_gt, images_in, masks)  # 喂给generator
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = images_gt
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)  # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)  # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)  # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # generator l1 loss
        # gen_l1_loss = self.l1_loss(outputs, images_gt) / torch.mean(masks)  # original
        gen_l1_loss = 6 * torch.mean(torch.abs(outputs - images_gt) * masks) + torch.mean(torch.abs(outputs - images_gt) * (1 - masks))  # 6 hole_loss + 1 valid_loss
        gen_loss += gen_l1_loss * self.config.L1_LOSS_WEIGHT

        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images_gt)
        gen_loss += gen_content_loss * self.config.PERCEP_LOSS_WEIGHT

        # generator style loss
        gen_style_loss = self.style_loss(outputs * masks, images_gt * masks)
        gen_loss += gen_style_loss * self.config.STYLE_LOSS_WEIGHT

        # create logs
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images_gt, images_in, masks):

        images_in = (F.interpolate(images_in, scale_factor=2, mode='bicubic') * masks) + (images_gt * (1 - masks))  # 低分辨率图 上采样后 和高分辨率gt融合
        outputs = self.generator(images_in, 1 - masks)  # 第2个参数mask被ysy修改为1-masks，因为InpaintGenerator中pconv接受的mask的格式是源1洞0. 而mask本身是源0洞1
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward(retain_graph = True)
        self.gen_optimizer.step()
