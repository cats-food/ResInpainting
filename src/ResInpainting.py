import os
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import InpaintingModel1, InpaintingModel2
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR


class ResInpainting():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 1:
            self.model_name = 'inpaint1'
        elif config.MODEL == 2:
            self.model_name = 'inpaint2'
        elif config.MODEL == 3:
            self.model_name = 'inpaint1-2'
        elif config.MODEL == 4:
            self.model_name = 'joint'

        self.debug = False
        self.inpaint_model1 = InpaintingModel1(config).to(config.DEVICE)
        self.inpaint_model2 = InpaintingModel2(config).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)
        else:  # train  mode
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)


        #self.samples_path = './samples'        # self.samples_path = os.path.join(config.PATH, 'samples')
        if os.path.exists(config.SAMPLE_PATH):
            self.samples_path = config.SAMPLE_PATH
        else:
            print('ysy warning: config.SAMPLE_PATH is invalid! samples will be saved to default path: ./samples')
            self.samples_path = './samples'

        #if config.RESULTS is not None:
        if os.path.exists(config.RESULTS):
            self.results_path = os.path.join(config.RESULTS)
        else:
            print('ysy warning: config.RESULTS is invalid! samples will be saved to default path: ./results')
            self.results_path = './results'  # self.results_path = os.path.join(config.PATH, 'results')

        if os.path.exists(config.LOG_PATH):
            self.log_file = os.path.join(config.LOG_PATH, 'log_' + self.model_name + '.dat')
        else:
            print('ysy warning: LOG_PATH is invalid! samples will be saved to default path: ./')
            self.log_file = 'log_' + self.model_name + '.dat'

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

    def load(self):
        if self.config.MODEL == 1:
            self.inpaint_model1.load()

        elif self.config.MODEL == 2:
            self.inpaint_model2.load()

        else:
            self.inpaint_model1.load()
            self.inpaint_model2.load()

    def save(self):
        if self.config.MODEL == 1:
            self.inpaint_model1.save()

        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            self.inpaint_model2.save()

        else:
            self.inpaint_model1.save()
            self.inpaint_model2.save()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                self.inpaint_model1.train()
                self.inpaint_model2.train()
                images2, images1, masks2, masks1 = self.cuda(*items)

                if model == 1:

                    outputs1, gen_loss, dis_loss, logs = self.inpaint_model1.process(images1, masks1)
                    outputs1_merged = (outputs1 * masks1) + (images1 * (1 - masks1))

                    psnr1 = self.psnr(self.postprocess(images1), self.postprocess(outputs1_merged))
                    mae1 = (torch.sum(torch.abs(images1 - outputs1_merged)) / torch.sum(images1)).float()
                    logs.append(('psnr1', psnr1.item()))
                    logs.append(('mae1', mae1.item()))

                    self.inpaint_model1.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model1.iteration


                elif model == 2:
                    outputs2, gen_loss, dis_loss, logs = self.inpaint_model2.process(images2, images1,  masks2) # inp2 输入为：高分辨真值，低分辨图，高分辨mask。低分辨图将在forward里上采样融合
                    outputs2_merged = (outputs2 * masks2) + (images2 * (1 - masks2))

                    # metrics
                    psnr2 = self.psnr(self.postprocess(images2), self.postprocess(outputs2_merged))
                    mae2 = (torch.sum(torch.abs(images2 - outputs2_merged)) / torch.sum(images2)).float()
                    logs.append(('psnr2', psnr2.item()))
                    logs.append(('mae2', mae2.item()))

                    # backward
                    self.inpaint_model2.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model2.iteration

                elif model == 3:

                    outputs1 = self.inpaint_model1(images1,masks1)
                    outputs1_merged = (outputs1 * masks1) + (images1 * (1 - masks1))
                    outputs2, gen_loss, dis_loss, logs = self.inpaint_model2.process(images2, outputs1_merged.detach(), masks2)
                    # inp2 输入为：高分辨真值，低分辨图，高分辨mask。低分辨图将在forward里上采样融合
                    outputs2_merged = (outputs2 * masks2) + (images2 * (1 - masks2))

                    # metrics
                    psnr2 = self.psnr(self.postprocess(images2), self.postprocess(outputs2_merged))
                    mae2 = (torch.sum(torch.abs(images2 - outputs2_merged)) / torch.sum(images2)).float()
                    logs.append(('psnr2', psnr2.item()))
                    logs.append(('mae2', mae2.item()))

                    # backward
                    self.inpaint_model2.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model2.iteration


                elif model == 4:
                    # train
                    outputs1, gen1_loss, dis1_loss, logs1 = self.inpaint_model1.process(images1,masks1)
                    outputs1_merged = (outputs1 * masks1) + (images1 * (1 - masks1))
                    outputs2, gen2_loss, dis2_loss, logs2 = self.inpaint_model2.process(images2, outputs1_merged, masks2)
                    outputs2_merged = (outputs2 * masks2) + (images2 * (1 - masks2))

                    # metrics
                    psnr1 = self.psnr(self.postprocess(images1), self.postprocess(outputs1_merged))
                    mae1 = (torch.sum(torch.abs(images1 - outputs1_merged)) / torch.sum(images1)).float()
                    psnr2 = self.psnr(self.postprocess(images2), self.postprocess(outputs2_merged))
                    mae2 = (torch.sum(torch.abs(images2 - outputs2_merged)) / torch.sum(images2)).float()

                    logs1.append(('psnr1', psnr1.item()))
                    logs1.append(('mae1', mae1.item()))
                    logs2.append(('psnr2', psnr2.item()))
                    logs2.append(('mae2', mae2.item()))
                    logs = logs1 + logs2

                    # backward
                    self.inpaint_model2.backward(gen2_loss, dis2_loss)
                    self.inpaint_model1.backward(gen1_loss, dis1_loss)
                    iteration = self.inpaint_model2.iteration

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images2), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

                if iteration >= max_iteration:
                    keep_training = False
                    break

        print('\nEnd training....')

    def test(self):
        self.inpaint_model1.eval()
        self.inpaint_model2.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        test_info = []
        for items in test_loader:
            name = self.test_dataset.load_name(index)  #
            images2, images1, masks2, masks1 = self.cuda(*items)
            index += 1
            if model == 1:
                outputs1 = self.inpaint_model1(images1, masks1)
                outputs1_merged = (outputs1 * masks1) + (images1 * (1 - masks1))
                outputs_merged = outputs1_merged
                psnr = self.psnr(self.postprocess(images1), self.postprocess(outputs1_merged))
                mae = (torch.sum(torch.abs(images1 - outputs1_merged)) / torch.sum(images1)).float()

            elif model == 2:

                outputs2 = self.inpaint_model2(images2, images1,  masks2)
                outputs2_merged = (outputs2 * masks2) + (images2 * (1 - masks2))
                outputs_merged = outputs2_merged
                psnr = self.psnr(self.postprocess(images2), self.postprocess(outputs2_merged))
                mae = (torch.sum(torch.abs(images2 - outputs2_merged)) / torch.sum(images2)).float()

            else:
                outputs1 = self.inpaint_model1(images1, masks1).detach()
                outputs1_merged = (outputs1 * masks1) + (images1 * (1 - masks1))
                outputs2 = self.inpaint_model2(images2, outputs1_merged, masks2)
                outputs2_merged = (outputs2 * masks2) + (images2 * (1 - masks2))
                outputs_merged = outputs2_merged
                psnr = self.psnr(self.postprocess(images2), self.postprocess(outputs2_merged))
                mae = (torch.sum(torch.abs(images2 - outputs2_merged)) / torch.sum(images2)).float()

            output = self.postprocess(outputs_merged)[0]  # [0, 1] float B*C*H*W ======> [0, 255] int  B*H*W*C
            print(index, name, '   ', 'psnr= ', psnr.item(), '  mae= ', mae.item())

            # save inpainted result
            imsave(output, os.path.join(self.results_path, name[0:-4]+'_out'+name[-4:]))

        print('\ntest results are saved to: %s' %(self.results_path))
        print('\nEnd test....')
        return test_info

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.test_dataset) == 0:
            return

        self.inpaint_model2.eval()

        model = self.config.MODEL
        items = next(self.sample_iterator)
        images2, images1, masks2, masks1 = self.cuda(*items)

        if model == 1:
            iteration = self.inpaint_model1.iteration
            inputs = (images1 * (1 - masks1)) + masks1
            outputs1 = self.inpaint_model1(images1, masks1)
            outputs = outputs1
            outputs_merged = (outputs1 * masks1) + (images1 * (1 - masks1))

        elif model == 2:
            iteration = self.inpaint_model2.iteration
            inputs = images1 #(torch.nn.functional.interpolate(images_in, scale_factor=2, mode='bicubic') * masks) + (images_gt * (1 - masks))
            outputs2 = self.inpaint_model2(images2, images1,  masks2)
            outputs = outputs2
            outputs_merged = (outputs2 * masks2) + (images2 * (1 - masks2))

        else:
            iteration = self.inpaint_model2.iteration
            inputs = (images2 * (1 - masks2)) + masks2
            outputs1 = self.inpaint_model1(images1, masks1).detach()
            outputs1_merged = (outputs1 * masks1) + (images1 * (1 - masks1))
            outputs2 = self.inpaint_model2(images2, outputs1_merged, masks2)
            outputs = outputs2
            outputs_merged = (outputs2 * masks2) + (images2 * (1 - masks2))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        images2 = stitch_images(
            self.postprocess(images2),  # gt
            self.postprocess(inputs),   # inputs
            self.postprocess(masks2),
            self.postprocess(outputs),   # raw outputs
            self.postprocess(outputs_merged),
            img_per_row = image_per_row
        )


        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(8) + ".jpg")
        create_dir(path)
        print('\nsaving sample ' + name)
        images2.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
