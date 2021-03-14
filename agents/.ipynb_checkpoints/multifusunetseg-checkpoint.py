from graphs.models.network import *
import numpy as np

from tqdm import tqdm
import shutil
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision

from datasets.cellseg import CellSegDataLoader
# from graphs.losses.bce import BinaryCrossEntropy
from kornia.losses import DiceLoss
from torch.optim import lr_scheduler

from torch.utils.tensorboard import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList, Dice, IoU
from utils.misc import print_cuda_statistics
import statistics

from agents.base import BaseAgent

cudnn.benchmark = True


class MultiFusUnetAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # define models
        self.num_cls = self.config.num_classes
        self.in_ch = self.config.input_channels
        self.model = AttU_Net_Fus(self.in_ch, self.num_cls, self.config.attention)
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.num_params = sum([np.prod(p.size()) for p in model_parameters])

        # define data_loader
        self.data_loader = CellSegDataLoader(self.config)

        # define loss
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_dice = DiceLoss()
        # self.loss_inv_dice = InvSoftDiceLoss()

        # define optimizers for both generator and discriminator
        self.optimizer = torch.optim.Adam(self.model.parameters(),
            lr=self.config.learning_rate)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0 # lowest value

        # Check is cuda is available or not
        self.is_cuda = torch.cuda.is_available()
        # Construct the flag and make sure that cuda is available
        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            torch.cuda.manual_seed_all(self.config.seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.logger.info("Operation will be on *****GPU-CUDA***** ")
            print_cuda_statistics()

        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.config.seed)
            self.logger.info("Operation will be on *****CPU***** ")


        self.model = self.model.to(self.device)
        self.loss_ce = self.loss_ce.to(self.device)
        self.loss_dice = self.loss_dice.to(self.device)
        # self.loss_inv_dice = self.loss_inv_dice.to(self.device)

        # Model Loading from the latest checkpoint if not found start from scratch.

        self.load_checkpoint(self.config.checkpoint_file)
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='UNet')

    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=0):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_loss_value': self.best_metric,
            'num_of_trainable_params': self.num_params
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + filename)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + filename,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def load_checkpoint(self, filename):

        filename = self.config.checkpoint_dir + filename
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            self.best_metric = checkpoint['best_metric_value']
            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")


    def run(self):
        """
        This function will the operator
        :return:
        """
        assert self.config.mode in ['train', 'test', 'random']
        try:
            if self.config.mode == 'test':
                self.test()
            else:
                self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training function, with per-epoch model saving
        """

        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            # self.scheduler.step(epoch)
            self.train_one_epoch()

            valid_mIoU, valid_mDice  = self.validate()
            # self.scheduler.step(valid_loss)

            is_best = valid_mDice > self.best_metric

            if(is_best):
                self.best_metric = valid_mDice
            self.save_checkpoint(is_best=is_best)

    def train_one_epoch(self):
        """
        One epoch training function
        """
        # Initialize tqdm
        tqdm_batch = tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
                          desc="Epoch-{}-".format(self.current_epoch))

        # Set the model to be in training mode (for batchnorm)
        self.model.train()
        

        # Initialize average meters of losses
        ce = AverageMeter()
        dice = AverageMeter()
        # inv_dice = AverageMeter()

        # Initialize average meters of metrics
        # accuracy = AverageMeter()
        dice_coeff_hard = AverageMeterList(self.num_cls)
        iou = AverageMeterList(self.num_cls)


        #epoch loss
        # metrics = IOUMetric(self.config.num_classes)

        for x, y in tqdm_batch:
            if self.cuda:
                x, y = x.pin_memory().cuda(async=self.config.async_loading), y.cuda(async=self.config.async_loading)
            x, y = Variable(x), Variable(y)
            # model
            pred = self.model(x)
            
            # loss
            ce_loss = self.loss_ce(pred, y)
            dice_loss = self.loss_dice(pred, y)
            # inv_dice_loss = self.loss_inv_dice(torch.sigmoid(pred), (y>0.5).float())
            cur_loss =  ce_loss + dice_loss
#             if np.isnan(float(cur_loss.item())):
#                 raise ValueError('Loss is nan during training...')

            # optimizer
            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()
            
            ce.update(ce_loss.item())
            dice.update(dice_loss.item())
            # inv_dice.update(inv_dice_loss.item())


            
            iter_dice_coeff_dice =  Dice(pred, y, self.num_cls)
            iter_iou = IoU(pred,y, self.num_cls)

            dice_coeff_hard.update(iter_dice_coeff_dice.cpu().numpy())
            iou.update(iter_iou.cpu().numpy())

            # pred_max = torch.sigmoid(pred) > 0.5
            # metrics.add_batch(pred_max.data.cpu().numpy(), y.data.cpu().numpy())

            self.current_iteration += 1

        grid_y = torchvision.utils.make_grid(y.unsqueeze(1))
        
        grid_pred_contour = torchvision.utils.make_grid(F.softmax(pred, 1)[:,0:1,:,:])
        grid_pred_cell = torchvision.utils.make_grid(F.softmax(pred, 1)[:,1:,:,:])
        self.summary_writer.add_image("epoch_train/gt", grid_y, self.current_iteration)
        self.summary_writer.add_image("epoch_train/pred_contour", grid_pred_contour, self.current_iteration)
        self.summary_writer.add_image("epoch_train/pred_cell", grid_pred_cell, self.current_iteration)
        self.summary_writer.add_scalar("epoch_train/ce_loss", ce.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_train/dice_loss", dice.val, self.current_iteration)
#         self.summary_writer.add_scalar("epoch_train/inv_dice_loss", inv_dice.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_train/dice_coeff_hard_1", dice_coeff_hard.val[0], self.current_iteration)
        self.summary_writer.add_scalar("epoch_train/dice_coeff_hard_2", dice_coeff_hard.val[1], self.current_iteration)

        self.summary_writer.add_scalar("epoch_train/iou_1", dice_coeff_hard.val[0], self.current_iteration)
        self.summary_writer.add_scalar("epoch_train/iou_2", dice_coeff_hard.val[1], self.current_iteration)

        tqdm_batch.close()

        #get metrics from another class

        print("Train Results at Epoch-" + str(self.current_epoch))
        print("CE loss: " + str(ce.val) )
        print("Dice Loss: "+ str(dice.val) )
        # print("Inverted Dice Loss: " + str(inv_dice.val))
        print("Dice Coefficient Hard 1: " + str(dice_coeff_hard.val[0]))
        print("Dice Coefficient Hard 2: " + str(dice_coeff_hard.val[1]))
        print("IoU 1: " + str(iou.val[0]))
        print("IoU 2: " + str(iou.val[1])+ "\n")
       
        # self.summary_writer.add_scalar("epoch_validation/mean_iou", epoch_mean_iou, self.current_iteration)


    def validate(self):
        """
        One epoch validation
        :return:
        """
        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                          desc="Valiation at -{}-".format(self.current_epoch))

        # set the model in training mode
        self.model.eval()

        ce = AverageMeter()
        dice = AverageMeter()
        # inv_dice = AverageMeter()
        # metrics = IOUMetric(self.config.num_classes)
        dice_coeff_hard = AverageMeterList(self.num_cls)
        iou = AverageMeterList(self.num_cls)


        for x, y in tqdm_batch:
            if self.cuda:
                x, y = x.pin_memory().cuda(async=self.config.async_loading), y.cuda(async=self.config.async_loading)
            x, y = Variable(x), Variable(y)
            # model
            pred = self.model(x)
            # loss
            ce_loss = self.loss_ce(pred, y)
            dice_loss = self.loss_dice(pred, y)
            # inv_dice_loss = self.loss_inv_dice(torch.sigmoid(pred), y)

            cur_loss =  dice_loss + ce_loss

            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during Validation.')

            # pred_max = torch.sigmoid(pred)
            # metrics.add_batch(pred_max.data.cpu().numpy(), y.data.cpu().numpy())

            iter_dice_coeff_dice =  Dice(pred, y)
            iter_iou = IoU(pred, y)

            ce.update(ce_loss.item())
            dice.update(dice_loss.item())
            # inv_dice.update(inv_dice_loss.item())

            dice_coeff_hard.update(iter_dice_coeff_dice.cpu().numpy())
            iou.update(iter_iou.cpu().numpy())
            
        
        grid_y = torchvision.utils.make_grid(y.unsqueeze(1))
        # epoch_acc, _, epoch_iou_class, epoch_mean_iou, _ = metrics.evaluate()
        grid_pred_contour = torchvision.utils.make_grid(F.softmax(pred, 1)[:,0:1,:,:])
        grid_pred_cell = torchvision.utils.make_grid(F.softmax(pred, 1)[:,1:,:,:])
        self.summary_writer.add_image("epoch_valid/gt", grid_y, self.current_iteration)
        self.summary_writer.add_image("epoch_valid/pred_contour", grid_pred_contour, self.current_iteration)
        self.summary_writer.add_image("epoch_valid/pred_cell", grid_pred_cell, self.current_iteration)
        self.summary_writer.add_scalar("epoch_valid/ce_loss", ce.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_valid/dice_loss", dice.val, self.current_iteration)
        # self.summary_writer.add_scalar("epoch_train/inv_dice_loss", inv_dice.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_valid/dice_coeff_hard_1", dice_coeff_hard.val[0], self.current_iteration)
        self.summary_writer.add_scalar("epoch_valid/dice_coeff_hard_2", dice_coeff_hard.val[1], self.current_iteration)

        self.summary_writer.add_scalar("epoch_valid/iou_1", iou.val[0], self.current_iteration)
        self.summary_writer.add_scalar("epoch_valid/iou_2", iou.val[1], self.current_iteration)


        #get metrics from another class

        print("validation Results at Epoch-" + str(self.current_epoch))
        print("CE loss: " + str(ce.val) )
        print("Dice Loss: "+ str(dice.val) )
        # print("Inverted Dice Loss: " + str(inv_dice.val))
        print("Dice Coefficient Hard 1: " + str(dice_coeff_hard.val[0]))
        print("Dice Coefficient Hard 2: " + str(dice_coeff_hard.val[1]))
        print("IoU 1: " + str(iou.val[0]))
        print("IoU 2: " + str(iou.val[1])+ "\n")

        tqdm_batch.close()

        return statistics.mean(iou.val), statistics.mean(dice_coeff_hard.val)

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        print("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        # self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
        # self.data_loader.finalize()
