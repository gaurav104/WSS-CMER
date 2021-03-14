import numpy as np

from tqdm import tqdm
import shutil

import torch
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision

from graphs.models.deform_unet import *
from datasets.cellseg import CellSegDataLoader
from graphs.losses.bce import BinaryCrossEntropy
from graphs.losses.dice import *

from torch.optim import lr_scheduler

from torch.utils.tensorboard import SummaryWriter
from utils.metrics import AverageMeter, Dice, IoU
from utils.misc import print_cuda_statistics

from agents.base import BaseAgent

cudnn.benchmark = True


class DUnetAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # define models
        self.model = DUNetV1V2(1, 1)

        # define data_loader
        self.data_loader = CellSegDataLoader(self.config)

        # define loss
        self.loss_bce = BinaryCrossEntropy()
        self.loss_dice = SoftDiceLoss()
        self.loss_inv_dice = InvSoftDiceLoss()

        # define optimizers for both generator and discriminator
        self.optimizer = torch.optim.Adam(self.model.parameters(),
        	lr=self.config.learning_rate)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_dice_coeff = 0

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
        self.loss_bce = self.loss_bce.to(self.device)
        self.loss_dice = self.loss_dice.to(self.device)
        self.loss_inv_dice = self.loss_inv_dice.to(self.device)

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
            'best_metric_value': self.best_valid_dice_coeff,
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
            self.best_valid_dice_coeff = checkpoint['best_metric_value']
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

            valid_dice_coefficient = self.validate()
            # self.scheduler.step(valid_loss)

            is_best = valid_dice_coefficient > self.best_valid_dice_coeff
            if is_best:
                self.best_valid_dice_coeff = valid_dice_coefficient
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
        bce = AverageMeter()
        dice = AverageMeter()
        inv_dice = AverageMeter()

        # Initialize average meters of metrics
        # accuracy = AverageMeter()
        dice_coeff_hard = AverageMeter()
        iou = AverageMeter()


        #epoch loss
        # metrics = IOUMetric(self.config.num_classes)

        for x, y in tqdm_batch:
            if self.cuda:
                x, y = x.pin_memory().cuda(async=self.config.async_loading), y.cuda(async=self.config.async_loading)
            x, y = Variable(x), Variable((y>0.5).float())
            # model
            pred = self.model(x)
            # loss
            bce_loss = self.loss_bce(torch.sigmoid(pred), (y>0.5).float())
            dice_loss = self.loss_dice(torch.sigmoid(pred), (y>0.5).float())
            inv_dice_loss = self.loss_inv_dice(torch.sigmoid(pred), (y>0.5).float())
            cur_loss =  bce_loss + dice_loss + inv_dice_loss
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')

            # optimizer
            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()
            
            bce.update(bce_loss.item())
            dice.update(dice_loss.item())
            inv_dice.update(inv_dice_loss.item())


            
            iter_dice_coeff_dice =  Dice(pred, y)
            iter_iou = IoU(pred, y)

            dice_coeff_hard.update(iter_dice_coeff_dice.item())
            iou.update(iter_iou.item())

            # pred_max = torch.sigmoid(pred) > 0.5
            # metrics.add_batch(pred_max.data.cpu().numpy(), y.data.cpu().numpy())

            self.current_iteration += 1

        grid_y = torchvision.utils.make_grid(y)
        grid_pred = torchvision.utils.make_grid(torch.sigmoid(pred))
        self.summary_writer.add_image("epoch_train/gt", grid_y, self.current_iteration)
        self.summary_writer.add_image("epoch_train/pred", grid_pred , self.current_iteration)
        self.summary_writer.add_scalar("epoch_train/bce_loss", bce.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_train/dice_loss", dice.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_train/inv_dice_loss", inv_dice.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_train/dice_coeff_hard", dice_coeff_hard.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_train/iou", dice_coeff_hard.val, self.current_iteration)

        tqdm_batch.close()

        #get metrics from another class

        print("Train Results at epoch-" + str(self.current_epoch))
        print("BCE loss: " + str(bce.val) )
        print("Dice Loss: "+ str(dice.val) )
        print("Inverted Dice Loss: " + str(inv_dice.val))
        print("Dice Coefficient Hard: " + str(dice_coeff_hard.val) )
        print("IoU: " + str(iou.val)+ "\n")
       
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

        bce = AverageMeter()
        dice = AverageMeter()
        inv_dice = AverageMeter()
        # metrics = IOUMetric(self.config.num_classes)
        dice_coeff_hard = AverageMeter()
        iou = AverageMeter()

        for x, y in tqdm_batch:
            if self.cuda:
                x, y = x.pin_memory().cuda(async=self.config.async_loading), y.cuda(async=self.config.async_loading)
            x, y = Variable(x), Variable((y>0.5).float())
            # model
            pred = self.model(x)
            # loss
            bce_loss = self.loss_bce(torch.sigmoid(pred), y)
            dice_loss = self.loss_dice(torch.sigmoid(pred), y)
            inv_dice_loss = self.loss_inv_dice(torch.sigmoid(pred), y)
            cur_loss =  bce_loss + dice_loss + inv_dice_loss

            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during Validation.')

            # pred_max = torch.sigmoid(pred)
            # metrics.add_batch(pred_max.data.cpu().numpy(), y.data.cpu().numpy())

            iter_dice_coeff_dice =  Dice(pred, y)
            iter_iou = IoU(pred, y)

            bce.update(bce_loss.item())
            dice.update(dice_loss.item())
            inv_dice.update(inv_dice_loss.item())

            dice_coeff_hard.update(iter_dice_coeff_dice.item())
            iou.update(iter_iou.item())

        # epoch_acc, _, epoch_iou_class, epoch_mean_iou, _ = metrics.evaluate()
        grid_y = torchvision.utils.make_grid(y)
        grid_pred = torchvision.utils.make_grid(torch.sigmoid(pred))
        self.summary_writer.add_image("epoch_validation/gt", grid_y, self.current_iteration)
        self.summary_writer.add_image("epoch_validation/pred", grid_pred , self.current_iteration)
        self.summary_writer.add_scalar("epoch_validation/bce_loss", bce.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_validation/dice_loss", dice.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_validation/inv_dice_loss", inv_dice.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_validation/dice_coeff_hard", dice_coeff_hard.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_validation/iou", iou.val, self.current_iteration)
        # self.summary_writer.add_scalar("epoch_validation/mean_iou", epoch_mean_iou, self.current_iteration)

       
        print("Validation Results at epoch-" + str(self.current_epoch))
        print("BCE loss: " + str(bce.val))
        print("Dice Loss: "+ str(dice.val))
        print("Inverted Dice Loss: " + str(inv_dice.val))
        print("Dice Coefficient Hard: " + str(dice_coeff_hard.val))
        print("IoU: " + str(iou.val)+ "\n")

        tqdm_batch.close()

        return dice_coeff_hard.val

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
