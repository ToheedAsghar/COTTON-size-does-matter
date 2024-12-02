# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torchvision

import torchvision.transforms as transforms

import os
import time
import argparse

from dataLoader import TryonDataset
# from dataLoader_processed import TryonDataset
from DiffAugment_pytorch import DiffAugment

# from util.utils import *
from datetime import datetime

from Focal_Loss import focal_loss
from mean_iou_evaluate import read_masks, mean_iou_score

from tensorboardX import SummaryWriter
from tqdm import tqdm
import glob
from unet import UNet, UNet_v2
import util.utils as utils
import numpy as np
import imageio
from pytorch_fid import fid_score
import yaml

 # New parsing:
    # LIP           -> New
    # 0             -> 0  Background
    # 1, 2          -> 1  Hair
    # 4, 13         -> 2  Face
    # 10            -> 3  Neck
    # 5             -> 4  Upper-clothes
    # 7             -> 5  Coat
    # 6             -> 6  dress
    # 9, 12         -> 7  Lower-clothes
    # 14            -> 8  Left-arm
    # 15            -> 9  Right-arm
    # 16            -> 10 Left-leg
    # 17            -> 11 Right-leg
    # 18            -> 12 Left-shoe 
    # 19            -> 13 Right-shoe
    # 3, 8, 11      -> 14 Accessories


def train(config):
    # Local weight and sample directories
    weight_dir = os.path.join('result', config['TRAINING_CONFIG']['TRAIN_DIR'], 'weights')
    sample_dir = os.path.join('result', config['TRAINING_CONFIG']['TRAIN_DIR'], 'samples')
    run_dir = os.path.join('result', config['TRAINING_CONFIG']['TRAIN_DIR'], 'runs')

    # Google Drive directory for saving weights
    drive_weight_dir = '/content/drive/My Drive/SIZE_DOES_MATTER/checkpoints'
    os.makedirs(drive_weight_dir, exist_ok=True)

    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    dataset = TryonDataset(config)
    dataloader = DataLoader(dataset, batch_size=config['TRAINING_CONFIG']['BATCH_SIZE'],
                            shuffle=True, num_workers=config['TRAINING_CONFIG']['NUM_WORKER'])
    print('Size of the dataset: %d, dataloader: %d' % (len(dataset), len(dataloader)))

    model = COTTON(config).cuda().train()
    discriminator = FashionOn_MultiD(shape=config['TRAINING_CONFIG']['RESOLUTION']).cuda().train()

    # Loss functions and optimizers
    criterion_VGG = FashionOn_VGGLoss()
    criterion_L1 = nn.L1Loss()
    criterionBCE = nn.BCELoss()
    focal_weight = [2, 3, 3, 3, 10, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2]
    criterion_focal = focal_loss(alpha=focal_weight, gamma=2)

    optimizer_G = torch.optim.Adam(model.parameters(), lr=config['TRAINING_CONFIG']['LR'],
                                   betas=(config['TRAINING_CONFIG']['BETA1'], config['TRAINING_CONFIG']['BETA2']))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config['TRAINING_CONFIG']['LR'],
                                   betas=(config['TRAINING_CONFIG']['BETA1'], config['TRAINING_CONFIG']['BETA2']))

    scheduler = None
    if config['TRAINING_CONFIG']['SCHEDULER'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G,
                                                               T_max=config['TRAINING_CONFIG']['T_MAX'],
                                                               eta_min=config['TRAINING_CONFIG']['ETA_MIN'])

    weight_path = os.path.join(weight_dir, '{}.pkl'.format(config['TRAINING_CONFIG']['TRAIN_DIR']))
    best_model_path = os.path.join(weight_dir, 'best_model.pkl')
    best_drive_path = os.path.join(drive_weight_dir, 'best_model.pkl')

    # Check for checkpoints on Google Drive first
    start_epoch = 0
    best_score = float('inf')
    checkpoint_found = False
    if os.path.isfile(best_drive_path):
        print("=> Found checkpoint on Google Drive: '{}'".format(best_drive_path))
        checkpoint = torch.load(best_drive_path, map_location='cpu')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        discriminator.load_state_dict(checkpoint['state_dict_D'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        if scheduler and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> Loaded checkpoint from Google Drive (epoch {})".format(checkpoint['epoch']))
        best_score = checkpoint.get('best_score', best_score)
        checkpoint_found = True
    elif os.path.isfile(weight_path):
        print("=> Found local checkpoint: '{}'".format(weight_path))
        checkpoint = torch.load(weight_path, map_location='cpu')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        discriminator.load_state_dict(checkpoint['state_dict_D'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        if scheduler and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> Loaded local checkpoint (epoch {})".format(checkpoint['epoch']))
        best_score = checkpoint.get('best_score', best_score)
        checkpoint_found = True

    if not checkpoint_found:
        print("=> No checkpoint found. Starting training from scratch.")

    board = SummaryWriter(run_dir)

    for epoch in range(start_epoch, config['TRAINING_CONFIG']['EPOCH']):
        print("epoch: " + str(epoch + 1))
        epoch_loss = 0

        for step, batch in enumerate(tqdm(dataloader)):
            total_step = epoch * len(dataloader) + step

            # Training logic
            human_masked = batch['human_masked'].cuda()
            human_pose = batch['human_pose'].cuda()
            human_parse_masked = batch['human_parse_masked'].cuda()
            human_img = batch['human_img'].cuda()
            human_parse_label = batch['human_parse_label'].cuda()
            c_aux = batch['c_aux_warped'].cuda()
            c_torso = batch['c_torso_warped'].cuda()

            c_img = torch.cat([c_torso, c_aux], dim=1)
            parsing_pred, parsing_pred_hard, tryon_img_fake = model(c_img, human_parse_masked, human_masked, human_pose)

            # Calculate losses
            loss_focal = criterion_focal(parsing_pred, human_parse_label)
            loss_content_gen, loss_style_gen = criterion_VGG(tryon_img_fake, human_img)
            loss_vgg = config['TRAINING_CONFIG']['LAMBDA_VGG_STYLE'] * loss_style_gen + config['TRAINING_CONFIG']['LAMBDA_VGG_CONTENT'] * loss_content_gen
            loss_global_L1 = criterion_L1(tryon_img_fake, human_img)

            loss_G = config['TRAINING_CONFIG']['LAMBDA_L1'] * loss_global_L1 + \
                     config['TRAINING_CONFIG']['LAMBDA_VGG'] * loss_vgg + \
                     config['TRAINING_CONFIG']['LAMBDA_FOCAL'] * loss_focal

            # Backpropagation and optimization
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            epoch_loss += loss_G.item()  # Accumulate generator loss

        # Save checkpoint
        save_info = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'state_dict_D': discriminator.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
            'best_score': best_score,
        }
        if scheduler:
            scheduler.step()
            save_info['scheduler'] = scheduler.state_dict()

        torch.save(save_info, weight_path)  # Save latest checkpoint locally

        # Save best model
        avg_epoch_loss = epoch_loss / len(dataloader)
        if avg_epoch_loss < best_score:
            best_score = avg_epoch_loss
            save_info['best_score'] = best_score
            torch.save(save_info, best_model_path)  # Save best model locally
            torch.save(save_info, best_drive_path)  # Save best model to Google Drive
            print(f"Best model updated with loss {best_score:.4f} and saved to Google Drive.")

def test(opt):
    record_file = os.path.join('result', config['TRAINING_CONFIG']['TRAIN_DIR'], 'FID_score_{}.txt'.format(config['MODE']))
    f = open(record_file, 'a')
    
    weight_dir = os.path.join('result', config['TRAINING_CONFIG']['TRAIN_DIR'], 'weights')
    weight_path = os.path.join(weight_dir, '{}.pkl'.format(config['TRAINING_CONFIG']['TRAIN_DIR']))

    val_folder = os.path.join('result', config['TRAINING_CONFIG']['TRAIN_DIR'], config['MODE'])
    GT_folder = os.path.join(val_folder, 'GT')
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(GT_folder, exist_ok=True)


    dataset = TryonDataset(config)
    dataloader = DataLoader(dataset, batch_size=config['VAL_CONFIG']['BATCH_SIZE'], \
                            shuffle=True, num_workers=config['TRAINING_CONFIG']['NUM_WORKER'])
    
    print('Size of the dataset: %d, dataloader: %d' % (len(dataset), len(dataloader)))
    model = COTTON(config).cuda().train()
    
    best_score = np.inf
    best_epoch = 0
    # pg_unet_wo_warp 30->
    for e in range(config['VAL_CONFIG']['START_EPOCH'], config['VAL_CONFIG']['END_EPOCH'], config['VAL_CONFIG']['EPOCH_STEP']):
        weight_name = '{}_{}.pkl'.format(weight_path.split('.')[0], e)
        if not os.path.isfile(weight_name):
            print("weight not found | {}".format(weight_name))
            break
        checkpoint = torch.load(weight_name, map_location='cpu')
        
        fid_pred_folder = os.path.join(val_folder, '{}'.format(e)) if config['TUCK'] else os.path.join(val_folder, '{}_untucked'.format(e))
        os.makedirs(fid_pred_folder, exist_ok=True)

        epoch_num = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(weight_path, checkpoint['epoch']))
        model.cuda().eval()
        start_time = time.time()

        for step, batch in enumerate(tqdm(dataloader)):
            
            # Name
            human_name = batch["human_name"]
            c_name = batch["c_name"]

            # Input
            human_masked = batch['human_masked'].cuda()
            human_pose = batch['human_pose'].cuda()
            human_parse_masked = batch['human_parse_masked'].cuda()
            c_aux = batch['c_aux_warped'].cuda()
            c_torso = batch['c_torso_warped'].cuda()
            c_rgb = batch['c_rgb'].cuda()


            # Supervision
            human_img = batch['human_img'].cuda()
            human_parse_label = batch['human_parse_label'].cuda()
            human_parse_masked_label = batch['human_parse_masked_label'].cuda()

            # print("c_torso.size() = {} [{}, {}]".format(c_torso.size(), torch.min(c_torso), torch.max(c_torso)))
            # print("c_aux.size() = {} [{}, {}]".format(c_aux.size(), torch.min(c_aux), torch.max(c_aux)))
            # print("human_parse_masked.size() = {} [{}, {}]".format(human_parse_masked.size(), torch.min(human_parse_masked), torch.max(human_parse_masked)))
            # print("human_masked.size() = {} [{}, {}]".format(human_masked.size(), torch.min(human_masked), torch.max(human_masked)))
            # print("human_pose.size() = {} [{}, {}]".format(human_pose.size(), torch.min(human_pose), torch.max(human_pose)))
            # exit()

            with torch.no_grad():
                c_img = torch.cat([c_torso, c_aux], dim=1)
                parsing_pred, parsing_pred_hard, tryon_img_fakes = model(c_img, human_parse_masked, human_masked, human_pose)

            for idx, tryon_img_fake in enumerate(tryon_img_fakes):
                utils.imsave_trainProcess([utils.remap(tryon_img_fake)], os.path.join(fid_pred_folder, c_name[idx]))
                utils.imsave_trainProcess([utils.remap(human_img)], os.path.join(GT_folder, c_name[idx]))
                # utils.imsave_trainProcess([utils.remap(tryon_img_fake)], os.path.join(fid_pred_folder, human_name[idx].replace('.jpg','') + '_' + c_name[idx]))
                # utils.imsave_trainProcess([utils.remap(human_img)], os.path.join(GT_folder, human_name[idx].replace('.jpg','') + '_' + c_name[idx]))

        print("cost {}/images secs [with average of {} images]".format((time.time()-start_time)/len(dataset), len(dataset)))

        fid = fid_score.calculate_fid_given_paths(paths=[GT_folder, fid_pred_folder],batch_size=50,device=torch.device(0),dims=2048,num_workers=0)

        if fid < best_score:
            best_score, best_epoch = fid, e
        print(e, fid)
        f.write('epoch:{} fid:{}\n'.format(e, fid))
        
    print('Best epoch:{}, Best fid:{}'.format(best_epoch, best_score))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default= "train")
    parser.add_argument("--config",
                        type=str,
                        default='configs/config_top.yaml')
    parser.add_argument('--untuck', action='store_true')
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--mask_arm', action='store_true')
    opt = parser.parse_args()

    config = yaml.load(open(opt.config, 'r'), Loader=yaml.FullLoader)
    config['MODE'] = opt.mode
    config['TUCK'] = not opt.untuck
    config['VAL_CONFIG']['SCALE'] = opt.scale
    config['VAL_CONFIG']['MASK_ARM'] = opt.mask_arm

    from model_end2end import COTTON, FashionOn_MultiD, FashionOn_VGGLoss

    if opt.mode == 'train':
        train(config)
    else:
        test(config)
