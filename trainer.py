# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import copy
import logging
import sys
import yaml

import numpy as np
import torch.distributed as dist
import torch
# Check if CUDA is available and set the default device
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Ensure torch is aware of the right device
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Check your installation and GPU.")
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from masks.random import MaskCollator as MBMaskCollator
from masks.utils import apply_masks

from utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from utils.tensors import repeat_interleave_batch

from dataloader.EEG_loader import load_eeg_data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#import ALL
from utils.distributed import (
    init_distributed,
    AllReduce
)

from helper import (
    load_checkpoint,
    init_model,
    init_opt)


# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    #root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    

    # -- init torch distributed backend
    
    
    
    

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    target_encoder = copy.deepcopy(encoder)

    # -- make data transforms
    mask_collator = MBMaskCollator()
    
            
            
        
    #path_indicies_to_use = [18,19]#,20,22,23,24,25,26,27,28]
    path_indicies_to_use = [1,2,3,5,6,8,11,12,13]
    path_indicies_to_use = [29,30,32,37,38,39,40,41]
    

    # -- init data-loaders/samplers
    import EEG_loader
    unsupervised_dataset, unsupervised_loader = EEG_loader.load_eeg_data(
    participant_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    batch_size=batch_size,
    #this is what defines the masking for the training step (mask_collator)
    collator=mask_collator,
    pin_mem=pin_mem,
    num_workers=num_workers
)
    ipe = len(unsupervised_loader)
    
    

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    #dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    #move to device
    encoder = encoder.to(device)
    predictor = predictor.to(device)
    target_encoder = target_encoder.to(device)
    encoder.to(device)
    predictor.to(device)
    target_encoder.to(device)
    
    
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'lr': lr
        }
        
    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        
        save_freq = checkpoint_freq
        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()
        last_loss = 0
        for epoch in range(start_epoch, num_epochs):
            logger.info(f'Epoch {epoch + 1}')

            loss_meter = AverageMeter()
            maskA_meter = AverageMeter()
            maskB_meter = AverageMeter()
            time_meter = AverageMeter()

            for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):
                imgs = udata['eeg'].to(device, non_blocking=True)  # Ensure imgs are on the device
                print("imgs shape: ", imgs.shape)
                #print shape of masks_enc
                print("mask encoder shape: ", masks_enc[0].shape)
                #print shape of masks_pred
                print("mask prediction shape: ",masks_pred[0].shape)
                
                print("Imgs shape: ", imgs.shape)
                
                #moving the encoder mask, and the predictor mask to the device
                masks_1 = [m.to(device, non_blocking=True) for m in masks_enc]  # Masks to device
                masks_2 = [m.to(device, non_blocking=True) for m in masks_pred]  # Masks to device

                print("Mask 1 shape: ", masks_1[0].shape)
                print("Mask 2 shape: ", masks_2[0].shape)
                
                # Ensure target encoder output is also moved to device if not done automatically
                h = target_encoder(imgs).to(device)
                #print("h shape after target encoder: ", h.shape)
                #h = F.layer_norm(h, (h.size(-1),)).to(device)  # Layer norm and ensure on device
                print("h shape before apply masks: ", h.shape) #torch.Size([16, 8, 192])
                h = apply_masks(h, masks_2)  # Apply masks, function now handles device matching
                h = repeat_interleave_batch(h, len(h), repeat=len(masks_1)) 
                                
                z = encoder(imgs, masks_1)
                z = predictor(z, masks_1, masks_2)

                loss = F.smooth_l1_loss(z, h)
                print(loss)
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN or Inf found in loss for iteration {itr}")
                    continue

               

                # Backward and optimize
                optimizer.zero_grad()
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                # Momentum update for target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

                # Update meters
                loss_meter.update(loss.item())
                last_loss = loss.item()
                # Assuming etime is measured somewhere

                # Logging
                if itr % log_freq == 0:
                    logger.info(f'[{epoch + 1}, {itr}] loss: {loss_meter.avg:.3f} '
                                f'(mem: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2e} MB)')

                csv_logger.log(epoch + 1, itr, loss_meter.val, maskA_meter.val, maskB_meter.val)

            # Optionally save model
            if epoch % save_freq == 0:
                torch.save(encoder.state_dict(), save_path.format(epoch=epoch))


            # -- Logging
            def log_stats(loss):
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   
                                  
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))

                    
            
            

            # Logging each iteration
            log_stats(last_loss)
            

            

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)


if __name__ == "__main__":
    main()
