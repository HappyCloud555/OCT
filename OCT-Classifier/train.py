import argparse
import logging
import math
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from mmengine import Config
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import dataset_builder
from model import model_builder
from optimizer import optim_builder
from scheduler import scheduler_builder
from trainer import trainer_builder
from utils.misc import AverageMeter, AverageMeterManeger, accuracy
from utils.ckpt_utils import save_ckpt_dict
from utils.config_utils import overwrite_config
from utils.log_utils import get_default_logger

torch.autograd.set_detect_anomaly(True)

# global variables
global logger
SCALER = None
past_epoch = 1
past_batch = 0
test_accs = []
best_test_acc = 0


def set_seed(args):
    """ set seed for the whole program for removing randomness
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_test_model(ema_model, model, use_ema):
    """ use ema model or test model
    """
    if use_ema:
        test_model = ema_model.ema
        test_prefix = "ema"
    else:
        test_model = model
        test_prefix = ""
    return test_model, test_prefix


def main():
    args = get_args()
    # prepare config and make output dir
    cfg = Config.fromfile(args.cfg)
    cfg = overwrite_config(cfg, args.other_args)
    cfg.resume = args.resume
    cfg.resume_pth = args.resume_pth
    cfg.data['eval_step'] = cfg.train.eval_step

    # set amp scaler, usually no use
    # global SCALER
    # if args.fp16:
    #     SCALER = torch.cuda.amp.GradScaler()
    # else:
    #     SCALER = None

    # set summary writer on rank 0
    os.makedirs(args.out, exist_ok=True)
    args.writer = SummaryWriter(args.out)
    cfg.dump(os.path.join(args.out, os.path.basename(args.cfg)))

    # set up logger
    global logger
    logger = get_default_logger(
        args=args,
        logger_name='root',
        default_level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s")

    # set random seed
    if args.seed is not None:
        cfg.seed = args.seed
    elif cfg.get("seed", None) is not None:
        args.seed = cfg.seed

    # set folds for stl10 dataset if used
    if "folds" in cfg.data.keys():
        cfg.data.folds = cfg.seed

    args.amp = False
    if cfg.get("amp", False) and cfg.amp.use:
        args.amp = True
        args.opt_level = cfg.amp.opt_level

    args.total_steps = cfg.train.total_steps
    args.eval_steps = cfg.train.eval_step

    # init dist params
    if torch.cuda.is_available():
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        device = torch.device('cpu', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()

    # set device
    args.device = device

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    # make dataset
    labeled_dataset, unlabeled_dataset, test_dataset, val_dataset = dataset_builder.build(cfg.data)
    logger.info(labeled_dataset)
    logger.info(unlabeled_dataset)
    logger.info(test_dataset)
    logger.info(val_dataset)

    # make dataset loader
    train_sampler = RandomSampler

    labeled_trainloader, unlabeled_trainloader, test_loader, val_loader = get_dataloader(
        cfg, train_sampler, labeled_dataset, unlabeled_dataset, test_dataset, val_dataset)

    model = model_builder.build(cfg.model, logger)
    model.to(args.device)

    # make optimizer,scheduler
    optimizer = optim_builder.build(cfg.optimizer, model)
    scheduler = set_scheduler(cfg, args, optimizer)

    # set ema
    args.use_ema = False
    ema_model = None
    if cfg.get("ema", False) and cfg.ema.use:
        args.use_ema = True
        from model.ema import ModelEMA
        ema_model = ModelEMA(args.device, model, cfg.ema.decay)

    # initialize from resume for fixed info and task_specific_info
    task_specific_info = dict()
    # build model trainer
    cfg.train.trainer['amp'] = args.amp
    model_trainer = trainer_builder.build(cfg.train.trainer)(device=device, all_cfg=cfg)
    log_info(cfg, args)
    model.zero_grad()
    # train loop
    acc = train(args, cfg, labeled_trainloader, unlabeled_trainloader, test_loader, val_loader,
                model, optimizer, ema_model, scheduler, model_trainer,
                task_specific_info)
    logger.info("=====================Training is over!=====================")


# resume from checkpoint
def resume(args,
           model,
           optimizer,
           scheduler,
           task_specific_info,
           ema_model=None, pth="checkpoint.pth.tar", is_temp=False):
    global past_epoch
    global past_batch
    logger.info("==> Resuming from checkpoint..")
    args.resume = os.path.join(args.out, pth)
    args.out = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    # Drawing Settings
    if not is_temp:
        past_batch = checkpoint['past_batch']
        past_epoch = checkpoint['past_epoch']
    model.load_state_dict(checkpoint['state_dict'])
    if args.use_ema:
        ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    for key in checkpoint.keys():
        if key not in [
            'post_batch', 'post_epoch', 'state_dict', 'ema_state_dict', 'best_acc',
            'optimizer', 'scheduler'
        ]:
            task_specific_info[key] = checkpoint[key]
            try:
                task_specific_info[key] = task_specific_info[key].to(
                    args.device)
            except:
                pass


# set scheduler
def set_scheduler(cfg, args, optimizer):
    args.epochs = math.ceil(cfg.train.total_steps / cfg.train.eval_step)
    args.eval_step = cfg.train.eval_step
    args.total_steps = cfg.train.total_steps
    scheduler = scheduler_builder.build(cfg.scheduler)(optimizer=optimizer)
    return scheduler


# log info before training
def log_info(cfg, args):
    logger.info(" ***** Running training *****")
    logger.info(f"  Task = {cfg.data.type}")
    if "num_labeled" in cfg.data.keys():
        logging_num_labeled = cfg.data.num_labeled
    elif "percent" in cfg.data.keys():
        logging_num_labeled = "{}%".format(cfg.data.percent)
    else:
        logging_num_labeled = "seed {}".format(cfg.seed)

    logger.info(f"  Num Label = {logging_num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size of labeled = {cfg.data.batch_size}")
    logger.info(
        f"  Batch size of all = {cfg.data.batch_size + math.floor(cfg.data.batch_size * cfg.data.mu)}")
    logger.info(f"  Total optimization steps = {args.total_steps}")
    logger.info(cfg)


# get args
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch+CCSSL ON OCT images Training')
    parser.add_argument('--cfg', default='./config.py', type=str, help='a config')
    parser.add_argument('--out', default='./result/', help='directory to output the result')
    parser.add_argument('--resume', default=False, type=bool,
                        help='resume from checkpoint')
    parser.add_argument('--resume_pth', default='checkpoint.pth.tar', type=str,
                        help='path to resume')
    parser.add_argument('--gpu-id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', default=None, type=int, help="random seed")
    parser.add_argument('--other-args', default='', type=str, help='other args to overwrite the config, keys are split \
                            by space and args split by |, such as \'seed 1|train trainer T 1|\' ')
    args = parser.parse_args()
    return args


# labeled_trainloader,unlabeled_trainloader,test_loader
def get_dataloader(cfg, train_sampler, labeled_dataset, unlabeled_dataset,
                   test_dataset, val_dataset):
    # prepare labeled_trainloader
    labeled_trainloader = DataLoader(labeled_dataset,
                                     sampler=train_sampler(labeled_dataset),
                                     batch_size=cfg.data.batch_size,
                                     num_workers=cfg.data.num_workers,
                                     drop_last=True)
    # prepare unlabeled_trainloader
    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=math.floor(cfg.data.batch_size * cfg.data.mu),
        num_workers=cfg.data.num_workers,
        drop_last=True)
    # prepare test_loader
    test_loader = DataLoader(test_dataset,
                             sampler=SequentialSampler(test_dataset),
                             batch_size=cfg.data.batch_size,
                             num_workers=cfg.data.num_workers)
    # prepare val_loader
    val_loader = DataLoader(val_dataset,
                            sampler=SequentialSampler(val_dataset),
                            batch_size=cfg.data.batch_size,
                            num_workers=cfg.data.num_workers)
    return labeled_trainloader, unlabeled_trainloader, test_loader, val_loader


# save model considered by epoch
def save_model_best(ema_model, model, args, loader, epoch, batch_idx,
                           optimizer, scheduler, logger, test_accs):
    temp_model, temp_prefix = get_test_model(ema_model, model, args.use_ema)
    temp_loss, temp_acc = test(args, loader, temp_model, epoch)
    args.writer.add_scalars(main_tag="test",
                            tag_scalar_dict={'Accuracy': temp_acc,
                                             'loss': temp_loss,
                                             }, global_step=epoch)
    global best_test_acc
    is_best = temp_acc > best_test_acc
    if is_best:
        best_test_acc = temp_acc
        # save best model
        save_ckpt_dict(args, model, ema_model, epoch, batch_idx,
                       optimizer, scheduler, temp_acc, logger)
    test_accs.append(temp_acc)
    logger.info('Best test acc of all epochs: {:.2f}'.format(best_test_acc))


# train_loop
def train(args, cfg, labeled_trainloader, unlabeled_trainloader, test_loader, val_loader,
          model, optimizer, ema_model, scheduler, model_trainer,
          task_specific_info):
    """
    Train function for training
    """
    global past_batch
    global past_epoch
    global test_accs
    global best_test_acc
    best_test_acc = 0.0
    # resume from checkpoint if quit accidentally
    if cfg.resume:
        logger.info("**************resume from checkpoint when start training**************")
        resume(args, model, optimizer, scheduler, task_specific_info, ema_model, pth=cfg.resume_pth)
        temp_best = torch.load(os.path.join(args.out, "checkpoint.pth.tar"))
        best_test_acc = temp_best["best_acc"]
        test_accs.append(best_test_acc)
    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    args.start_epoch = past_epoch
    for epoch in range(args.start_epoch, args.epochs + 1):
        best_val_acc = 0
        val_accs = []
        # init logger
        meter_manager = AverageMeterManeger()
        meter_manager.register('batch_time')
        meter_manager.register('data_time')
        end = time.time()
        model.train()
        args.start_batch = past_batch
        # start from step next to checkpoint
        save_batch = 0
        for batch_idx in range(args.start_batch + 1, args.eval_step + 1):
            try:
                data_x = next(labeled_iter)
            except Exception:
                labeled_iter = iter(labeled_trainloader)
                data_x = next(labeled_iter)
            try:
                data_u = next(unlabeled_iter)
            except Exception:
                unlabeled_iter = iter(unlabeled_trainloader)
                data_u = next(unlabeled_iter)

            meter_manager.data_time.update(time.time() - end)
            # calculate loss
            loss_dict = model_trainer.compute_loss(
                data_x=data_x,
                data_u=data_u,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                iter=batch_idx,
                ema_model=ema_model,
                task_specific_info=task_specific_info,
                SCALER=SCALER if SCALER is not None else None)
            # update logger
            meter_manager.try_register_and_update(loss_dict)

            # step
            if SCALER is not None:
                SCALER.step(optimizer)
            else:
                optimizer.step()
            scheduler.step()

            # Updates the scale for next iteration
            if SCALER is not None:
                SCALER.update()

            # update ema if needed
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            meter_manager.batch_time.update(time.time() - end)
            end = time.time()
            meter_desc = meter_manager.get_desc()
            train_acc = meter_manager.get_avg_metric("pseudo_acc")
            train_loss = meter_manager.get_avg_metric("loss")
            if batch_idx % 100 == 0:
                # val - process， save model when the val_accuracy is the highest
                val_model, val_prefix = get_test_model(ema_model, model,
                                                       args.use_ema)
                val_loss, val_acc = test(args, val_loader, val_model, epoch, is_val=True)
                val_accs.append(val_acc)
                is_val_best = val_acc >= best_val_acc
                if is_val_best:
                    best_val_acc = val_acc
                # save model per 100 batches
                    save_ckpt_dict(args, model, ema_model, epoch, batch_idx,
                                   optimizer, scheduler, val_acc, logger, is_val=True)
                logger.info(
                    " Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f} val_loss:{val_loss:.4f} "
                    " val_acc:{val_acc:.2f} {desc} "
                    .format(epoch=epoch,
                            epochs=args.epochs,
                            batch=batch_idx,
                            iter=args.eval_step,
                            lr=scheduler.get_last_lr()[0],
                            val_loss=val_loss,
                            val_acc=val_acc,
                            desc=meter_desc)
                )
                args.writer.add_scalars(main_tag="train",
                                        tag_scalar_dict={'Validation Accuracy': val_acc,
                                                         'Validation loss': val_loss * 3,
                                                         'Training Accuracy': train_acc * 100,
                                                         'Training loss': train_loss * 3,
                                                         }, global_step=batch_idx + (epoch - 1) * cfg.data.eval_step)
            else:
                logger.info(
                    " Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f} "
                    "{desc}"
                    .format(epoch=epoch,
                            epochs=args.epochs,
                            batch=batch_idx,
                            iter=args.eval_step,
                            lr=scheduler.get_last_lr()[0],
                            desc=meter_desc)
                )
                args.writer.add_scalars(main_tag="train",
                                        tag_scalar_dict={'Training Accuracy': train_acc * 100,
                                                         'Training loss': train_loss * 3,
                                                         }, global_step=batch_idx + (epoch - 1) * cfg.data.eval_step)
            save_batch = batch_idx
        # Drawing settings
        if cfg.resume:
            past_batch = 0
        if epoch >= 2:
            # Replicas were created and the model with the highest accuracy in the validation set was used as a test
            model_test = model_builder.build(cfg.model, logger, is_temp=True)
            model_test.to(args.device)
            optimizer_test = optim_builder.build(cfg.optimizer, model_test)
            scheduler_test = set_scheduler(cfg, args, optimizer_test)
            task_specific_info_test = dict()
            if cfg.get("ema", False) and cfg.ema.use:
                args.use_ema = True
                from model.ema import ModelEMA
                ema_model_test = ModelEMA(args.device, model_test, cfg.ema.decay)
            else:
                ema_model_test = None
            resume(args, model_test, optimizer_test, scheduler_test,
                   task_specific_info_test, ema_model_test, is_temp=True)
            save_model_best(ema_model_test, model_test, args, test_loader, epoch, save_batch,
                                   optimizer_test, scheduler_test, logger, test_accs)
            del model_test, optimizer_test, scheduler_test, task_specific_info_test, ema_model_test
        else:
            save_model_best(ema_model, model, args, test_loader, epoch, save_batch,
                       optimizer, scheduler, logger, test_accs)
    args.writer.close()
    return best_test_acc


# test/validate step
def test(args, test_loader, model, epoch, is_val=False):
    """ Test function for model and loader
        when the model is ema model, will test the ema model
        when the model is model, will test the regular model
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    end = time.time()
    # The validation process does not require a progress bar
    if not is_val:
        test_loader = tqdm(test_loader, disable=False)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()
            if torch.cuda.is_available():
                inputs = inputs.to(args.device)
                targets = targets.to(args.device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = F.cross_entropy(outputs, targets)
            prec = accuracy(outputs, targets)
            losses.update(loss.item(), inputs.shape[0])
            acc.update(prec.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not is_val:
                test_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. \
                    Batch: {bt:.3f}s. Loss: {loss:.4f}. acc: {acc:.2f}."
                    .format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        acc=acc.avg
                    ))
        if not is_val:
            test_loader.close()
    if not is_val:
        logger.info("Epoch {} test_acc: {:.2f}".format(epoch, acc.avg))
    return losses.avg, acc.avg


if __name__ == '__main__':
    main()
