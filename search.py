""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config import SearchConfig
import utils
from models.search_cnn import SearchCNNController
from architect import Architect
from visualize import plot

import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp
import copy
import os

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'

config = SearchConfig()

device = torch.device("cuda:0")
weights_conv_path = 'weights_conv.log'
weights_fc_path = 'weights_fc.log'
weights_all_path = 'weights_all.log'

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)


def main(rank, lock):
    dist.init_process_group("nccl", rank=rank, world_size=len(config.gpus))

    with lock:
        if os.path.exists(weights_conv_path):
            os.remove(weights_conv_path)
        if os.path.exists(weights_fc_path):
            os.remove(weights_fc_path)
        if os.path.exists(weights_all_path):
            os.remove(weights_all_path)

    if device == torch.device("cuda:0"):
        logger.info("Logger is set - training start")

    # set default gpu device id
    # torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=False)

    net_crit = nn.CrossEntropyLoss().to(device)
    model = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers,
                                net_crit, device_ids=config.gpus)
    model = model.to(device)

    # weights optimizer
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)
    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)

    # split data to train/validation
    n_train = len(train_data)
    split = n_train // 2
    indices = list(range(n_train))

    # train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    # valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    # train_dataset = torch.utils.data.Subset(train_data, indices[:split])
    # val_dataset = torch.utils.data.Subset(train_data, indices[split:])
    train_dataset, val_dataset = torch.utils.data.random_split(train_data, [int(split), n_train-split])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=config.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)
    
    if torch.cuda.device_count() > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    architect = Architect(model, config.w_momentum, config.w_weight_decay)

    # training loop
    best_top1 = 0.
    for epoch in range(config.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        with lock:
            if device == torch.device("cuda:0"):
                model.module.print_alphas(logger)

        train_sampler.set_epoch(epoch)
        # training
        train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch, lock)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(valid_loader, model, epoch, cur_step)

        # log
        # genotype
        genotype = model.module.genotype()
        with lock:
            if device == torch.device("cuda:0"):
                logger.info("genotype = {}".format(genotype))
                # genotype as a image
                plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
                caption = "Epoch {}".format(epoch+1)
                plot(genotype.normal, plot_path + "-normal", caption)
                plot(genotype.reduce, plot_path + "-reduce", caption)
                # save
                if best_top1 < top1:
                    best_top1 = top1
                    best_genotype = genotype
                    is_best = True
                else:
                    is_best = False
                utils.save_checkpoint(model, config.path, is_best)
                print("")

                logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
                logger.info("Best Genotype = {}".format(best_genotype))


def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch, lock):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    with lock:
            if device == torch.device("cuda:0"):
                writer.add_scalar('train/lr', lr, cur_step)

    model.train()
    pre_weight = model.module.net.stem[0].weight[0][0].clone()

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = trn_X.size(0)

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)
        alpha_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad()
        #（3）将训练的forward替换成量化训练的quantize_forward;
        logits = model.module(trn_X)
        loss = model.module.criterion(logits, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.module.weights(), config.w_grad_clip)
        with lock:
            if device == torch.device("cuda:0"):
                if (step+1)%10 == 0:
                    print('step: ', step+1)
                
                current_lr = w_optim.param_groups[0]['lr']
                weights_conv = model.module.net.stem[0].weight
                grad_conv = weights_conv.grad
                with open(weights_conv_path, 'a') as file:
                    file.write('\n\n-----------\n[step, device] is: [' + str(step+1) + ', '+ str(device) + ']:\n')
                    # str_conv_weight = utils.getString(weights_conv, 8)
                    file.write('conv weight is\n' + str(weights_conv) + '\n')

                    file.write('w_optim\'s cur_lr is: ' + str(current_lr) + '\n')
                    file.write('conv weight grad is: \n' + str(grad_conv) + '\n')
                    file.write('lr x grad is: \n' + str(grad_conv*current_lr) + '\n')

                    weights_conv_should_be = weights_conv - grad_conv*current_lr 
                    file.write('should next weight be: \n' + str(weights_conv_should_be) + '\n')

                    
                weights_fc = model.module.net.linear.weight
                grad_fc = weights_fc.grad
                with open(weights_fc_path, 'a') as file:
                    file.write('\n\n-----------\n[step, device] is: [' + str(step+1) + ', '+ str(device) + ']:\n')
                    file.write('fc weight is\n' + str(weights_fc[0][0]) + '\n')

                    file.write('w_optim\'s cur_lr is: ' + str(current_lr) + '\n')
                    file.write('fc weight grad is: ' + str(grad_fc[0][0]) + '\n')
                    file.write('lr x grad is: ' + str(grad_fc[0][0]*current_lr) + '\n')

                    weights_fc_should_be = weights_fc - grad_fc*current_lr
                    file.write('should next weight be: \n' + str(weights_fc_should_be[0][0]) + '\n')

        # test all weights
        with lock:
            if device == torch.device("cuda:0"):
                weights_all_should_be = copy.deepcopy(list(model.named_parameters()))
                for index, (n, p) in enumerate(model.named_parameters()):
                    weights_all_should_be[index] = (p-p.grad*current_lr).clone()


        w_optim.step()

        with lock:
            if device == torch.device("cuda:0"):
                with open(weights_all_path, 'a') as file:
                    for index, (n, p) in enumerate(model.named_parameters()):
                        fit_gd_code = torch.allclose(weights_all_should_be[index], p)
                        file.write(str(n) + ' fit gd code? ' + str(fit_gd_code) +'\n')
        

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        with lock:
            if device == torch.device("cuda:0"):
                if step % config.print_freq == 0 or step == len(train_loader)-1:
                    logger.info(
                        "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                        "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                            epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                            top1=top1, top5=top5))

                writer.add_scalar('train/loss', loss.item(), cur_step)
                writer.add_scalar('train/top1', prec1.item(), cur_step)
                writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    with lock:
        if device == torch.device("cuda:0"):
            logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


def validate(valid_loader, model, epoch, cur_step):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model.module(X)
            loss = model.module.criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            with lock:
                if device == torch.device("cuda:0"):
                    if step % config.print_freq == 0 or step == len(valid_loader)-1:
                        logger.info(
                            "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                            "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                                epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                                top1=top1, top5=top5))

                    writer.add_scalar('val/loss', losses.avg, cur_step)
                    writer.add_scalar('val/top1', top1.avg, cur_step)
                    writer.add_scalar('val/top5', top5.avg, cur_step)

    with lock:
            if device == torch.device("cuda:0"):
                logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return top1.avg


if __name__ == "__main__":
    lock = mp.Lock() # for test

    size = len(config.gpus)
    processes = []
    for rank in range(size):
        device = torch.device("cuda:" + str(rank))
        p = mp.Process(target=main, args=(rank, lock))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
