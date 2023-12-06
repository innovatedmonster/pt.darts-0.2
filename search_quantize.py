""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config import SearchConfig
import utils
from models.search_cnn2 import SearchCNNController
from architect2 import Architect #量化版本的architect
from visualize import plot

import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp
import copy

import importlib
importlib.invalidate_caches()

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12335'

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
device = torch.device("cuda:0")
DEVICE0 = torch.device("cuda:0")

config = SearchConfig()

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)

weights_conv_path = 'weights_conv.log'
alpha_path = 'alphas.log'
param_path = 'params.log'
weights_fc_path = 'weights_fc.log'
weights_all_path = 'weights_all.log'

def main(rank, lock):
    #数据并行,启动多进程
    dist.init_process_group("nccl", rank=rank, world_size=len(config.gpus))

    #test
    import os
    with lock:
        if os.path.exists(weights_conv_path):
            os.remove(weights_conv_path)
        if os.path.exists(alpha_path):
            os.remove(alpha_path)
        if os.path.exists(param_path):
            os.remove(param_path)
        if os.path.exists(weights_fc_path):
            os.remove(weights_fc_path)
        if os.path.exists(weights_all_path):
            os.remove(weights_all_path)

    if device == DEVICE0:
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

    model = model.to(device)#这里我在quantize之前就将模型移动到cuda，这会导致qunatize后模型一部分量化算子留在cpu上从而报错

    #quantized1, quantize
    model.quantize(num_bits=8)#如果在optim之后才quantize，会导致quantize中生成的权重没有被放入到optim中？是的

    # weights optimizer, config.w_lr
    # w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
    #                           weight_decay=config.w_weight_decay)
    # tmp test
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay,
                              dampening=0, nesterov=False, maximize=False, differentiable=False)

    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)

    # split data to train/validation
    n_train = len(train_data)
    split = n_train // 2
    indices = list(range(n_train))
    #数据并行，处理数据1，和train_batch_sampler冲突，暂时不使用，但是可能会造成训练效果差的后果,见https://zhuanlan.zhihu.com/p/358974461
    # train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    # valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    # train_dataset = torch.utils.data.Subset(train_data, indices[:split])
    # val_dataset = torch.utils.data.Subset(train_data, indices[split:])

    train_dataset, val_dataset = torch.utils.data.random_split(train_data, [int(split), n_train-split])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    #数据并行，处理数据2，如果不指定，不会对batch_size进行划分，见https://blog.csdn.net/weixin_41978699/article/details/121412128
    #这里有个坑，指定了batch_sampler，就不能指定batch_size, shuffle, sampler, and drop_last
    #重要：数据并行中，DataLoader中batch_size指定了每个gpu运行的batch_size，一个epoch的总batch_size为batch_sampler的*gpu个数
    #前提是使用了DistributedSampler类的train_sampler
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size,
                                            #    batch_sampler=train_batch_sampler,
                                               sampler=train_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=config.batch_size,
                                            #    batch_sampler=valid_batch_sampler,
                                               sampler=valid_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)
    
    model = model.to(device)#在quantize后必须把模型再移动到cuda上一次
    #新版数据并行
    if torch.cuda.device_count() > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    architect = Architect(model, config.w_momentum, config.w_weight_decay)

    #test w_optim's elements
    if device == DEVICE0:
        with open(param_path, 'a') as file:
            for n, p in model.named_parameters():
                if 'stem' not in n and 'cell' not in n:
                    file.write(str(n) + '\n')

    # training loop
    best_top1 = 0.
    for epoch in range(config.epochs):
        #tmp test
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        # print('learning rate is ', lr)

        #fix, if you use 'DataParallel', the model will be wrapped in DataParallel(). 
        # It means you need to change the model.function() to model.module.function() in the following codes. 
        # For example, model.function --> model.module.function
        if device == DEVICE0:
            model.module.print_alphas(logger)

        #test
        # with open(file_path, 'a') as file:
        #     file.write('\n\n-----------\nepoch is: ' + str(epoch+1) + '\n')
        #     file.write('after clean model is: ')

        train_sampler.set_epoch(epoch)# 为了每个epoch之间是不同顺序的

        quantize_aware_train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch, lock)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        #bug fixed, 训练->推理的循环会导致上一次的freeze生成的qi在本次train和freeze时出现，从而报错

        #bug, NaN产生的根本原因是上次epoch产生的QPram在本次epoch中对w、b的异常更新
        # model.module.freeze()
        
        #tmp test fake_quantize_inference
        # top1 = quantize_inference(valid_loader, model, epoch, cur_step)
        top1 = fake_quantize_inference(valid_loader, model, epoch, cur_step)

        #bug, 复合算子比如QSearchCell直接调用父类cleanQi 
        # 语句if 'q' in name and isinstance(module, (QModule, QModule2)):
        # 会调用两次同一个QModule的cleanQi
        # model.module.cleanQi()
        
        #test
        # with open(file_path, 'a') as file:
        #     # 或者遍历模型的state_dict中的键
        #     for name, param in model.state_dict().items():
        #         if "q" in name:
        #             file.write(str(name) + str(param) + '\n')

        # 先从混合边中选出softmax(alpha)最大的op, 然后每个中间结点的输入边只保留2条
        genotype = model.module.genotype()
        if device == DEVICE0:
            logger.info("genotype = {}".format(genotype))

        # genotype as a image
        plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
        caption = "Epoch {}".format(epoch+1)
        if device == DEVICE0:
            plot(genotype.normal, plot_path + "-normal", caption)
            plot(genotype.reduce, plot_path + "-reduce", caption)

        with lock:
            if device == DEVICE0:
                if best_top1 < top1:
                    best_top1 = top1
                    best_genotype = genotype
                    is_best = True
                else:
                    is_best = False
                #数据并行，保存模型
                utils.save_checkpoint(model, config.path, is_best)

    if device == DEVICE0:
        logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
        logger.info("Best Genotype = {}".format(best_genotype))
    #（5）将普通推理改成量化推理；


def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()

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
        logits = model(trn_X)
        loss = model.criterion(logits, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        
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

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

def quantize_aware_train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch, lock):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    if device == DEVICE0:
        writer.add_scalar('train/lr', lr, cur_step)

    model.train()

    # DataParallel， 数据不再移动到device即0上，因为在调用nn.DataParallel时就已经移动到对应gpu上了？
    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = trn_X.size(0)

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()

        #test, 打印w_optim中的参数
        # 在第一个epoch后alpha之外的参数的导数变成了整数
        # file_path = 'w_optim.log'
        # with open(file_path, 'a') as file:
        #     file.write('step is: ' + str(step+1) + ', w_optim are:\n')
        #     for group in w_optim.param_groups:
        #         for param in group['params']:
        #                 file.write(str(type(param.data)) + '\n')

        #bug, NaN出现
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)#此处更新了alpha的梯度

        alpha_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad()
        #fix DataParallel
        logits = model.module.quantize_forward(trn_X)
        loss = model.module.criterion(logits, trn_y)
        loss.backward()

        # test loss from quant
        # with torch.no_grad():#防止内存泄漏
        #     if step % config.print_freq == 0 or step == len(train_loader)-1:
        #         logits_quant = model(trn_X)
        #         loss_quant = model.criterion(logits_quant, logits)
        #         logger.info('loss_quant is %s\n', loss_quant)

        # gradient clipping
        nn.utils.clip_grad_norm_(model.module.weights(), config.w_grad_clip)           

        #test weights, lr, grad
        # with lock:
        #     if device == DEVICE0:
        #         if (step+1)%10 == 0:
        #             print('step: ', step+1)
                
        #         current_lr = w_optim.param_groups[0]['lr']
        #         weights_conv = model.module.net.qStem[0].conv_module.weight[0][0]
        #         grad_conv = model.module.net.qStem[0].conv_module.weight.grad[0][0]
        #         with open(weights_conv_path, 'a') as file:
        #             file.write('\n\n-----------\n[step, device] is: [' + str(step+1) + ', '+ str(device) + ']:\n')
        #             # str_conv_weight = utils.getString(weights_conv, 8)
        #             file.write('conv weight is\n' + str(weights_conv) + '\n')

        #             file.write('w_optim\'s cur_lr is: ' + str(current_lr) + '\n')
        #             file.write('conv weight grad is: \n' + str(grad_conv) + '\n')
        #             file.write('lr x grad is: \n' + str(grad_conv*current_lr) + '\n')

        #             weights_conv_should_be = weights_conv - grad_conv*current_lr 
        #             file.write('should next weight be: \n' + str(weights_conv_should_be) + '\n')

                    
        #         weights_fc = model.module.net.qlinear.fc_module.weight
        #         grad_fc = model.module.net.qlinear.fc_module.weight.grad
        #         with open(weights_fc_path, 'a') as file:
        #             file.write('\n\n-----------\n[step, device] is: [' + str(step+1) + ', '+ str(device) + ']:\n')
        #             file.write('fc weight is\n' + str(weights_fc[0][0]) + '\n')

        #             file.write('w_optim\'s cur_lr is: ' + str(current_lr) + '\n')
        #             file.write('fc weight grad is: ' + str(grad_fc[0][0]) + '\n')
        #             file.write('lr x grad is: ' + str(grad_fc[0][0]*current_lr) + '\n')

        #             weights_fc_should_be = weights_fc - grad_fc*current_lr
        #             file.write('should next weight be: \n' + str(weights_fc_should_be[0][0]) + '\n')

        # test all weights
        # with lock:
        #     current_lr = w_optim.param_groups[0]['lr']
        #     if device == DEVICE0:
        #         weights_all_should_be = copy.deepcopy(list(model.named_parameters()))
        #         for index, (n, p) in enumerate(model.named_parameters()):
        #             if 'stem' not in n and 'cell' not in n:
        #                 weights_all_should_be[index] = (p-p.grad*current_lr).clone()

        w_optim.step()

        # with lock:
        #     if device == DEVICE0:
        #         with open(weights_all_path, 'a') as file:
        #             if step == 0:
        #                 for index, (n, p) in enumerate(model.named_parameters()):
        #                     if 'stem' not in n and 'cell' not in n:
        #                     # if 'cell' not in n:
        #                         fit_gd_code = torch.allclose(weights_all_should_be[index], p)
        #                         file.write(str(n) + ' fit gd code? ' + str(fit_gd_code) +'\n')
        #                 print('done')

        # test alpha from different process
        # with lock:
        #     with open(alpha_path, 'a') as file:
        #         file.write('\n\n-----------\n[step, device] is: [' + str(step+1) + ', '+ str(device) + ']:\n')
        #         for i, param_normal in enumerate(model.module.alpha_normal):
        #             file.write('param_normal:\n' + str(param_normal))
        # dist.barrier()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if device == DEVICE0:
            if step % config.print_freq == 0 or step == len(train_loader)-1:
                logger.info(
                    "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                        top1=top1, top5=top5))
                #test
        

            writer.add_scalar('train/loss', loss.item(), cur_step)
            writer.add_scalar('train/top1', prec1.item(), cur_step)
            writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

        #bug prob, reason unknown
        # with lock:
        #     if step == 2:
        #         break
        
    dist.barrier()


    if device == DEVICE0:
        logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


def quantize_inference(valid_loader, model, epoch, cur_step):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model.module.quantize_inference(X)
            loss = model.module.criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if device == DEVICE0:
                if step % config.print_freq == 0 or step == len(valid_loader)-1:
                    logger.info(
                        "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                        "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                            epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                            top1=top1, top5=top5))
                #test
                # if step+1 == 2:
                #     break
        if device == DEVICE0:
            writer.add_scalar('val/loss', losses.avg, cur_step)
            writer.add_scalar('val/top1', top1.avg, cur_step)
            writer.add_scalar('val/top5', top5.avg, cur_step)

            logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return top1.avg

def fake_quantize_inference(valid_loader, model, epoch, cur_step):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model.module.fake_quantize_inference(X)
            loss = model.module.criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if device == DEVICE0:
                if step % config.print_freq == 0 or step == len(valid_loader)-1:
                    logger.info(
                        "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                        "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                            epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                            top1=top1, top5=top5))
                #test
                # if step+1 == 2:
                #     break
        if device == DEVICE0:
            writer.add_scalar('val/loss', losses.avg, cur_step)
            writer.add_scalar('val/top1', top1.avg, cur_step)
            writer.add_scalar('val/top5', top5.avg, cur_step)

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

