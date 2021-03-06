from __future__ import print_function
import datetime
import os
import time
import sys

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms

sys.path.append("../Interface")
from ztNAS_model_change import *

import utils
from optimizer import PruneAdam
from termplot import Plot


import random

try:
    from apex import amp
except ImportError:
    amp = None


def re_train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, layer_names,
                    layer_pattern, data_loader_test, explore, apex=False):
    # Z, U = utils.initialize_Z_and_U(model, layer_names)

    # Plot([float(x) for x in list(Z[layer_names[-1]].flatten())], plot_type=2)

    re_train_start_time = time.time()


    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)

    batch_idx = 0

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image)

        # loss = utils.admm_loss(device, model, layer_names, criterion, Z, U, output, target, rho)
        loss = criterion(output, target)

        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.prune_step(layer_names, layer_pattern)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

        batch_idx += 1
        if batch_idx==100:
            total_time = time.time() - re_train_start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Elapsed Time {} for {} batches".format(total_time_str,batch_idx))
            # evaluate(model, criterion, data_loader_test, device=device)
            if explore:
                return
        elif batch_idx==200:
            total_time = time.time() - re_train_start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Elapsed Time {} for {} batches".format(total_time_str,batch_idx))
            evaluate(model, criterion, data_loader_test, device=device)
        elif batch_idx%1000==0:
            total_time = time.time() - re_train_start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Elapsed Time {} for {} batches".format(total_time_str,batch_idx))
            evaluate(model, criterion, data_loader_test, device=device)



def train_one_epoch(model, criterion, admm_optimizer, data_loader, device, epoch, print_freq, layer_names,
                    percent, pattern, Z, U, arg_rho, apex=False):



    # Plot([float(x) for x in list(Z[layer_names[-1]].flatten())], plot_type=2)

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)


    rho = arg_rho
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image)



        loss = utils.admm_loss(device, model, layer_names, criterion, Z, U, output, target, rho)

        admm_optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, admm_optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        admm_optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=admm_optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))


    print("=" * 10, "Entering ADMM Optimization")
    X = utils.update_X(model, layer_names)
    Z, layer_pattern = utils.update_Z_Pattern(X, U, layer_names, pattern)
    U = utils.update_U(U, X, Z, layer_names)

    return Z,U




def evaluate(model, criterion, data_loader, device, exploration=False, print_freq=100):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        batch_idx = 0
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

            batch_idx+=1
            # if exploration and batch_idx == 50:
            #     print("Exit early due to explroation")
            #     return metric_logger.acc1, metric_logger.acc5
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    # return metric_logger.acc1.global_avg
    return metric_logger.acc1.global_avg, metric_logger.acc5.global_avg


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, cache_dataset, distributed):
    # Data loading code
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_train to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler

#
# def modify_model(vgg):
#
#     for param in vgg.parameters():
#         param.requires_grad = False
#     layers = list(vgg.layer4[0].children())[:-1]
#
#     bck1 = vgg.state_dict()["layer4.0.conv1.weight"][:]
#     bck2 = vgg.state_dict()["layer4.0.bn1.weight"][:]
#     bck3 = vgg.state_dict()["layer4.0.bn1.bias"][:]
#     bck4 = vgg.state_dict()["layer4.0.conv2.weight"][:]
#     bck5 =  vgg.state_dict()["layer4.0.bn1.running_mean"][:]
#     bck6 =  vgg.state_dict()["layer4.0.bn1.running_var"][:]
#
#     ch = 460
#
#     print("="*100)
#     for name, param in vgg.named_parameters():
#         print (name,param.requires_grad,param.data.shape,param.data.min())
#
#
#     layers[0] = torch.nn.Conv2d(256, ch, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#     layers[1] = torch.nn.BatchNorm2d(ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     layers[3] = torch.nn.Conv2d(ch, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#
#     vgg.layer4[0].conv1 = layers[0]
#     vgg.layer4[0].bn1 = layers[1]
#     vgg.layer4[0].conv2 = layers[3]
#
#     print(bck1.shape, bck4.shape)
#
#     vgg.state_dict()["layer4.0.conv1.weight"][:] = bck1[0:ch,:,:,:]
#     vgg.state_dict()["layer4.0.bn1.weight"][:] = bck2[0:ch]
#     vgg.state_dict()["layer4.0.bn1.bias"][:] = bck3[0:ch]
#     vgg.state_dict()["layer4.0.conv2.weight"][:] = bck4[:,0:ch,:,:]
#     vgg.state_dict()["layer4.0.bn1.running_mean"][:] = bck5[0:ch]
#     vgg.state_dict()["layer4.0.bn1.running_var"][:] =bck6[0:ch]
#
#
#
#     print("="*100)
#     for name, param in vgg.named_parameters():
#         print (name,param.requires_grad,param.data.shape,param.data.min())
#
#
#     ''' VGG Modifications
#     for param in vgg.parameters():
#         param.requires_grad = False
#     layers = list(vgg.features.children())[:-1]
#
#     features_3_weight = vgg.state_dict()["features.3.weight"][:]
#     features_3_bias = vgg.state_dict()["features.3.bias"][:]
#     features_6_weight = vgg.state_dict()["features.6.weight"][:]
#     features_6_bias = vgg.state_dict()["features.6.bias"][:]
#
#     ch = 126
#
#     layers[3] = torch.nn.Conv2d(64, ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     layers[6] = torch.nn.Conv2d(ch, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#
#     features = torch.nn.Sequential(*layers)
#     vgg.features = features
#
#     vgg.state_dict()["features.3.weight"][:] = features_3_weight[0:ch,:,:,:]
#     vgg.state_dict()["features.3.bias"][:] = features_3_bias[0:ch]
#     vgg.state_dict()["features.6.weight"][:] = features_6_weight[:,0:ch,:,:]
#     vgg.state_dict()["features.6.bias"][:] = features_6_bias[:]
#
#     '''
#     return vgg



def main(args,layer_train_para,layer_names,layer_kernel_inc,pattern):
    if args.apex:
        if sys.version_info < (3, 0):
            raise RuntimeError("Apex currently only supports Python 3. Aborting.")
        if amp is None:
            raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                               "to enable mixed-precision training.")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir,
                                                                   args.cache_dataset, args.distributed)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    model = torchvision.models.__dict__[args.model](pretrained=args.pretrained)



    # layer_train_para = [
    #     "layer1.0.conv1.weight",
    #     "layer1.0.bn1.weight",
    #     "layer1.0.bn1.bias",
    #     "layer1.0.conv2.weight",
    #     "layer1.0.bn2.weight",
    #     "layer1.0.bn2.bias",
    #     "layer1.1.conv1.weight",
    #     "layer1.1.bn1.weight",
    #     "layer1.1.bn1.bias",
    #     "layer1.1.conv2.weight",
    #     "layer1.1.bn2.weight",
    #     "layer1.1.bn2.bias",
    #     "layer2.0.conv2.weight",
    #     "layer2.0.bn2.weight",
    #     "layer2.0.bn2.bias",
    #     "layer2.0.conv1.weight",
    #     "layer2.0.bn1.weight",
    #     "layer2.0.bn1.bias",
    #     "layer2.0.downsample.0.weight",
    #     "layer2.0.downsample.1.weight",
    #     "layer2.0.downsample.1.bias"]
    #
    # layer_names = [
    #     "layer1.0.conv1",
    #     "layer1.0.conv2",
    #     "layer1.1.conv1",
    #     "layer1.1.conv2",
    #     "layer2.0.conv2",
    #     "layer2.1.conv1",
    #     "layer2.1.conv2"
    # ]
    #
    # layer_kernel_inc = [
    #     # "layer2.0.conv1",
    #     # "layer2.0.downsample.0"
    # ]
    #
    # pattern = {}
    # pattern[0] = torch.tensor([[0, 0, 0],
    #                            [1, 1, 1],
    #                            [1, 1, 1]], dtype=torch.float32)
    #
    # pattern[1] = torch.tensor([[1, 1, 1],
    #                            [1, 1, 1],
    #                            [0, 0, 0]], dtype=torch.float32)
    #
    # pattern[2] = torch.tensor([[1, 1, 0],
    #                            [1, 1, 0],
    #                            [1, 1, 0]], dtype=torch.float32)
    #
    # pattern[3] = torch.tensor([[0, 1, 1],
    #                            [0, 1, 1],
    #                            [0, 1, 1]], dtype=torch.float32)

    layers = {}
    ki_layers = {}
    # for layer_name, layer in model.named_modules():
    for layer_name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            # if is_same(layer.kernel_size) == 3 and layer.in_channels == 512:
            # if is_same(layer.kernel_size) == 3:
            if layer_name in layer_names:
                # layer_names.append(layer_name)
                layers[layer_name] = layer
            if layer_name in layer_kernel_inc:
                ki_layers[layer_name] = layer



        # print(layer_name)
            # if is_same(layer.kernel_size) == 3 and layer.in_channels==512:
            #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #     mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=torch.float32, device=device)
            #     ztNAS_add_kernel_mask(model, layer, layer_name, mask=mask)

    #model = modify_model(model)



    # for name, param in model.named_parameters():
    #     names = [n + "." for n in name.split(".")[:-1]]
    #     if "".join(names)[:-1] not in layer_names:
    #         param.requires_grad = False
    #     else:
    #         break

    for name, param in model.named_parameters():
        if name in layer_train_para:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad, param.data.shape)

    # print(model)

    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    admm_optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, eps=args.adam_epsilon)

    admm_re_train_optimizer = PruneAdam(
        model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level
                                          )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        # for name, param in model.named_parameters():
        #     print(name)git oull
        #     print(param)

        layer_pattern = utils.get_layers_pattern(model, layer_names, pattern, device)
        utils.print_prune(model, layer_names, layer_pattern)

        for layer_name in layer_names:
            ztNAS_add_kernel_mask(model, layers[layer_name], layer_name, is_pattern=True,
                                  pattern=layer_pattern[layer_name].to(device))

        # print(model)
        model.to(device)
        evaluate(model, criterion, data_loader_test, device=device)

        # evaluate(model, criterion, data_loader_test, device=device)
        return

    if args.retrain_only:
        epoch = 999
        print("Start re-training")
        start_time = time.time()
        print("=" * 10, "Applying pruning model")
        layer_pattern = utils.get_layers_pattern(model, layer_names, pattern, device)
        # utils.print_prune(model, layer_names, layer_pattern)

        for layer_name in layer_names:
            ztNAS_add_kernel_mask(model, layers[layer_name], layer_name, is_pattern=True,
                                  pattern=layer_pattern[layer_name].to(device))

        for layer_name in layer_kernel_inc:
            ztNAS_modify_kernel_shape(model,ki_layers[layer_name], layer_name,2)

        # print(model)
        model.to(device)
        # evaluate(model, criterion, data_loader_test, device=device)

        print("=" * 10, "Retrain")

        re_train_one_epoch(model, criterion, admm_re_train_optimizer, data_loader, device, epoch, args.print_freq,
                           layer_names, layer_pattern, data_loader_test, args.exploration, args.apex)

        acc1, acc5 = evaluate(model, criterion, data_loader_test, device=device, exploration = args.exploration)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        return acc1, acc5


    print("Start training")
    start_time = time.time()

    Z, U = utils.initialize_Z_and_U(model, layer_names)
    rho = args.rho
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)


        Z, U = train_one_epoch(model, criterion, admm_optimizer, data_loader, device, epoch, args.print_freq,
                                        layer_names, percent, pattern, Z, U, rho, args.apex)

        rho = rho*10
        lr_scheduler.step()

        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))

        evaluate(model, criterion, data_loader_test, device=device)


    print("="*10,"Applying pruning model")
    layer_pattern = utils.get_layers_pattern(model, layer_names, pattern, device)
    # utils.print_prune(model, layer_names, layer_pattern)

    for layer_name in layer_names:
        ztNAS_add_kernel_mask(model, layers[layer_name], layer_name, is_pattern=True, pattern=layer_pattern[layer_name].to(device))

    # print(model)
    model.to(device)
    # evaluate(model, criterion, data_loader_test, device=device)

    print("=" * 10, "Retrain")

    re_train_one_epoch(model, criterion, admm_re_train_optimizer, data_loader, device, epoch, args.print_freq,
                       layer_names, layer_pattern, data_loader_test, args.exploration, args.apex)

    evaluate(model, criterion, data_loader_test, device=device)

    if args.output_dir:
        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch+1,
            'args': args}
        utils.save_on_master(
            checkpoint,
            os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
        utils.save_on_master(
            checkpoint,
            os.path.join(args.output_dir, 'checkpoint.pth'))


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('Training time {}'.format(total_time_str))




def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-path', default='/mnt/weiwen/ImageNet', help='dataset')
    parser.add_argument('--model', default='resnet18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--rho', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')


    parser.add_argument('--adam_epsilon', type=float, default=1e-8, metavar='E',
                        help='adam epsilon (default: 1e-8)')

    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument(
        "--retrain-only",
        dest="retrain_only",
        help="Only retrain the model",
        action="store_true",
    )

    parser.add_argument(
        "--explore-reduction",
        dest="expore_reduction",
        help="Only explore reduction the model",
        action="store_true",
    )

    parser.add_argument(
        "--exploration",
        dest="exploration",
        help="Help to exit retrain early",
        action="store_true",
    )

    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args



def explore_one(args,pattern,k_expand):
    pattern_num = 4

    start_time = time.time()

    # Manually finetune

    for i in range(pattern_num):
        pattern[i] = pattern[i].reshape((3, 3))

    layer_pattern_train_para = [
        "layer1.0.conv1.weight",
        "layer1.0.bn1.weight",
        "layer1.0.bn1.bias",
        "layer1.0.conv2.weight",
        "layer1.0.bn2.weight",
        "layer1.0.bn2.bias",
        "layer1.1.conv1.weight",
        "layer1.1.bn1.weight",
        "layer1.1.bn1.bias",
        "layer1.1.conv2.weight",
        "layer1.1.bn2.weight",
        "layer1.1.bn2.bias",
        "layer2.0.conv2.weight",
        "layer2.0.bn2.weight",
        "layer2.0.bn2.bias",
        "layer2.1.conv1.weight",
        "layer2.1.bn1.weight",
        "layer2.1.bn1.bias",
        "layer2.1.conv2.weight",
        "layer2.1.bn2.weight",
        "layer2.1.bn2.bias"]

    layer_names = [
        "layer1.0.conv1",
        "layer1.0.conv2",
        "layer1.1.conv1",
        "layer1.1.conv2",
        "layer2.0.conv2",
        "layer2.1.conv1",
        "layer2.1.conv2"
    ]


    if k_expand == 0:
        layer_k_expand_train_para = []
        layer_kernel_inc = []
    elif k_expand == 1:
        layer_k_expand_train_para = ["layer2.0.conv1.weight", "layer2.0.bn1.weight", "layer2.0.bn1.bias"]
        layer_kernel_inc = ["layer2.0.conv1"]
    elif k_expand == 2:
        layer_k_expand_train_para = ["layer2.0.downsample.0.weight", "layer2.0.downsample.1.weight",
                                     "layer2.0.downsample.1.bias"]
        layer_kernel_inc = ["layer2.0.downsample.0"]
    else:
        layer_k_expand_train_para = [
            "layer2.0.conv1.weight",
            "layer2.0.bn1.weight",
            "layer2.0.bn1.bias",
            "layer2.0.downsample.0.weight",
            "layer2.0.downsample.1.weight",
            "layer2.0.downsample.1.bias"]
        layer_kernel_inc = [
            "layer2.0.conv1",
            "layer2.0.downsample.0"
        ]

    layer_train_para = layer_pattern_train_para + layer_k_expand_train_para

    acc1, acc5 = main(args, layer_train_para, layer_names, layer_kernel_inc, pattern)
    print("*" * 40, "Manually explore", "*" * 40)
    for k, v in pattern.items():
        print(k, v)
    print("4", k_expand)

    print("*" * 100)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Search time {}'.format(total_time_str))
    return acc1, acc5


if __name__ == "__main__":
    args = parse_args()
    pattern_num = 4

    start_time = time.time()


    if args.expore_reduction:
        search_counts = 300

        for outer_idx in range(search_counts):

            search_point = {}

            pattern = {}
            for i in range(pattern_num):
                P = torch.ones(9)
                P[random.sample(range(9),3)] = 0
                search_point[i] = P
                pattern[i] = P.reshape((3,3))
            # print(pattern)

            layer_pattern_train_para = [
                "layer1.0.conv1.weight",
                "layer1.0.bn1.weight",
                "layer1.0.bn1.bias",
                "layer1.0.conv2.weight",
                "layer1.0.bn2.weight",
                "layer1.0.bn2.bias",
                "layer1.1.conv1.weight",
                "layer1.1.bn1.weight",
                "layer1.1.bn1.bias",
                "layer1.1.conv2.weight",
                "layer1.1.bn2.weight",
                "layer1.1.bn2.bias",
                "layer2.0.conv2.weight",
                "layer2.0.bn2.weight",
                "layer2.0.bn2.bias",
                "layer2.1.conv1.weight",
                "layer2.1.bn1.weight",
                "layer2.1.bn1.bias",
                "layer2.1.conv2.weight",
                "layer2.1.bn2.weight",
                "layer2.1.bn2.bias"]

            layer_names = [
                "layer1.0.conv1",
                "layer1.0.conv2",
                "layer1.1.conv1",
                "layer1.1.conv2",
                "layer2.0.conv2",
                "layer2.1.conv1",
                "layer2.1.conv2"
            ]


            k_expand = random.choice(range(4))
            search_point[pattern_num] = k_expand
            if k_expand==0:
                layer_k_expand_train_para = []
                layer_kernel_inc = []
            elif k_expand==1:
                layer_k_expand_train_para = ["layer2.0.conv1.weight","layer2.0.bn1.weight","layer2.0.bn1.bias"]
                layer_kernel_inc = ["layer2.0.conv1"]
            elif k_expand==2:
                layer_k_expand_train_para = ["layer2.0.downsample.0.weight","layer2.0.downsample.1.weight","layer2.0.downsample.1.bias"]
                layer_kernel_inc = ["layer2.0.downsample.0"]
            else:
                layer_k_expand_train_para = [
                    "layer2.0.conv1.weight",
                    "layer2.0.bn1.weight",
                    "layer2.0.bn1.bias",
                    "layer2.0.downsample.0.weight",
                    "layer2.0.downsample.1.weight",
                    "layer2.0.downsample.1.bias"]
                layer_kernel_inc = [
                    "layer2.0.conv1",
                    "layer2.0.downsample.0"
                ]

            layer_train_para = layer_pattern_train_para+layer_k_expand_train_para

            main(args,layer_train_para,layer_names,layer_kernel_inc,pattern)
            print("*" * 40,outer_idx,"/",search_counts,"*" * 40)
            for k, v in search_point.items():
                print(k, v)
            print("*" * 100)
    else:

        # Manually finetune

        pattern = {}
        pattern[0] = torch.tensor([1., 1., 1., 1., 0., 1., 1., 0., 0.])
        pattern[1] = torch.tensor([0., 0., 1., 1., 1., 1., 1., 0., 1.])
        pattern[2] = torch.tensor([1., 1., 0., 1., 1., 0., 1., 1., 0.])
        pattern[3] = torch.tensor([1., 0., 0., 1., 1., 1., 0., 1., 1.])

        for i in range(pattern_num):
            pattern[i] = pattern[i].reshape((3, 3))

        layer_pattern_train_para = [
            "layer1.0.conv1.weight",
            "layer1.0.bn1.weight",
            "layer1.0.bn1.bias",
            "layer1.0.conv2.weight",
            "layer1.0.bn2.weight",
            "layer1.0.bn2.bias",
            "layer1.1.conv1.weight",
            "layer1.1.bn1.weight",
            "layer1.1.bn1.bias",
            "layer1.1.conv2.weight",
            "layer1.1.bn2.weight",
            "layer1.1.bn2.bias",
            "layer2.0.conv2.weight",
            "layer2.0.bn2.weight",
            "layer2.0.bn2.bias",
            "layer2.1.conv1.weight",
            "layer2.1.bn1.weight",
            "layer2.1.bn1.bias",
            "layer2.1.conv2.weight",
            "layer2.1.bn2.weight",
            "layer2.1.bn2.bias"]

        layer_names = [
            "layer1.0.conv1",
            "layer1.0.conv2",
            "layer1.1.conv1",
            "layer1.1.conv2",
            "layer2.0.conv2",
            "layer2.1.conv1",
            "layer2.1.conv2"
        ]

        k_expand = 1

        if k_expand == 0:
            layer_k_expand_train_para = []
            layer_kernel_inc = []
        elif k_expand == 1:
            layer_k_expand_train_para = ["layer2.0.conv1.weight", "layer2.0.bn1.weight", "layer2.0.bn1.bias"]
            layer_kernel_inc = ["layer2.0.conv1"]
        elif k_expand == 2:
            layer_k_expand_train_para = ["layer2.0.downsample.0.weight", "layer2.0.downsample.1.weight",
                                         "layer2.0.downsample.1.bias"]
            layer_kernel_inc = ["layer2.0.downsample.0"]
        else:
            layer_k_expand_train_para = [
                "layer2.0.conv1.weight",
                "layer2.0.bn1.weight",
                "layer2.0.bn1.bias",
                "layer2.0.downsample.0.weight",
                "layer2.0.downsample.1.weight",
                "layer2.0.downsample.1.bias"]
            layer_kernel_inc = [
                "layer2.0.conv1",
                "layer2.0.downsample.0"
            ]

        layer_train_para = layer_pattern_train_para + layer_k_expand_train_para

        main(args, layer_train_para, layer_names, layer_kernel_inc, pattern)
        print("*" * 40, "Manually explore", "*" * 40)
        for k,v in pattern.items():
            print(k,v)
        print("4",k_expand)

        print("*" * 100)



    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Search time {}'.format(total_time_str))


