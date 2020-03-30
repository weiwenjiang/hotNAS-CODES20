from __future__ import print_function
from collections import defaultdict, deque
import datetime
import time
import torch
import torch.distributed as dist

import errno
import os
import numpy as np

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)



def admm_loss(device, model, layer_names, criterion, Z, U, output, target):
    idx = 0
    loss = criterion(output, target)
    for name in layer_names:
        u = U[name].to(device)
        z = Z[name].to(device)
        loss += 1e-2 / 2 * (model.state_dict()[name + ".weight"][:] - z + u).norm()
        idx += 1
    return loss


def initialize_Z_and_U(model, layer_names):
    Z = {}
    U = {}

    for name in layer_names:
        Z[name] = model.state_dict()[name + ".weight"][:].detach().cpu().clone()
        U[name] = torch.zeros_like(model.state_dict()[name + ".weight"][:]).cpu()

    return Z, U

def update_X(model, layer_names):
    X = {}
    for name in layer_names:
        X[name] = model.state_dict()[name + ".weight"][:].detach().cpu().clone()
    return X

def update_U(U, X, Z, layer_names):
    new_U = {}
    for name in layer_names:
        new_u = U[name] + X[name] - Z[name]
        new_U[name] = new_u
    return new_U



def update_Z_Pattern(X, U, layer_names, pattern):
    new_Z = {}
    layer_pattern = {}
    for name in layer_names:

        z = (X[name] + U[name])
        shape = list(z.shape[:-2])
        shape.append(1)
        shape.append(1)
        after_pattern_0 = z * pattern[0]
        after_norm_0 = after_pattern_0.norm(dim=(2, 3)).reshape(shape)
        after_pattern_1 = z * pattern[1]
        after_norm_1 = after_pattern_1.norm(dim=(2, 3)).reshape(shape)
        after_pattern_2 = z * pattern[2]
        after_norm_2 = after_pattern_2.norm(dim=(2, 3)).reshape(shape)
        after_pattern_3 = z * pattern[3]
        after_norm_3 = after_pattern_3.norm(dim=(2, 3)).reshape(shape)

        max_norm = (torch.max(torch.max(torch.max(after_norm_0, after_norm_1), after_norm_2), after_norm_3))
        tmp_pattern = torch.zeros_like(z)

        tmp_pattern = tmp_pattern + (after_norm_0 == max_norm).float() * pattern[0] + \
            ((after_norm_1 == max_norm) & (after_norm_0 != max_norm)).float() * pattern[1] + \
                  (after_norm_2 == max_norm).float() * pattern[2] + \
                  (after_norm_3 == max_norm).float() * pattern[3]

        z = z * tmp_pattern

        new_Z[name] = z
        layer_pattern[name] = tmp_pattern

    return new_Z,layer_pattern





def update_Z(X, U, layer_names, percent):
    new_Z = {}
    idx = 0

    for name in layer_names:
        z = X[name] + U[name]
        pcen = np.percentile(abs(z), 100 * percent[idx])
        under_threshold = abs(z) < pcen
        z.data[under_threshold] = 0
        new_Z[name] = z
        idx += 1
    return new_Z


def prune_weight(weight, device, percent):
    weight_numpy = weight.detach().cpu().numpy()
    pcen = np.percentile(abs(weight_numpy), 100 * percent)
    under_threshold = abs(weight_numpy) < pcen
    weight_numpy[under_threshold] = 0
    mask = torch.Tensor(abs(weight_numpy) >= pcen).to(device)
    return mask

def apply_prune(model, layer_names, device, percent):
    print("Apply Pruning based on percent:",percent)
    dict_mask = {}
    idx = 0
    for name in layer_names:
        mask = prune_weight(model.state_dict()[name + ".weight"][:], device, percent[idx])
        model.state_dict()[name + ".weight"][:].data.mul_(mask)
        dict_mask[name] = mask
        idx += 1
    return dict_mask


def apply_prune_pattern(model, layer_names, pattern, device):
    print("Apply Pruning based on pattern")
    # for name in layer_names:
    #     model.state_dict()[name + ".weight"][:].data.mul_((layer_pattern[name]).to(device))
    layer_pattern = {}
    for name in layer_names:
        z = model.state_dict()[name + ".weight"][:].detach().cpu().clone()
        shape = list(z.shape[:-2])
        shape.append(1)
        shape.append(1)
        after_pattern_0 = z * pattern[0]
        after_norm_0 = after_pattern_0.norm(dim=(2, 3)).reshape(shape)
        after_pattern_1 = z * pattern[1]
        after_norm_1 = after_pattern_1.norm(dim=(2, 3)).reshape(shape)
        after_pattern_2 = z * pattern[2]
        after_norm_2 = after_pattern_2.norm(dim=(2, 3)).reshape(shape)
        after_pattern_3 = z * pattern[3]
        after_norm_3 = after_pattern_3.norm(dim=(2, 3)).reshape(shape)

        max_norm = (torch.max(torch.max(torch.max(after_norm_0, after_norm_1), after_norm_2), after_norm_3))
        tmp_pattern = torch.zeros_like(z)

        tmp_pattern = tmp_pattern + (after_norm_0 == max_norm).float() * pattern[0] + \
                      ((after_norm_1 == max_norm) & (after_norm_0 != max_norm)).float() * pattern[1] + \
                      (after_norm_2 == max_norm).float() * pattern[2] + \
                      (after_norm_3 == max_norm).float() * pattern[3]

        model.state_dict()[name + ".weight"][:].data.mul_((tmp_pattern).to(device))
        layer_pattern[name] = tmp_pattern

    return layer_pattern

def print_prune(model, layer_names):
    prune_param, total_param = 0, 0

    if len(layer_names)==0:
        print("No pruning layer is provided")
    else:
        for name in layer_names:
            print("[at weight {}]".format(name))
            print("percentage of pruned: {:.4f}%".format(100 * (abs(model.state_dict()[name + ".weight"][:]) == 0).sum().item() / model.state_dict()[name + ".weight"][:].numel()))
            print("nonzero parameters after pruning: {} / {}\n".format((model.state_dict()[name + ".weight"][:] != 0).sum().item(), model.state_dict()[name + ".weight"][:].numel()))

            total_param += model.state_dict()[name + ".weight"][:].numel()
            prune_param += (model.state_dict()[name + ".weight"][:] != 0).sum().item()

        print("total nonzero parameters after pruning: {} / {} ({:.4f}%)".
              format(prune_param, total_param,
                     100 * (total_param - prune_param) / total_param))