"""
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
import paddle
import argparse
import os
import sys
import time
import math
import random
import datetime
import subprocess
from collections import defaultdict, deque
import numpy as np
from PIL import ImageFilter, ImageOps


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img
        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def load_pretrained_weights(
    model, pretrained_weights, checkpoint_key, model_name, patch_size
):
    if os.path.isfile(pretrained_weights):
        state_dict = paddle.load(path=pretrained_weights)
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.set_state_dict(state_dict=state_dict, use_structured_name=False)
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(
                pretrained_weights, msg
            )
        )
    else:
        print(
            "Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate."
        )
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif model_name == "xcit_small_12_p16":
            url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
        elif model_name == "xcit_small_12_p8":
            url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
        elif model_name == "xcit_medium_24_p16":
            url = (
                "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
            )
        elif model_name == "xcit_medium_24_p8":
            url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        if url is not None:
            print(
                "Since no pretrained weights have been provided, we load the reference pretrained DINO weights."
            )
        else:
            print(
                "There is no reference weights available for this model => We use random weights."
            )


def load_pretrained_linear_weights(linear_classifier, model_name, patch_size):
    url = None
    if model_name == "vit_small" and patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth"
    elif model_name == "vit_small" and patch_size == 8:
        url = "dino_deitsmall8_pretrain/dino_deitsmall8_linearweights.pth"
    elif model_name == "vit_base" and patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth"
    elif model_name == "vit_base" and patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth"
    elif model_name == "resnet50":
        url = "dino_resnet50_pretrain/dino_resnet50_linearweights.pth"
    if url is not None:
        print("We load the reference pretrained linear weights.")
    else:
        print("We use random linear weights.")


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-06)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))
    checkpoint = paddle.load(path=ckp_path)
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.set_state_dict(
                    state_dict=checkpoint[key], use_structured_name=False
                )
                print(
                    "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                        key, ckp_path, msg
                    )
                )
            except TypeError:
                try:
                    msg = value.set_state_dict(state_dict=checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print(
                        "=> failed to load '{}' from checkpoint: '{}'".format(
                            key, ckp_path
                        )
                    )
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    paddle.seed(seed=seed)
    paddle.seed(seed=seed)
    np.random.seed(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
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
        t = paddle.to_tensor(
            data=[self.count, self.total], dtype="float64", place="cuda"
        )
        paddle.distributed.barrier()
        paddle.distributed.all_reduce(tensor=t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = paddle.to_tensor(data=list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = paddle.to_tensor(data=list(self.deque), dtype="float32")
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
            value=self.value,
        )


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with paddle.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = paddle.stack(x=values, axis=0)
        paddle.distributed.all_reduce(tensor=values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, paddle.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if paddle.device.cuda.device_count() >= 1:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if paddle.device.cuda.device_count() >= 1:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=paddle.device.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.6f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def is_dist_avail_and_initialized():
    if not paddle.distributed.is_available():
        return False
    if not paddle.distributed.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return paddle.distributed.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return paddle.distributed.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        paddle.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % paddle.device.cuda.device_count()
    elif paddle.device.cuda.device_count() >= 1:
        print("Will run the code on one GPU.")
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
    else:
        print("Does not support training without GPU.")
        sys.exit(1)
    paddle.distributed.init_parallel_env()
    paddle.device.set_device(device=args.gpu)
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    paddle.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.shape[0]
    _, pred = output.topk(k=maxk, axis=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.equal(y=target.reshape(1, -1).expand_as(y=pred))
    return [
        (
            correct[:k].reshape(-1).astype(dtype="float32").sum(axis=0)
            * 100.0
            / batch_size
        )
        for k in topk
    ]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.",
            stacklevel=2,
        )
    with paddle.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(min=2 * l - 1, max=2 * u - 1)
        tensor.erfinv_()
        tensor.multiply_(y=paddle.to_tensor(std * math.sqrt(2.0)))
        tensor.add_(y=paddle.to_tensor(mean))
        tensor.clip_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class LARS(paddle.optimizer.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """

    def __init__(
        self,
        params,
        lr=0,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @paddle.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad
                if dp is None:
                    continue
                if p.ndim != 1:
                    dp = dp.add(p, alpha=g["weight_decay"])
                if p.ndim != 1:
                    param_norm = paddle.linalg.norm(x=p)
                    update_norm = paddle.linalg.norm(x=dp)
                    one = paddle.ones_like(x=param_norm)
                    q = paddle.where(
                        condition=param_norm > 0.0,
                        x=paddle.where(
                            condition=update_norm > 0,
                            x=g["eta"] * param_norm / update_norm,
                            y=one,
                        ),
                        y=one,
                    )
                    dp = dp.mul(q)
                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = paddle.zeros_like(x=p)
                mu = param_state["mu"]
                mu.multiply_(y=paddle.to_tensor(g["momentum"])).add_(
                    y=paddle.to_tensor(dp)
                )
                p.add_(y=paddle.to_tensor(-g["lr"] * mu))


class MultiCropWrapper(paddle.nn.Layer):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        backbone.fc, backbone.head = paddle.nn.Identity(), paddle.nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        idx_crops = paddle.cumsum(
            x=paddle.unique_consecutive(
                x=paddle.to_tensor(data=[inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            axis=0,
        )

        start_idx, output = 0, paddle.empty(shape=[0])
        for end_idx in idx_crops:
            _out = self.backbone(paddle.concat(x=x[start_idx:end_idx]))
            if isinstance(_out, tuple):
                _out = _out[0]
            output = paddle.concat(x=(output, _out))
            start_idx = end_idx
        return self.head(output)


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not not param.stop_gradient:
            continue
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]


def has_batchnorms(model):
    bn_types = (
        paddle.nn.BatchNorm1D,
        paddle.nn.BatchNorm2D,
        paddle.nn.BatchNorm3D,
        paddle.nn.SyncBatchNorm,
    )
    for name, module in model.named_sublayers():
        if isinstance(module, bn_types):
            return True
    return False


class PCA:
    """
    Class to  compute and apply PCA.
    """

    def __init__(self, dim=256, whit=0.5):
        self.dim = dim
        self.whit = whit
        self.mean = None

    def train_pca(self, cov):
        """
        Takes a covariance matrix (np.ndarray) as input.
        """
        d, v = np.linalg.eigh(cov)
        eps = d.max() * 1e-05
        n_0 = (d < eps).sum()
        if n_0 > 0:
            d[d < eps] = eps
        totenergy = d.sum()
        idx = np.argsort(d)[::-1][: self.dim]
        d = d[idx]
        v = v[:, idx]
        print("keeping %.2f %% of the energy" % (d.sum() / totenergy * 100.0))
        d = np.diag(1.0 / d**self.whit)
        self.dvt = np.dot(d, v.T)

    def apply(self, x):
        if isinstance(x, np.ndarray):
            if self.mean is not None:
                x -= self.mean
            return np.dot(self.dvt, x.T).T
        if "gpu" in str(x.place):
            if self.mean is not None:
                x -= paddle.to_tensor(data=self.mean, dtype="float32")
            x = x
            perm_5 = list(range(x.ndim))
            perm_5[0] = 1
            perm_5[1] = 0
            x = paddle.mm(
                input=paddle.to_tensor(data=self.dvt, dtype="float32"),
                mat2=x.transpose(perm=perm_5),
            )
            perm_6 = list(range(x.ndim))
            perm_6[0] = 1
            perm_6[1] = 0
            return x.transpose(perm=perm_6)
        if self.mean is not None:
            x -= paddle.to_tensor(data=self.mean, dtype="float32")
        x = x
        perm_7 = list(range(x.ndim))
        perm_7[0] = 1
        perm_7[1] = 0
        x = paddle.mm(
            input=paddle.to_tensor(data=self.dvt, dtype="float32"),
            mat2=x.transpose(perm=perm_7),
        )
        perm_8 = list(range(x.ndim))
        perm_8[0] = 1
        perm_8[1] = 0
        return x.transpose(perm=perm_8)


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    Returns
    -------
    ap    : average precision
    """
    nimgranks = len(ranks)
    ap = 0
    recall_step = 1.0 / nres
    for j in np.arange(nimgranks):
        rank = ranks[j]
        if rank == 0:
            precision_0 = 1.0
        else:
            precision_0 = float(j) / rank
        precision_1 = float(j + 1) / (rank + 1)
        ap += (precision_0 + precision_1) * recall_step / 2.0
    return ap


def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.
         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only
           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """
    map = 0.0
    nq = len(gnd)
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0
    for i in np.arange(nq):
        qgnd = np.array(gnd[i]["ok"])
        if qgnd.shape[0] == 0:
            aps[i] = float("nan")
            prs[i, :] = float("nan")
            nempty += 1
            continue
        try:
            qgndj = np.array(gnd[i]["junk"])
        except:
            qgndj = np.empty(0)
        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgndj)]
        k = 0
        ij = 0
        if len(junk):
            ip = 0
            while ip < len(pos):
                while ij < len(junk) and pos[ip] > junk[ij]:
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap
        pos += 1
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j])
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]
    map = map / (nq - nempty)
    pr = pr / (nq - nempty)
    return map, aps, pr, prs


def multi_scale(samples, model):
    v = None
    for s in [1, 1 / 2 ** (1 / 2), 1 / 2]:
        if s == 1:
            inp = samples.clone()
        else:
            inp = paddle.nn.functional.interpolate(
                x=samples, scale_factor=s, mode="bilinear", align_corners=False
            )
        feats = model(inp).clone()
        if v is None:
            v = feats
        else:
            v += feats
    v /= 3
    v /= v.norm()
    return v
