import paddle
from paddle import distributed as dist

import os
import sys
import pickle
import argparse
from PIL import Image, ImageFile
import numpy as np
import utils
import vision_transformer as vits
from eval_knn import extract_features


class OxfordParisDataset(paddle.io.Dataset):
    def __init__(self, dir_main, dataset, split, transform=None, imsize=None):
        if dataset not in ["roxford5k", "rparis6k"]:
            raise ValueError("Unknown dataset: {}!".format(dataset))
        gnd_fname = os.path.join(dir_main, dataset, "gnd_{}.pkl".format(dataset))
        with open(gnd_fname, "rb") as f:
            cfg = pickle.load(f)
        cfg["gnd_fname"] = gnd_fname
        cfg["ext"] = ".jpg"
        cfg["qext"] = ".jpg"
        cfg["dir_data"] = os.path.join(dir_main, dataset)
        cfg["dir_images"] = os.path.join(cfg["dir_data"], "jpg")
        cfg["n"] = len(cfg["imlist"])
        cfg["nq"] = len(cfg["qimlist"])
        cfg["im_fname"] = config_imname
        cfg["qim_fname"] = config_qimname
        cfg["dataset"] = dataset
        self.cfg = cfg
        self.samples = cfg["qimlist"] if split == "query" else cfg["imlist"]
        self.transform = transform
        self.imsize = imsize

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = os.path.join(self.cfg["dir_images"], self.samples[index] + ".jpg")
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        if self.imsize is not None:
            img.thumbnail((self.imsize, self.imsize), Image.ANTIALIAS)
        if self.transform is not None:
            img = self.transform(img)
        return img, index


def config_imname(cfg, i):
    return os.path.join(cfg["dir_images"], cfg["imlist"][i] + cfg["ext"])


def config_qimname(cfg, i):
    return os.path.join(cfg["dir_images"], cfg["qimlist"][i] + cfg["qext"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Image Retrieval on revisited Paris and Oxford")
    parser.add_argument(
        "--data_path", default="/path/to/revisited_paris_oxford/", type=str
    )
    parser.add_argument(
        "--dataset", default="roxford5k", type=str, choices=["roxford5k", "rparis6k"]
    )
    parser.add_argument("--multiscale", default=False, type=utils.bool_flag)
    parser.add_argument("--imsize", default=224, type=int, help="Image size")
    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="Path to pretrained weights to evaluate.",
    )
    parser.add_argument("--use_cuda", default=True, type=utils.bool_flag)
    parser.add_argument("--arch", default="vit_small", type=str, help="Architecture")
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )
    parser.add_argument(
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")',
    )
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="Please ignore and do not set this argument.",
    )
    args = parser.parse_args()
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    transform = paddle.vision.transforms.Compose(
        [
            paddle.vision.transforms.ToTensor(),
            paddle.vision.transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
        ]
    )
    dataset_train = OxfordParisDataset(
        args.data_path,
        args.dataset,
        split="train",
        transform=transform,
        imsize=args.imsize,
    )
    dataset_query = OxfordParisDataset(
        args.data_path,
        args.dataset,
        split="query",
        transform=transform,
        imsize=args.imsize,
    )
    sampler = paddle.io.DistributedBatchSampler(
        dataset=dataset_train, shuffle=False, batch_size=1
    )
    data_loader_train = paddle.io.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_query = paddle.io.DataLoader(
        dataset=dataset_query,
        batch_size=1,
        num_workers=args.num_workers,
        drop_last=False,
    )
    print(f"train: {len(dataset_train)} imgs / query: {len(dataset_query)} imgs")
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        # model = torch.hub.load('facebookresearch/xcit:main', args.arch,num_classes=0)
        pass
    elif args.arch in paddle.vision.models.__dict__.keys():
        model = paddle.vision.models.__dict__[args.arch](num_classes=0)
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    if args.use_cuda:
        model
    model.eval()
    if os.path.isfile(args.pretrained_weights):
        state_dict = paddle.load(path=args.pretrained_weights)
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.set_state_dict(state_dict=state_dict, use_structured_name=False)
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(
                args.pretrained_weights, msg
            )
        )
    elif args.arch == "vit_small" and args.patch_size == 16:
        print(
            "Since no pretrained weights have been provided, we load pretrained DINO weights on Google Landmark v2."
        )
    else:
        print("Warning: We use random weights.")
    train_features = extract_features(
        model, data_loader_train, args.use_cuda, multiscale=args.multiscale
    )
    query_features = extract_features(
        model, data_loader_query, args.use_cuda, multiscale=args.multiscale
    )
    if utils.get_rank() == 0:
        train_features = paddle.nn.functional.normalize(x=train_features, axis=1, p=2)
        query_features = paddle.nn.functional.normalize(x=query_features, axis=1, p=2)
        sim = paddle.mm(input=train_features, mat2=query_features.T)
        ranks = paddle.argsort(x=-sim, axis=0).cpu().numpy()
        gnd = dataset_train.cfg["gnd"]
        ks = [1, 5, 10]
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g["ok"] = np.concatenate([gnd[i]["easy"], gnd[i]["hard"]])
            g["junk"] = np.concatenate([gnd[i]["junk"]])
            gnd_t.append(g)
        mapM, apsM, mprM, prsM = utils.compute_map(ranks, gnd_t, ks)
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g["ok"] = np.concatenate([gnd[i]["hard"]])
            g["junk"] = np.concatenate([gnd[i]["junk"], gnd[i]["easy"]])
            gnd_t.append(g)
        mapH, apsH, mprH, prsH = utils.compute_map(ranks, gnd_t, ks)
        print(
            ">> {}: mAP M: {}, H: {}".format(
                args.dataset,
                np.around(mapM * 100, decimals=2),
                np.around(mapH * 100, decimals=2),
            )
        )
        print(
            ">> {}: mP@k{} M: {}, H: {}".format(
                args.dataset,
                np.array(ks),
                np.around(mprM * 100, decimals=2),
                np.around(mprH * 100, decimals=2),
            )
        )

    dist.barrier()
