import paddle
from paddle.vision import transforms
from paddle.nn import functional as F
from paddle import distributed as dist

import os
import sys
import argparse
import utils
import vision_transformer as vits


def extract_feature_pipeline(args):
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    dataset_train = ReturnIndexDataset(
        os.path.join(args.data_path, "train"), transform=transform
    )
    dataset_val = ReturnIndexDataset(
        os.path.join(args.data_path, "val"), transform=transform
    )
    sampler = paddle.io.DistributedBatchSampler(
        dataset=dataset_train, shuffle=False, batch_size=1
    )
    data_loader_train = paddle.io.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = paddle.io.DataLoader(
        dataset=dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        drop_last=False,
    )
    print(
        f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs."
    )
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        #  model = torch.hub.load('facebookresearch/xcit:main', args.arch,num_classes=0)
        pass
    elif args.arch in paddle.vision.models.__dict__.keys():
        model = paddle.vision.models.__dict__[args.arch](num_classes=0)
        model.fc = paddle.nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model
    utils.load_pretrained_weights(
        model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size
    )
    model.eval()
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train, args.use_cuda)
    print("Extracting features for val set...")
    test_features = extract_features(model, data_loader_val, args.use_cuda)
    if utils.get_rank() == 0:
        train_features = F.normalize(x=train_features, axis=1, p=2)
        test_features = F.normalize(x=test_features, axis=1, p=2)
    train_labels = paddle.to_tensor(data=[s[-1] for s in dataset_train.samples]).astype(
        dtype="int64"
    )
    test_labels = paddle.to_tensor(data=[s[-1] for s in dataset_val.samples]).astype(
        dtype="int64"
    )
    if args.dump_features and dist.get_rank() == 0:
        paddle.save(
            obj=train_features.cpu(),
            path=os.path.join(args.dump_features, "trainfeat.pth"),
        )
        paddle.save(
            obj=test_features.cpu(),
            path=os.path.join(args.dump_features, "testfeat.pth"),
        )
        paddle.save(
            obj=train_labels.cpu(),
            path=os.path.join(args.dump_features, "trainlabels.pth"),
        )
        paddle.save(
            obj=test_labels.cpu(),
            path=os.path.join(args.dump_features, "testlabels.pth"),
        )
    return train_features, test_features, train_labels, test_labels


@paddle.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples
        index = index
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples).clone()
        if dist.get_rank() == 0 and features is None:
            features = paddle.zeros(shape=[len(data_loader.dataset), feats.shape[-1]])
            if use_cuda:
                features = features
            print(f"Storing features into tensor of shape {features.shape}")
            y_all = paddle.empty(
                shape=[dist.get_world_size(), index.shape[0]],
                dtype=index.dtype,
            )
        y_l = list(y_all.unbind(axis=0))
        y_all_reduce = dist.all_gather(tensor_list=y_l, tensor=index, sync_op=not True)
        y_all_reduce.wait()
        index_all = paddle.concat(x=y_l)
        feats_all = paddle.empty(
            shape=[dist.get_world_size(), feats.shape[0], feats.shape[1]],
            dtype=feats.dtype,
        )
        output_l = list(feats_all.unbind(axis=0))
        output_all_reduce = dist.all_gather(
            tensor_list=output_l, tensor=feats, sync_op=not True
        )
        output_all_reduce.wait()
        if dist.get_rank() == 0:
            if use_cuda:
                features.scatter_(index_all, paddle.concat(x=output_l))
            else:
                features.scatter_(index_all.cpu(), paddle.concat(x=output_l).cpu())
    return features


@paddle.no_grad()
def knn_classifier(
    train_features, train_labels, test_features, test_labels, k, T, num_classes=1000
):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = paddle.zeros(shape=[k, num_classes]).to(train_features.place)
    for idx in range(0, num_test_images, imgs_per_chunk):
        features = test_features[idx : min(idx + imgs_per_chunk, num_test_images), :]
        targets = test_labels[idx : min(idx + imgs_per_chunk, num_test_images)]
        batch_size = targets.shape[0]
        similarity = paddle.mm(input=features, mat2=train_features)
        distances, indices = similarity.topk(k=k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(shape=[batch_size, -1])
        retrieved_neighbors = paddle.take_along_axis(
            arr=candidates, axis=1, indices=indices
        )
        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.put_along_axis_(
            axis=1, indices=retrieved_neighbors.view(-1, 1), values=1
        )
        distances_transform = distances.clone().divide_(y=paddle.to_tensor(T)).exp_()
        probs = paddle.sum(
            x=paddle.multiply(
                x=retrieval_one_hot.view(batch_size, -1, num_classes),
                y=paddle.to_tensor(distances_transform.view(batch_size, -1, 1)),
            ),
            axis=1,
        )
        _, predictions = (
            paddle.sort(descending=True, x=probs, axis=1),
            paddle.argsort(descending=True, x=probs, axis=1),
        )
        correct = predictions.equal(y=targets.data.view(-1, 1))
        start_0 = correct.shape[1] + 0 if 0 < 0 else 0
        top1 = top1 + paddle.slice(correct, [1], [start_0], [start_0 + 1]).sum().item()
        start_1 = correct.shape[1] + 0 if 0 < 0 else 0
        top5 = (
            top5
            + paddle.slice(correct, [1], [start_1], [start_1 + min(5, k)]).sum().item()
        )
        total += targets.shape[0]
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


class ReturnIndexDataset(paddle.vision.datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation with weighted k-NN on ImageNet")
    parser.add_argument(
        "--batch_size_per_gpu", default=128, type=int, help="Per-GPU batch-size"
    )
    parser.add_argument(
        "--nb_knn",
        default=[10, 20, 100, 200],
        nargs="+",
        type=int,
        help="Number of NN to use. 20 is usually working the best.",
    )
    parser.add_argument(
        "--temperature",
        default=0.07,
        type=float,
        help="Temperature used in the voting coefficient",
    )
    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="Path to pretrained weights to evaluate.",
    )
    parser.add_argument(
        "--use_cuda",
        default=True,
        type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM",
    )
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
        "--dump_features",
        default=None,
        help="Path where to save computed features, empty for no saving",
    )
    parser.add_argument(
        "--load_features",
        default=None,
        help="""If the features have
        already been computed, where to find them.""",
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
    parser.add_argument("--data_path", default="/path/to/imagenet/", type=str)
    args = parser.parse_args()
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    if args.load_features:
        train_features = paddle.load(
            path=os.path.join(args.load_features, "trainfeat.pth")
        )
        test_features = paddle.load(
            path=os.path.join(args.load_features, "testfeat.pth")
        )
        train_labels = paddle.load(
            path=os.path.join(args.load_features, "trainlabels.pth")
        )
        test_labels = paddle.load(
            path=os.path.join(args.load_features, "testlabels.pth")
        )
    else:
        (
            train_features,
            test_features,
            train_labels,
            test_labels,
        ) = extract_feature_pipeline(args)
    if utils.get_rank() == 0:
        if args.use_cuda:
            train_features = train_features
            test_features = test_features
            train_labels = train_labels
            test_labels = test_labels
        print("Features are ready!\nStart the k-NN classification.")
        for k in args.nb_knn:
            top1, top5 = knn_classifier(
                train_features,
                train_labels,
                test_features,
                test_labels,
                k,
                args.temperature,
            )
            print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")
    dist.barrier()
