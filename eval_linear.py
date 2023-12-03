import paddle
from paddle.vision import transforms

import sys
import os
import argparse
import json
from pathlib import Path
import utils
import vision_transformer as vits


def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (
            args.n_last_blocks + int(args.avgpool_patchtokens)
        )
    elif "xcit" in args.arch:
        # model = torch.hub.load('facebookresearch/xcit:main', args.arch,num_classes=0)
        pass
        embed_dim = model.embed_dim
    elif args.arch in paddle.vision.models.__dict__.keys():
        model = paddle.vision.models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = paddle.nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
    model
    model.eval()
    utils.load_pretrained_weights(
        model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size
    )
    print(f"Model {args.arch} built.")
    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier
    linear_classifier = paddle.DataParallel(layers=linear_classifier)
    val_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    dataset_val = paddle.vision.datasets.ImageFolder(
        os.path.join(args.data_path, "val"), transform=val_transform
    )
    val_loader = paddle.io.DataLoader(
        dataset=dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
    )
    if args.evaluate:
        utils.load_pretrained_linear_weights(
            linear_classifier, args.arch, args.patch_size
        )
        test_stats = validate_network(
            val_loader,
            model,
            linear_classifier,
            args.n_last_blocks,
            args.avgpool_patchtokens,
        )
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        return
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    dataset_train = paddle.vision.datasets.ImageFolder(
        os.path.join(args.data_path, "train"), transform=train_transform
    )
    sampler = paddle.io.DistributedBatchSampler(
        dataset=dataset_train, shuffle=True, batch_size=1
    )
    train_loader = paddle.io.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(
        f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs."
    )
    optimizer = paddle.optimizer.SGD(
        linear_classifier.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.0,
        momentum=0.9,
        weight_decay=0,
    )
    tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(
        T_max=args.epochs, eta_min=0, learning_rate=optimizer.get_lr()
    )
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr
    to_restore = {"epoch": 0, "best_acc": 0.0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        train_stats = train(
            model,
            linear_classifier,
            optimizer,
            train_loader,
            epoch,
            args.n_last_blocks,
            args.avgpool_patchtokens,
        )
        scheduler.step()
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(
                val_loader,
                model,
                linear_classifier,
                args.n_last_blocks,
                args.avgpool_patchtokens,
            )
            print(
                f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
            )
            best_acc = max(best_acc, test_stats["acc1"])
            print(f"Max accuracy so far: {best_acc:.2f}%")
            log_stats = {
                **{k: v for k, v in log_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
            }
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            paddle.save(
                obj=save_dict, path=os.path.join(args.output_dir, "checkpoint.pth.tar")
            )
    print(
        """Training of the supervised linear classifier on frozen features completed.
Top-1 test accuracy: {acc:.1f}""".format(
            acc=best_acc
        )
    )


def train(model, linear_classifier, optimizer, loader, epoch, n, avgpool):
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    for inp, target in metric_logger.log_every(loader, 20, header):
        inp = inp
        target = target
        with paddle.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = paddle.concat(
                    x=[x[:, 0] for x in intermediate_output], axis=-1
                )
                if avgpool:
                    output = paddle.concat(
                        x=(
                            output.unsqueeze(axis=-1),
                            paddle.mean(
                                x=intermediate_output[-1][:, 1:], axis=1
                            ).unsqueeze(axis=-1),
                        ),
                        axis=-1,
                    )
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        output = linear_classifier(output)
        loss = paddle.nn.CrossEntropyLoss()(output, target)
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        paddle.device.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@paddle.no_grad()
def validate_network(val_loader, model, linear_classifier, n, avgpool):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        inp = inp
        target = target
        with paddle.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = paddle.concat(
                    x=[x[:, 0] for x in intermediate_output], axis=-1
                )
                if avgpool:
                    output = paddle.concat(
                        x=(
                            output.unsqueeze(axis=-1),
                            paddle.mean(
                                x=intermediate_output[-1][:, 1:], axis=1
                            ).unsqueeze(axis=-1),
                        ),
                        axis=-1,
                    )
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        output = linear_classifier(output)
        loss = paddle.nn.CrossEntropyLoss()(output, target)
        if linear_classifier.module.num_labels >= 5:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        else:
            (acc1,) = utils.accuracy(output, target, topk=(1,))
        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        if linear_classifier.module.num_labels >= 5:
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    if linear_classifier.module.num_labels >= 5:
        print(
            "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
                top1=metric_logger.acc1,
                top5=metric_logger.acc5,
                losses=metric_logger.loss,
            )
        )
    else:
        print(
            "* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}".format(
                top1=metric_logger.acc1, losses=metric_logger.loss
            )
        )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LinearClassifier(paddle.nn.Layer):
    """Linear layer to train on top of frozen features"""

    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = paddle.nn.Linear(in_features=dim, out_features=num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.linear(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Evaluation with linear classification on ImageNet"
    )
    parser.add_argument(
        "--n_last_blocks",
        default=4,
        type=int,
        help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""",
    )
    parser.add_argument(
        "--avgpool_patchtokens",
        default=False,
        type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""",
    )
    parser.add_argument("--arch", default="vit_small", type=str, help="Architecture")
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )
    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="Path to pretrained weights to evaluate.",
    )
    parser.add_argument(
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")',
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of epochs of training."
    )
    parser.add_argument(
        "--lr",
        default=0.001,
        type=float,
        help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""",
    )
    parser.add_argument(
        "--batch_size_per_gpu", default=128, type=int, help="Per-GPU batch-size"
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
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--val_freq", default=1, type=int, help="Epoch frequency for validation."
    )
    parser.add_argument(
        "--output_dir", default=".", help="Path to save logs and checkpoints"
    )
    parser.add_argument(
        "--num_labels",
        default=1000,
        type=int,
        help="Number of labels for linear classifier",
    )
    parser.add_argument(
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    args = parser.parse_args()
    eval_linear(args)
