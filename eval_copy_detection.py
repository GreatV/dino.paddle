import paddle
from paddle.vision import transforms
from paddle.nn import functional as F
from paddle import distributed as dist

import os
import sys
import argparse
from PIL import Image
import numpy as np
import utils
import vision_transformer as vits
from eval_knn import extract_features


class CopydaysDataset:
    def __init__(self, basedir):
        self.basedir = basedir
        self.block_names = (
            ["original", "strong"]
            + [("jpegqual/%d" % i) for i in [3, 5, 8, 10, 15, 20, 30, 50, 75]]
            + [("crops/%d" % i) for i in [10, 15, 20, 30, 40, 50, 60, 70, 80]]
        )
        self.nblocks = len(self.block_names)
        self.query_blocks = range(self.nblocks)
        self.q_block_sizes = np.ones(self.nblocks, dtype=int) * 157
        self.q_block_sizes[1] = 229
        self.database_blocks = [0]

    def get_block(self, i):
        dirname = self.basedir + "/" + self.block_names[i]
        fnames = [
            (dirname + "/" + fname)
            for fname in sorted(os.listdir(dirname))
            if fname.endswith(".jpg")
        ]
        return fnames

    def get_block_filenames(self, subdir_name):
        dirname = self.basedir + "/" + subdir_name
        return [
            fname for fname in sorted(os.listdir(dirname)) if fname.endswith(".jpg")
        ]

    def eval_result(self, ids, distances):
        j0 = 0
        for i in range(self.nblocks):
            j1 = j0 + self.q_block_sizes[i]
            block_name = self.block_names[i]
            I = ids[j0:j1]
            sum_AP = 0
            if block_name != "strong":
                positives_per_query = [[i] for i in range(j1 - j0)]
            else:
                originals = self.get_block_filenames("original")
                strongs = self.get_block_filenames("strong")
                positives_per_query = [
                    [j for j, bname in enumerate(originals) if bname[:4] == qname[:4]]
                    for qname in strongs
                ]
            for qno, Iline in enumerate(I):
                positives = positives_per_query[qno]
                ranks = []
                for rank, bno in enumerate(Iline):
                    if bno in positives:
                        ranks.append(rank)
                sum_AP += score_ap_from_ranks_1(ranks, len(positives))
            print("eval on %s mAP=%.3f" % (block_name, sum_AP / (j1 - j0)))
            j0 = j1


def score_ap_from_ranks_1(ranks, nres):
    """Compute the average precision of one search.
    ranks = ordered list of ranks of true positives
    nres  = total number of positives in dataset
    """
    ap = 0.0
    recall_step = 1.0 / nres
    for ntp, rank in enumerate(ranks):
        if rank == 0:
            precision_0 = 1.0
        else:
            precision_0 = ntp / float(rank)
        precision_1 = (ntp + 1) / float(rank + 1)
        ap += (precision_1 + precision_0) * recall_step / 2.0
    return ap


class ImgListDataset(paddle.io.Dataset):
    def __init__(self, img_list, transform=None):
        self.samples = img_list
        self.transform = transform

    def __getitem__(self, i):
        with open(self.samples[i], "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, i

    def __len__(self):
        return len(self.samples)


def is_image_file(s):
    ext = s.split(".")[-1]
    if ext in ["jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp"]:
        return True
    return False


@paddle.no_grad()
def extract_features(image_list, model, args):
    transform = transforms.Compose(
        [
            transforms.Resize((args.imsize, args.imsize), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    tempdataset = ImgListDataset(image_list, transform=transform)
    data_loader = paddle.io.DataLoader(
        tempdataset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        drop_last=False,
        sampler=paddle.io.DistributedBatchSampler(
            dataset=tempdataset, shuffle=False, batch_size=1
        ),
    )
    features = None
    for samples, index in utils.MetricLogger(delimiter="  ").log_every(data_loader, 10):
        samples, index = samples, index
        feats = model.get_intermediate_layers(samples, n=1)[0].clone()
        cls_output_token = feats[:, 0, :]
        b, h, w, d = (
            len(samples),
            int(samples.shape[-2] / model.patch_embed.patch_size),
            int(samples.shape[-1] / model.patch_embed.patch_size),
            feats.shape[-1],
        )
        feats = feats[:, 1:, :].reshape(b, h, w, d)
        feats = feats.clip(min=1e-06).transpose(perm=[0, 3, 1, 2])
        feats = (
            F.avg_pool2d(kernel_size=(h, w), x=feats.pow(y=4), exclusive=False)
            .pow(y=1.0 / 4)
            .reshape(b, -1)
        )
        feats = paddle.concat(x=(cls_output_token, feats), axis=1)
        if dist.get_rank() == 0 and features is None:
            features = paddle.zeros(shape=[len(data_loader.dataset), feats.shape[-1]])
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
            if args.use_cuda:
                features.scatter_(index_all, paddle.concat(x=output_l))
            else:
                features.scatter_(index_all.cpu(), paddle.concat(x=output_l).cpu())
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Copy detection on Copydays")
    parser.add_argument(
        "--data_path",
        default="/path/to/copydays/",
        type=str,
        help="See https://lear.inrialpes.fr/~jegou/data.php#copydays",
    )
    parser.add_argument(
        "--whitening_path",
        default="/path/to/whitening_data/",
        type=str,
        help="""Path to directory with images used for computing the whitening operator.
        In our paper, we use 20k random images from YFCC100M.""",
    )
    parser.add_argument(
        "--distractors_path",
        default="/path/to/distractors/",
        type=str,
        help="Path to directory with distractors images. In our paper, we use 10k random images from YFCC100M.",
    )
    parser.add_argument(
        "--imsize", default=320, type=int, help="Image size (square image)"
    )
    parser.add_argument(
        "--batch_size_per_gpu", default=16, type=int, help="Per-GPU batch-size"
    )
    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="Path to pretrained weights to evaluate.",
    )
    parser.add_argument("--use_cuda", default=True, type=utils.bool_flag)
    parser.add_argument("--arch", default="vit_base", type=str, help="Architecture")
    parser.add_argument(
        "--patch_size", default=8, type=int, help="Patch resolution of the model."
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
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.eval()
    utils.load_pretrained_weights(
        model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size
    )
    dataset = CopydaysDataset(args.data_path)
    queries = []
    for q in dataset.query_blocks:
        queries.append(extract_features(dataset.get_block(q), model, args))
    if utils.get_rank() == 0:
        queries = paddle.concat(x=queries)
        print(f"Extraction of queries features done. Shape: {queries.shape}")
    database = []
    for b in dataset.database_blocks:
        database.append(extract_features(dataset.get_block(b), model, args))
    if os.path.isdir(args.distractors_path):
        print("Using distractors...")
        list_distractors = [
            os.path.join(args.distractors_path, s)
            for s in os.listdir(args.distractors_path)
            if is_image_file(s)
        ]
        database.append(extract_features(list_distractors, model, args))
    if utils.get_rank() == 0:
        database = paddle.concat(x=database)
        print(
            f"Extraction of database and distractors features done. Shape: {database.shape}"
        )
    if os.path.isdir(args.whitening_path):
        print(
            f"Extracting features on images from {args.whitening_path} for learning the whitening operator."
        )
        list_whit = [
            os.path.join(args.whitening_path, s)
            for s in os.listdir(args.whitening_path)
            if is_image_file(s)
        ]
        features_for_whitening = extract_features(list_whit, model, args)
        if utils.get_rank() == 0:
            mean_feature = paddle.mean(x=features_for_whitening, axis=0)
            database -= mean_feature
            queries -= mean_feature
            pca = utils.PCA(dim=database.shape[-1], whit=0.5)
            cov = (
                paddle.mm(input=features_for_whitening.T, mat2=features_for_whitening)
                / features_for_whitening.shape[0]
            )
            pca.train_pca(cov.cpu().numpy())
            database = pca.apply(database)
            queries = pca.apply(queries)
    if utils.get_rank() == 0:
        database = F.normalize(x=database, axis=1, p=2)
        queries = F.normalize(x=queries, axis=1, p=2)
        similarity = paddle.mm(input=queries, mat2=database.T)
        distances, indices = similarity.topk(k=20, largest=True, sorted=True)
        retrieved = dataset.eval_result(indices, distances)

    dist.barrier()
