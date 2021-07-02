import datetime
import os
import time

import torch
import torch.utils.data
from cvtk.losses.train_segmentation_with_bbox import (grid_loss, line_loss,
                                                      topk_loss)
from cvtk.models.segmentation import coco_utils, segmentation, utils
from references.segmentation.visualize import display_image


def evaluate(model, transform, data_loader, device, num_classes):
    print(data_loader.dataset.names)
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            output = model(image.to(device))

            _output = output["out"]
            _target = transform(_output, target,
                                topk=1, balance=False, eps=1e-5)

            confmat.update(_target.flatten(), _output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def test_one_epoch(model, data_loader, device, save_to=None):
    if save_to is None:
        save_to = "/workspace/results/0000"

    model.eval()
    utils.mkdir(save_to)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    names = data_loader.dataset.names[1:]
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            output = model(image.to(device))

            display_image(image, output["out"], target, save_to, names)

    return save_to


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = "Epoch: [{}]".format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        output = model(image.to(device))
        loss = criterion(output, target,
                         topk=None, balance=True, eps=1e-3)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(
            loss=loss.item(), lr=optimizer.param_groups[0]["lr"])


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    dataset = coco_utils.ToyDataset(args.data_path, args.train, max_size=args.max_size, single_cls=args.single_cls,
                                    crop_size=args.crop_size, phase="train")
    dataset_test = coco_utils.ToyDataset(args.data_path, args.val, max_size=args.max_size, single_cls=args.single_cls,
                                         crop_size=args.crop_size, phase="val")

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn, drop_last=True)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    num_classes = len(data_loader.dataset.names)
    model = segmentation.__dict__[args.model](pretrained=args.pretrained,
                                              num_classes=num_classes,
                                              aux_loss=args.aux_loss)

    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters()
                    if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters()
                    if p.requires_grad]},
    ]
    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters()
                  if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(
            checkpoint["model"], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    if args.test_only:
        save_to = os.path.join(args.output_dir, "images_test")
        save_to = test_one_epoch(
            model, data_loader_test, device=device, save_to=save_to)
        print(save_to)
        return

    transform = topk_loss.transform
    if args.loss_fn == "grid":
        criterion = grid_loss.criterion
    elif args.loss_fn == "line":
        criterion = line_loss.criterion
    elif args.loss_fn == "topk":
        criterion = topk_loss.criterion
    else:
        raise NotImplementedError(
            "loss {} is not supported".format(args.loss_fn))

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader,
                        lr_scheduler, device, epoch, args.print_freq)
        confmat = evaluate(model, transform, data_loader_test,
                           device=device, num_classes=num_classes)
        print(confmat)
        print(confmat.mat)
        utils.save_on_master(
            {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "names": data_loader.dataset.names,
                "epoch": epoch,
                "args": args
            },
            os.path.join(args.output_dir, "model_{}.pth".format(epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Segmentation Training")

    parser.add_argument("--data-path", default="/workspace/coco",
                        help="dataset path")
    parser.add_argument("--train", default="coco.json",
                        help="subset for train")
    parser.add_argument("--val", default="coco.json",
                        help="subset for val")
    parser.add_argument("--single-cls", action="store_true",
                        help="single")
    parser.add_argument("--max-size", default=512, type=int,
                        help="smallest max size")
    parser.add_argument("--crop-size", default=480, type=int,
                        help="train phase")

    parser.add_argument("--device", default="cuda",
                        help="device")
    parser.add_argument("--model", default="fcn_resnet50",
                        help="model")
    parser.add_argument("--aux-loss", action="store_true",
                        help="auxiliar loss")
    parser.add_argument("--loss-fn", default="topk",
                        help="specifying loss function")
    parser.add_argument("--epochs", default=30, type=int,
                        help="number of total epochs")
    parser.add_argument("-b", "--batch-size", default=8, type=int,
                        help="batch size for train")
    parser.add_argument("-j", "--workers", default=16, type=int,
                        help="data loading workers")
    parser.add_argument("--lr", default=0.01, type=float,
                        help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="momentum")
    parser.add_argument("--wd", default=1e-4, type=float, dest="weight_decay")

    parser.add_argument("--print-freq", default=10, type=int,
                        help="print frequency")
    parser.add_argument("--output-dir", default=".",
                        help="where to save")
    parser.add_argument("--resume", default="",
                        help="resume from checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int,
                        help="start epoch")
    parser.add_argument("--test-only", action="store_true",
                        help="only test the model")
    parser.add_argument("--pretrained", action="store_true",
                        help="pre-trained from modelzoo")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int,
                        help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://",
                        help="url used to set up distributed training")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
