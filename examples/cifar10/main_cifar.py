# https://github.com/uoguelph-mlrg/Cutout
from __future__ import print_function
import argparse
from statistics import mean
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
import logging
import bagua.torch_api as bagua
import resnet
import resnet9
import dla
import matplotlib.pyplot as plt
import time
import csv
import pandas as pd

def train(args, model, train_loader, optimizer, epoch, test_loader):
    model.train()
    # loss_fn = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        if args.fuse_optimizer:
            optimizer.fuse_step()
        else:
            optimizer.step()
        if batch_idx % args.log_interval == 0:
            logging.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
        # test(model, test_loader)

def test(model, test_loader):
    model.eval()
    test_loss = 0
    test_losses = []
    correct = 0
    # loss_fn = F.cross_entropy()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            curr_loss = F.cross_entropy(output, target).item()
            test_loss += F.cross_entropy(
                output, target
            ).item()  # sum up batch loss
            test_losses.append(curr_loss)
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    avg_loss = mean(test_losses)

    logging.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            avg_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    accuracy = 100 * correct / len(test_loader.dataset)
    return accuracy, test_loss

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        metavar="N",
        help="number of epochs to train (default: 60)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="gradient_allreduce",
        help="gradient_allreduce, bytegrad, decentralized, low_precision_decentralized, qadam, async, adasum",
    )
    parser.add_argument(
        "--set-deterministic",
        action="store_true",
        default=False,
        help="set deterministic or not",
    )
    parser.add_argument(
        "--fuse-optimizer",
        action="store_true",
        default=False,
        help="fuse optimizer or not",
    )

    args = parser.parse_args()
    if args.set_deterministic:
        print("set_deterministic: True")
        np.random.seed(666)
        random.seed(666)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(666)
        torch.cuda.manual_seed_all(666 + int(bagua.get_rank()))
        torch.set_printoptions(precision=10)

    torch.cuda.set_device(bagua.get_local_rank())
    bagua.init_process_group()

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)
    if bagua.get_rank() == 0:
        logging.getLogger().setLevel(logging.INFO)

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if bagua.get_local_rank() == 0:
        dataset1 = datasets.CIFAR10(
            "../data", train=True, download=True, transform=transform_train
        )
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        dataset1 = datasets.CIFAR10(
            "../data", train=True, download=True, transform=transform_train
        )

    dataset2 = datasets.CIFAR10("../data", train=False, transform=transform_test)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset1, num_replicas=bagua.get_world_size(), rank=bagua.get_rank()
    )
    train_kwargs.update(
        {
            "sampler": train_sampler,
            "batch_size": args.batch_size // bagua.get_world_size(),
            "shuffle": False,
        }
    )
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = resnet.ResNet18().cuda()
    lr = 0.1

    optimizer = optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 35, 45], gamma=0.2)
    
    if args.algorithm == "gradient_allreduce":
        from bagua.torch_api.algorithms import gradient_allreduce

        algorithm = gradient_allreduce.GradientAllReduceAlgorithm()
    elif args.algorithm == "decentralized":
        from bagua.torch_api.algorithms import decentralized

        algorithm = decentralized.DecentralizedAlgorithm()
    elif args.algorithm == "low_precision_decentralized":
        from bagua.torch_api.algorithms import decentralized

        algorithm = decentralized.LowPrecisionDecentralizedAlgorithm()
    elif args.algorithm == "bytegrad":
        from bagua.torch_api.algorithms import bytegrad

        algorithm = bytegrad.ByteGradAlgorithm()
    elif args.algorithm == "qadam":
        from bagua.torch_api.algorithms import q_adam

        optimizer = q_adam.QAdamOptimizer(
            model.parameters(), lr=args.lr, warmup_steps=100
        )
        algorithm = q_adam.QAdamAlgorithm(optimizer)
    elif args.algorithm == "async":
        from bagua.torch_api.algorithms import async_model_average

        algorithm = async_model_average.AsyncModelAverageAlgorithm(
            sync_interval_ms=args.async_sync_interval,
        )
    # These lines added
    elif args.algorithm == "adasum":
        import adasum

        algorithm = adasum.AdasumAlgorithm(model.parameters())
    else:
        raise NotImplementedError

    model = model.with_bagua(
        [optimizer],
        algorithm,
        do_flatten=not args.fuse_optimizer,
    )

    if args.fuse_optimizer:
        optimizer = bagua.contrib.fuse_optimizer(optimizer)

    test_accuracies = []
    test_losses = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):

        train(args, model, train_loader, optimizer, epoch, test_loader)

        acc, loss = test(model, test_loader)
        scheduler.step()

        test_accuracies.append(acc)
        test_losses.append(loss)

    if args.save_model:
        torch.save(model.state_dict(), "cifar10_cnn.pt")

    end = time.time()
    time_ellapsed = end - start

    pd.DataFrame(np.array(test_accuracies)).to_csv(f"test_accuracy_{args.algorithm}_{bagua.get_world_size()}gpus.csv", index=True)
    pd.DataFrame(np.array(test_losses)).to_csv(f"test_error.csv_{args.algorithm}_{bagua.get_world_size()}gpus.csv", index=True)
    with open(f"test_accuracy_{args.algorithm}_{bagua.get_world_size()}gpus.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow([time_ellapsed])
        writer.writerow([args.batch_size])


if __name__ == "__main__":
    main()
