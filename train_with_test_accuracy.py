import torchvision
import contextlib
import torch
import io
import numpy as np
from tqdm import tqdm

from models import (
    MultiLayerPerceptron,
    NodePerturbMLP,
    KolenPollackMLP,
    WeightPerturbMLP,
    FeedbackAlignmentMLP,
    HebbianBackpropMultiLayerPerceptron,
    HebbianMultiLayerPerceptron,
)
from visualization import plot_class_distribution
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# Define hyperparameters
BATCH_SIZE = 64
input_size = 10  # Example input size
hidden_size = 50  # Example hidden size
output_size = 1  # Example output size


def download_mnist(train_prop=0.8, keep_prop=0.5):

    valid_prop = 1 - train_prop

    discard_prop = 1 - keep_prop

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    with contextlib.redirect_stdout(io.StringIO()):  # to suppress output

        full_train_set = torchvision.datasets.MNIST(
            root="./data/", train=True, download=True, transform=transform
        )
        full_test_set = torchvision.datasets.MNIST(
            root="./data/", train=False, download=True, transform=transform
        )

    train_set, valid_set, _ = torch.utils.data.random_split(
        full_train_set, [train_prop * keep_prop, valid_prop * keep_prop, discard_prop]
    )
    test_set, _ = torch.utils.data.random_split(
        full_test_set, [keep_prop, discard_prop]
    )

    print("Number of examples retained:")
    print(f"  {len(train_set)} (training)")
    print(f"  {len(valid_set)} (validation)")
    print(f"  {len(test_set)} (test)")

    return train_set, valid_set, test_set


train_set, valid_set, test_set = download_mnist()

with contextlib.redirect_stdout(io.StringIO()):
    # Load the MNIST dataset, 50K training images, 10K validation, 10K testing
    train_set = datasets.MNIST(
        "./", transform=transforms.ToTensor(), train=True, download=True
    )
    test_set = datasets.MNIST(
        "./", transform=transforms.ToTensor(), train=False, download=True
    )

    rng_data = np.random.default_rng(seed=42)
    train_num = 50000
    shuffled_train_idx = rng_data.permutation(train_num)

    full_train_images = train_set.data.numpy().astype(float) / 255
    train_images = (
        full_train_images[shuffled_train_idx[:train_num]].reshape((-1, 784)).T.copy()
    )
    valid_images = (
        full_train_images[shuffled_train_idx[train_num:]].reshape((-1, 784)).T.copy()
    )
    test_images = (test_set.data.numpy().astype(float) / 255).reshape((-1, 784)).T

    full_train_labels = torch.nn.functional.one_hot(
        train_set.targets, num_classes=10
    ).numpy()
    train_labels = full_train_labels[shuffled_train_idx[:train_num]].T.copy()
    valid_labels = full_train_labels[shuffled_train_idx[train_num:]].T.copy()
    test_labels = (
        torch.nn.functional.one_hot(test_set.targets, num_classes=10).numpy().T
    )

    full_train_images = None
    full_train_labels = None
    train_set = None
    test_set = None


def restrict_classes(dataset, classes=[6], keep=True):
    """
    Removes or keeps specified classes in a dataset.

    Arguments:
    - dataset (torch dataset or subset): Dataset with class targets.
    - classes (list): List of classes to keep or remove.
    - keep (bool): If True, the classes specified are kept. If False, they are
    removed.

    Returns:
    - new_dataset (torch dataset or subset): Datset restricted as specified.
    """

    if hasattr(dataset, "dataset"):
        indices = np.asarray(dataset.indices)
        targets = dataset.dataset.targets[indices]
        dataset = dataset.dataset
    else:
        indices = np.arange(len(dataset))
        targets = dataset.targets

    specified_idxs = np.isin(targets, np.asarray(classes))
    if keep:
        retain_indices = indices[specified_idxs]
    else:
        retain_indices = indices[~specified_idxs]

    new_dataset = torch.utils.data.Subset(dataset, retain_indices)

    return new_dataset


class BasicOptimizer(torch.optim.Optimizer):
    """
    Simple optimizer class based on the SGD optimizer.
    """

    def __init__(self, params, lr=0.01, weight_decay=0):
        """
        Initializes a basic optimizer object.

        Arguments:
        - params (generator): Generator for torch model parameters.
        - lr (float, optional): Learning rate.
        - weight_decay (float, optional): Weight decay.
        """

        # Check parameter
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
        )

        super().__init__(params, defaults)

    def step(self):
        """
        Performs a single optimization step.
        """

        for group in self.param_groups:
            for p in group["params"]:

                # only update parameters with gradients
                if p.grad is not None:

                    # apply weight decay to gradient, if applicable
                    if group["weight_decay"] != 0:
                        p.grad = p.grad.add(p, alpha=group["weight_decay"])

                    # apply gradient-based update
                    p.data.add_(p.grad, alpha=-group["lr"])


def train_model(MLP, train_loader, valid_loader, test_loader, optimizer, num_epochs=5):
    """
    Train a model for several epochs.

    Arguments:
    - MLP (torch model): Model to train.
    - train_loader (torch dataloader): Dataloader to use to train the model.
    - valid_loader (torch dataloader): Dataloader to use to validate the model.
    - test_loader (torch dataloader): Dataloader to use to test the model.
    - optimizer (torch optimizer): Optimizer to use to update the model.
    - num_epochs (int, optional): Number of epochs to train model.

    Returns:
    - results_dict (dict): Dictionary storing results across epochs on training,
      validation, and test data.
    """

    results_dict = {
        "avg_train_losses": list(),
        "avg_valid_losses": list(),
        "avg_test_losses": list(),
        "avg_train_accuracies": list(),
        "avg_valid_accuracies": list(),
        "avg_test_accuracies": list(),
    }

    for e in tqdm(range(num_epochs)):
        no_train = True if e == 0 else False  # to get a baseline
        latest_epoch_results_dict = train_epoch(
            MLP, train_loader, valid_loader, test_loader, optimizer=optimizer, no_train=no_train
        )

        for key, result in latest_epoch_results_dict.items():
            if key in results_dict.keys() and isinstance(results_dict[key], list):
                results_dict[key].append(latest_epoch_results_dict[key])
            else:
                results_dict[key] = result  # copy latest

    return results_dict


def train_epoch(MLP, train_loader, valid_loader, test_loader, optimizer, no_train=False):
    """
    Train a model for one epoch.

    Arguments:
    - MLP (torch model): Model to train.
    - train_loader (torch dataloader): Dataloader to use to train the model.
    - valid_loader (torch dataloader): Dataloader to use to validate the model.
    - test_loader (torch dataloader): Dataloader to use to test the model.
    - optimizer (torch optimizer): Optimizer to use to update the model.
    - no_train (bool, optional): If True, the model is not trained for the
      current epoch. Allows a baseline (chance) performance to be computed in the
      first epoch before training starts.

    Returns:
    - epoch_results_dict (dict): Dictionary storing epoch results on training,
      validation, and test data.
    """

    criterion = torch.nn.NLLLoss()

    epoch_results_dict = dict()
    for dataset in ["train", "valid", "test"]:
        for sub_str in ["correct_by_class", "seen_by_class"]:
            epoch_results_dict[f"{dataset}_{sub_str}"] = {
                i: 0 for i in range(MLP.num_outputs)
            }

    MLP.train()
    train_losses, train_acc = list(), list()
    for X, y in train_loader:
        y_pred = MLP(X, y=y)
        loss = criterion(torch.log(y_pred), y)
        acc = (torch.argmax(y_pred.detach(), axis=1) == y).sum() / len(y)
        train_losses.append(loss.item() * len(y))
        train_acc.append(acc.item() * len(y))

        for label in range(MLP.num_outputs):
            y_label_idx = y == label
            epoch_results_dict["train_seen_by_class"][label] += y_label_idx.sum()
            epoch_results_dict["train_correct_by_class"][label] += (
                torch.argmax(y_pred.detach(), axis=1)[y_label_idx] == label
            ).sum()

        if not no_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    MLP.eval()
    valid_losses, valid_acc = list(), list()
    with torch.no_grad():
        for X, y in valid_loader:
            y_pred = MLP(X, y=y)
            loss = criterion(torch.log(y_pred), y)
            acc = (torch.argmax(y_pred.detach(), axis=1) == y).sum() / len(y)
            valid_losses.append(loss.item() * len(y))
            valid_acc.append(acc.item() * len(y))

            for label in range(MLP.num_outputs):
                y_label_idx = y == label
                epoch_results_dict["valid_seen_by_class"][label] += y_label_idx.sum()
                epoch_results_dict["valid_correct_by_class"][label] += (
                    torch.argmax(y_pred.detach(), axis=1) == label
                ).sum()

    test_losses, test_acc = list(), list()
    with torch.no_grad():
        for X, y in test_loader:
            y_pred = MLP(X, y=y)
            loss = criterion(torch.log(y_pred), y)
            acc = (torch.argmax(y_pred.detach(), axis=1) == y).sum() / len(y)
            test_losses.append(loss.item() * len(y))
            test_acc.append(acc.item() * len(y))

            for label in range(MLP.num_outputs):
                y_label_idx = y == label
                epoch_results_dict["test_seen_by_class"][label] += y_label_idx.sum()
                epoch_results_dict["test_correct_by_class"][label] += (
                    torch.argmax(y_pred.detach(), axis=1) == label
                ).sum()

    avg_train_loss = np.sum(train_losses) / np.sum(
        list(epoch_results_dict["train_seen_by_class"].values())
    )
    avg_valid_loss = np.sum(valid_losses) / np.sum(
        list(epoch_results_dict["valid_seen_by_class"].values())
    )
    avg_test_loss = np.sum(test_losses) / np.sum(
        list(epoch_results_dict["test_seen_by_class"].values())
    )
    avg_train_acc = np.sum(train_acc) / np.sum(
        list(epoch_results_dict["train_seen_by_class"].values())
    )
    avg_valid_acc = np.sum(valid_acc) / np.sum(
        list(epoch_results_dict["valid_seen_by_class"].values())
    )
    avg_test_acc = np.sum(test_acc) / np.sum(
        list(epoch_results_dict["test_seen_by_class"].values())
    )

    epoch_results_dict["avg_train_loss"] = avg_train_loss
    epoch_results_dict["avg_valid_loss"] = avg_valid_loss
    epoch_results_dict["avg_test_loss"] = avg_test_loss
    epoch_results_dict["avg_train_acc"] = avg_train_acc
    epoch_results_dict["avg_valid_acc"] = avg_valid_acc
    epoch_results_dict["avg_test_acc"] = avg_test_acc

    return epoch_results_dict
