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


def train_model(MLP, train_loader, valid_loader, optimizer, num_epochs=5):
    """
    Train a model for several epochs.

    Arguments:
    - MLP (torch model): Model to train.
    - train_loader (torch dataloader): Dataloader to use to train the model.
    - valid_loader (torch dataloader): Dataloader to use to validate the model.
    - optimizer (torch optimizer): Optimizer to use to update the model.
    - num_epochs (int, optional): Number of epochs to train model.

    Returns:
    - results_dict (dict): Dictionary storing results across epochs on training
      and validation data.
    """

    results_dict = {
        "avg_train_losses": list(),
        "avg_valid_losses": list(),
        "avg_train_accuracies": list(),
        "avg_valid_accuracies": list(),
    }

    for e in tqdm(range(num_epochs)):
        no_train = True if e == 0 else False  # to get a baseline
        latest_epoch_results_dict = train_epoch(
            MLP, train_loader, valid_loader, optimizer=optimizer, no_train=no_train
        )

        for key, result in latest_epoch_results_dict.items():
            if key in results_dict.keys() and isinstance(results_dict[key], list):
                results_dict[key].append(latest_epoch_results_dict[key])
            else:
                results_dict[key] = result  # copy latest

    return results_dict


def train_epoch(MLP, train_loader, valid_loader, optimizer, no_train=False):
    """
    Train a model for one epoch.

    Arguments:
    - MLP (torch model): Model to train.
    - train_loader (torch dataloader): Dataloader to use to train the model.
    - valid_loader (torch dataloader): Dataloader to use to validate the model.
    - optimizer (torch optimizer): Optimizer to use to update the model.
    - no_train (bool, optional): If True, the model is not trained for the
      current epoch. Allows a baseline (chance) performance to be computed in the
      first epoch before training starts.

    Returns:
    - epoch_results_dict (dict): Dictionary storing epoch results on training
      and validation data.
    """

    criterion = torch.nn.NLLLoss()

    epoch_results_dict = dict()
    for dataset in ["train", "valid"]:
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
        update_results_by_class_in_place(
            y,
            y_pred.detach(),
            epoch_results_dict,
            dataset="train",
            num_classes=MLP.num_outputs,
        )
        optimizer.zero_grad()
        if not no_train:
            loss.backward()
            optimizer.step()

    num_items = len(train_loader.dataset)
    epoch_results_dict["avg_train_losses"] = np.sum(train_losses) / num_items
    epoch_results_dict["avg_train_accuracies"] = np.sum(train_acc) / num_items * 100

    MLP.eval()
    valid_losses, valid_acc = list(), list()
    with torch.no_grad():
        for X, y in valid_loader:
            y_pred = MLP(X)
            loss = criterion(torch.log(y_pred), y)
            acc = (torch.argmax(y_pred, axis=1) == y).sum() / len(y)
            valid_losses.append(loss.item() * len(y))
            valid_acc.append(acc.item() * len(y))
            update_results_by_class_in_place(
                y, y_pred.detach(), epoch_results_dict, dataset="valid"
            )

    num_items = len(valid_loader.dataset)
    epoch_results_dict["avg_valid_losses"] = np.sum(valid_losses) / num_items
    epoch_results_dict["avg_valid_accuracies"] = np.sum(valid_acc) / num_items * 100

    return epoch_results_dict


def update_results_by_class_in_place(
    y, y_pred, result_dict, dataset="train", num_classes=10
):
    """
    Updates results dictionary in place during a training epoch by adding data
    needed to compute the accuracies for each class.

    Arguments:
    - y (torch Tensor): target labels
    - y_pred (torch Tensor): predicted targets
    - result_dict (dict): Dictionary storing epoch results on training
      and validation data.
    - dataset (str, optional): Dataset for which results are being added.
    - num_classes (int, optional): Number of classes.
    """

    correct_by_class = None
    seen_by_class = None

    y_pred = np.argmax(y_pred, axis=1)
    if len(y) != len(y_pred):
        raise RuntimeError("Number of predictions does not match number of targets.")

    for i in result_dict[f"{dataset}_seen_by_class"].keys():
        idxs = np.where(y == int(i))[0]
        result_dict[f"{dataset}_seen_by_class"][int(i)] += len(idxs)

        num_correct = int(sum(y[idxs] == y_pred[idxs]))
        result_dict[f"{dataset}_correct_by_class"][int(i)] += num_correct


def train_model_extended(
    model_type="backprop",
    keep_num_classes="all",
    lr=1e-4,
    num_epochs=5,
    partial_backprop=False,
    num_hidden=100,
    bias=False,
    batch_size=32,
    plot_distribution=False,
):
    """
    Initializes model and optimizer, restricts datasets to specified classes and
    trains the model. Returns the trained model and results dictionary.

    Arguments:
    - model_type (str, optional): model to initialize ("backprop" or "Hebbian")
    - keep_num_classes (str or int, optional): number of classes to keep (from 0)
    - lr (float or list, optional): learning rate for both or each layer
    - num_epochs (int, optional): number of epochs to train model.
    - partial_backprop (bool, optional): if True, backprop is used to train the
      final Hebbian learning model.
    - num_hidden (int, optional): number of hidden units in the hidden layer
    - bias (bool, optional): if True, each linear layer will have biases in
        addition to weights.
    - batch_size (int, optional): batch size for dataloaders.
    - plot_distribution (bool, optional): if True, dataset class distributions
      are plotted.

    Returns:
    - MLP (torch module): Model
    - results_dict (dict): Dictionary storing results across epochs on training
      and validation data.
    """

    if isinstance(keep_num_classes, str):
        if keep_num_classes == "all":
            num_classes = 10
            use_train_set = train_set
            use_valid_set = valid_set
        else:
            raise ValueError("If 'keep_classes' is a string, it should be 'all'.")
    else:
        num_classes = int(keep_num_classes)
        use_train_set = restrict_classes(train_set, np.arange(keep_num_classes))
        use_valid_set = restrict_classes(valid_set, np.arange(keep_num_classes))

    if plot_distribution:
        plot_class_distribution(use_train_set, use_valid_set)

    train_loader = torch.utils.data.DataLoader(
        use_train_set, batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        use_valid_set, batch_size=batch_size, shuffle=False
    )

    model_params = {
        "num_hidden": num_hidden,
        "num_outputs": num_classes,
        "bias": bias,
    }

    if model_type.lower() == "backprop":
        Model = MultiLayerPerceptron
    elif model_type.lower() == "hebbian":
        if partial_backprop:
            Model = HebbianBackpropMultiLayerPerceptron
        else:
            Model = HebbianMultiLayerPerceptron
            model_params["clamp_output"] = True
    else:
        raise ValueError(
            f"Got {model_type} model type, but expected 'backprop' or 'hebbian'."
        )

    MLP = Model(**model_params)

    if isinstance(lr, list):
        if len(lr) != 2:
            raise ValueError("If 'lr' is a list, it must be of length 2.")
        optimizer = BasicOptimizer(
            [
                {"params": MLP.lin1.parameters(), "lr": lr[0]},
                {"params": MLP.lin2.parameters(), "lr": lr[1]},
            ]
        )
    else:
        optimizer = BasicOptimizer(MLP.parameters(), lr=lr)

    results_dict = train_model(
        MLP, train_loader, valid_loader, optimizer, num_epochs=num_epochs
    )

    return MLP, results_dict


# Initialize models
weight_perturb_mlp = WeightPerturbMLP(input_size, hidden_size, output_size)
node_perturb_mlp = NodePerturbMLP(input_size, hidden_size, output_size)
feedback_align_mlp = FeedbackAlignmentMLP(input_size, hidden_size, output_size)
kolen_pollack_mlp = KolenPollackMLP(input_size, hidden_size, output_size)

# Initialize DataLoader for test set
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False
)

# List of all MLP models
mlps = [weight_perturb_mlp, node_perturb_mlp, feedback_align_mlp, kolen_pollack_mlp]

# Compute losses
# losses = compute_losses_for_all_mlps(mlps, test_loader)
# print(losses)

if __name__ == "__main__":
    pass
