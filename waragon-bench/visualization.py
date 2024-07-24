import numpy as np
import matplotlib.pyplot as plt


def get_plotting_color(dataset="train", model_idx=None):
    if model_idx is not None:
        dataset = None

    if model_idx == 0 or dataset == "train":
        color = "#1F77B4"  # blue
    elif model_idx == 1 or dataset == "valid":
        color = "#FF7F0E"  # orange
    elif model_idx == 2 or dataset == "test":
        color = "#2CA02C"  # green
    else:
        if model_idx is not None:
            raise NotImplementedError("Colors only implemented for up to 3 models.")
        else:
            raise NotImplementedError(
                f"{dataset} dataset not recognized. Expected 'train', 'valid' "
                "or 'test'."
            )

    return color


def plot_class_distribution(
    train_set, valid_set=None, test_set=None, num_classes=10, ax=None
):
    """
    Function for plotting the number of examples per class in each subset.

    Arguments:
    - train_set (torch dataset or torch dataset subset): training dataset
    - valid_set (torch dataset or torch dataset subset, optional): validation
      dataset
    - test_set (torch dataset or torch dataset subset, optional): test
      dataset
    - num_classes (int, optional): Number of classes in the data.
    - ax (plt subplot, optional): Axis on which to plot images. If None, a new
      axis will be created.

    Returns:
    - ax (plt subplot): Axis on which images were plotted.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))

    bins = np.arange(num_classes + 1) - 0.5

    for dataset_name, dataset in [
        ("train", train_set),
        ("valid", valid_set),
        ("test", test_set),
    ]:
        if dataset is None:
            continue

        if hasattr(dataset, "dataset"):
            targets = dataset.dataset.targets[dataset.indices]
        else:
            targets = dataset.targets

        outputs = ax.hist(
            targets,
            bins=bins,
            alpha=0.3,
            color=get_plotting_color(dataset_name),
            label=dataset_name,
        )

        per_class = len(targets) / num_classes
        ax.axhline(
            per_class, ls="dashed", color=get_plotting_color(dataset_name), alpha=0.8
        )

    ax.set_xticks(range(num_classes))
    ax.set_title("Counts per class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.legend(loc="center right")

    return ax


if __name__ == "__main__":
    pass
