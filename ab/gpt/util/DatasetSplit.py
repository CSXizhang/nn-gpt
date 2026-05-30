from __future__ import annotations

import random
from typing import Any, Optional


def normalize_split_protocol(raw: Any) -> str:
    normalized = str(raw or "official").strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if normalized in {"721", "7/2/1", "702010", "70/20/10"}:
        return "721"
    return "official"


def stratified_721_indices(targets: Any, *, seed: int = 42) -> tuple[list[int], list[int], list[int]]:
    by_class: dict[int, list[int]] = {}
    for index, target in enumerate(list(targets)):
        by_class.setdefault(int(target), []).append(int(index))

    rng = random.Random(int(seed))
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for label in sorted(by_class):
        indices = list(by_class[label])
        rng.shuffle(indices)
        n_total = len(indices)
        n_train = int(round(n_total * 0.70))
        n_val = int(round(n_total * 0.20))
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train : n_train + n_val])
        test_indices.extend(indices[n_train + n_val :])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def split_existing_dataset_721(
    train_source: Any,
    *,
    seed: int = 42,
    eval_source: Optional[Any] = None,
) -> dict[str, Any]:
    from torch.utils.data import Subset

    targets = getattr(train_source, "targets", None)
    if targets is None:
        targets = getattr(train_source, "labels", None)
    if targets is None:
        raise ValueError("Cannot build 7/2/1 split: dataset has no targets/labels attribute")

    train_indices, val_indices, test_indices = stratified_721_indices(
        targets,
        seed=int(seed),
    )
    eval_dataset = eval_source if eval_source is not None else train_source
    return {
        "protocol": "721",
        "seed": int(seed),
        "train": Subset(train_source, train_indices),
        "reward_eval": Subset(eval_dataset, val_indices),
        "heldout_test": Subset(eval_dataset, test_indices),
    }


def build_cifar10_split_datasets(
    *,
    root: str,
    train_transform: Any,
    eval_transform: Any,
    download: bool,
    protocol: str = "official",
    seed: int = 42,
) -> dict[str, Any]:
    from torchvision import datasets

    split_protocol = normalize_split_protocol(protocol)
    if split_protocol == "721":
        train_source = datasets.CIFAR10(
            root=root,
            train=True,
            download=download,
            transform=train_transform,
        )
        eval_source = datasets.CIFAR10(
            root=root,
            train=True,
            download=download,
            transform=eval_transform,
        )
        return split_existing_dataset_721(
            train_source,
            seed=int(seed),
            eval_source=eval_source,
        )

    return {
        "protocol": "official",
        "seed": int(seed),
        "train": datasets.CIFAR10(
            root=root,
            train=True,
            download=download,
            transform=train_transform,
        ),
        "reward_eval": datasets.CIFAR10(
            root=root,
            train=False,
            download=download,
            transform=eval_transform,
        ),
        "heldout_test": None,
    }
