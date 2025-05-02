import itertools
import logging
import random
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.backends import cudnn

_PERMUTATIONS = 10000


def is_in_string(candidates: list, target: str):
    """
    Checks if a {target} is in a list of strings {candidates}.
    :param candidates: A list of strings.
    :param target: A string.
    :return: True if a {target} in {candidates}, otherwise False.
    """
    found = False
    for v in candidates:
        if v in target:
            found = True
        else:
            found = False

        if found:
            return found

    return found


def set_seed(seed: int) -> None:
    """
    Sets seed for reproducible results.
    :param seed: int value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)
#
#
# def get_torch_generator(seed):
#     g = torch.Generator()
#     g.manual_seed(seed)
#     return g


def set_logging(log_dir: str) -> None:
    """
    Creates a logging file.
    :param log_dir: a logging directory
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("{}/logs.log".format(log_dir)),
            logging.StreamHandler()
        ]
    )

    logging.info("Logging directory is {}".format(log_dir))


def create_artificial_data(brain_input_dim: int, label: int,
                           noise_level: int = 0.3):
    brain_data = torch.empty(brain_input_dim)
    if label == 1:
        brain_data.normal_(mean=0.5, std=1)
    elif label == 0:
        brain_data.normal_(mean=-0.5, std=1)
    else:
        raise ValueError("Label {} is not recognized".format(label))

    # inject noise:
    random_indices = random.sample(range(1, brain_input_dim),
                                   int(brain_input_dim * noise_level))
    noise = np.random.uniform(-4, 4, len(random_indices))
    for (i, v) in zip(random_indices, noise):
        brain_data[i] = v

    return brain_data


def get_cartesian_product(values: list[list]) -> list[tuple]:
    return list(itertools.product(*values))


def _perm_fun(x: np.ndarray, n_test: int, n_control: int):
    n = n_test + n_control
    idx_control = set(random.sample(range(n), n_control))
    idx_test = set(range(n)) - idx_control
    return x[list(idx_test)].mean() - x[list(idx_control)].mean()


def run_permutation_test(a: pd.Series, b: pd.Series, sign: Literal['>', '<', 'either'] = ">"):
    observed_difference = np.mean(a) - np.mean(b)
    perm_diffs = [
        _perm_fun(np.concatenate((a, b), axis=None), a.shape[0], b.shape[0]) for
        _ in range(_PERMUTATIONS)]
    if sign == ">":
        p = np.mean([diff > observed_difference for diff in perm_diffs])
    if sign == "<":
        p = np.mean([diff < observed_difference for diff in perm_diffs])
    if sign == "either":
        p1 = np.mean([abs(diff) > abs(observed_difference) for diff in perm_diffs])
        p2 = np.mean([diff < observed_difference for diff in perm_diffs])
        p = p1 if p2 > p1 else p1

    if p < 0.01:
        v = "*"
    else:
        v = "ns"

    print(v)

    return v
