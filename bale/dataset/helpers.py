import copy
import os
import random
from enum import Enum
from typing import Optional

import mne
import numpy
import numpy as np
import pandas as pd
import torch
from mne_bids import BIDSPath, read_raw_bids
from nemo.utils import get_all_subjects, get_all_channels
from torch import Tensor
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from torchvision import transforms

from bale import EVENTS_TYPE

CHANNELS_3D_ORDERED = ["Empty", "S10_D7", "Empty", "S7_D7", "Empty", "Empty",
                       "S4_D2", "Empty", "S1_D2", "Empty",
                       "S10_D8", "Empty", "S8_D7", "Empty", "S7_D5", "S4_D4",
                       "Empty", "S3_D2", "Empty", "S1_D1",
                       "Empty", "S8_D8", "Empty", "S8_D5", "Empty", "Empty",
                       "S3_D4", "Empty", "S3_D1", "Empty",
                       "S9_D8", "Empty", "S8_D6", "Empty", "S6_D5", "S5_D4",
                       "Empty", "S3_D3", "Empty", "S2_D1",
                       "Empty", "S9_D6", "Empty", "S6_D6", "Empty", "Empty",
                       "S5_D3", "Empty", "S2_D3", "Empty"]

GRID = ["Empty", "Empty", "Empty", "Empty", "Empty", "Empty", "Empty", "Empty", "Empty",
        "Empty", "Empty", "Empty", "Empty", "Empty", "Empty", "Empty", "Empty", "Empty",
        "Empty", "S10_D7", "S7_D7", "Empty", "Empty", "Empty", "S4_D2", "S1_D2", "Empty",
        "Empty", "S10_D8", "S8_D7", "S7_D5", "Empty", "S4_D4", "S3_D2", "S1_D1", "Empty",
        "Empty", "S8_D8", "S8_D5", "Empty", "Empty", "Empty", "S3_D4", "S3_D1", "Empty",
        "Empty", "S9_D8", "S9_D8", "S6_D5", "Empty", "S5_D4", "S3_D3", "S2_D1", "Empty",
        "Empty", "S9_D6", "S6_D6", "Empty", "Empty", "Empty", "S5_D3", "S2_D3", "Empty",
        "Empty", "Empty", "Empty", "Empty", "Empty", "Empty", "Empty", "Empty", "Empty",
        "Empty", "Empty", "Empty", "Empty", "Empty", "Empty", "Empty", "Empty", "Empty",
        ]


class AttractTasks(str, Enum):
    TASK_1 = "task1"  # (1) Preferred (female, HI+LOW) vs non-preferred (male, HI+LOW)
    TASK_2 = "task2"  # (2) Preferred (male, HI+LOW) vs non-preferred (female, HI+LOW)
    TASK_3 = "task3"  # (3) Preferred (female, HI) vs non-preferred (male, LOW)
    TASK_4 = "task4"  # (4) Preferred (male, HI) vs non-preferred (female, LOW)
    TASK_5 = "task5"  # (5) Preferred (female) vs non-preferred (male)
    TASK_6 = "task6"  # (6) Preferred (male) vs non-preferred (female)
    TASK_7 = "task7"  # (7) Preferred: Female (attract, HI) vs Female (unattract, LOW)
    TASK_8 = "task8"  # (8) Preferred: Male (attract, HI) vs Male (unattract, LOW)
    TASK_34 = "task34"  # (34) Preferred (HI) vs non-preferred (LOW)
    TASK_ATTRACT = "attract"  # (attract) HI vs LOW

    def get_tasks(self, is_preferred: bool, is_female: bool):
        if is_preferred and is_female:
            return [self.TASK_1, self.TASK_2, self.TASK_3]
        elif not is_preferred and is_female:
            return [self.TASK_2, self.TASK_4, self.TASK_6]
        elif not is_preferred and not is_female:
            return [self.TASK_1, self.TASK_2, self.TASK_3]
        elif is_preferred and not is_female:
            return [self.TASK_2, self.TASK_4, self.TASK_6]


def get_windowed_features(features, windows, random_average=False):
    points_per_window = features.shape[1] // windows
    split_indices = [points_per_window * w_i for w_i in range(1, windows)]
    split_features = np.split(features, split_indices, axis=1)
    if random_average and points_per_window > 1:
        samples_for_average = int(points_per_window // 1.5)
    else:
        samples_for_average = points_per_window

    mean_features = [np.mean(i[:, np.random.choice(points_per_window,
                                                   size=samples_for_average,
                                                   replace=False)], axis=1) for
                     i in split_features]
    new_features = np.vstack(mean_features)
    new_features = new_features.T  # (channels, time)

    return new_features


def get_windowed_features_v2(features, windows: int = 10):
    data = copy.deepcopy(features)
    brain_data_groups = [np.array_split(i, windows, axis=-1) for i in data]
    brain_data_mean_groups = [np.mean(i, axis=-1) for sample in brain_data_groups for i in sample]
    brain_data_mean_groups = [brain_data_mean_groups[x:x + windows] for x in
                              range(0, len(brain_data_mean_groups), windows)]
    brain_data = [np.hstack(i).reshape(-1, windows) for i in brain_data_mean_groups]
    brain_data = np.stack(brain_data, axis=0)
    return brain_data

def shift_data(data: numpy.array, dim: int = 1, roll_value_min: int = -32,
               roll_value_max: int = 32):
    if roll_value_max == 0:
        roll_value_max = 1
    roll_value = np.random.randint(low=roll_value_min, high=roll_value_max)
    result = np.roll(data, shift=roll_value, axis=dim)
    return result


def intensify_data(data: numpy.array, factor=0.05, axis=(1, 2)):
    if random.random() > 0.5:
        average_intensity = np.mean(data, axis=axis)
        intensity = np.abs(average_intensity * random.uniform(0, factor))
        if random.random() > 0.5:
            intensified_dim1 = np.add(data[0, ...], intensity[0])
            intensified_dim2 = np.add(data[1, ...], intensity[1])
        else:
            intensified_dim1 = np.subtract(data[0, ...], intensity[0])
            intensified_dim2 = np.subtract(data[1, ...], intensity[1])
        data[0, ...] = intensified_dim1
        data[1, ...] = intensified_dim2

    return data


def select_channels_by_type(ch_values, channel_type):
    return dict((k, v) for k, v in ch_values.items() if channel_type in k)


def create_ordered_vector(ch_values, channel_type):
    return [ch_values.get(k + " " + channel_type, 0) for k in GRID]


def merge_two_dicts(a: dict, b: dict) -> dict:
    """
    A recursive method that stacks the values of two dicts.
    If the keys are not the same in both dicts, an error will be thrown.
    Only numpy arrays will be stacked.
    :param a:
    :param b:
    :return: A dict that contains the values from both dicts.
    """

    if not bool(a):
        return b
    if not bool(b):
        return a

    if len(set(a.keys()).intersection((b.keys()))) == 0:
        return a | b

    assert set(a.keys()) == set(b.keys())
    for (k1, v1), (k2, v2) in zip(a.items(), b.items()):
        assert k1 == k2
        if isinstance(a[k1], dict):
            merge_two_dicts(a[k1], b[k2])
        else:
            assert k1 == k2
            if isinstance(v1, np.ndarray):
                a[k1] = np.hstack((v1, v2)) if len(v1.shape) == 1 else np.vstack((v1, v2))
            else:
                assert v1 == v2
                a[k1] = v1

    return a


def fill_list_from_dict(a: dict, key_to_search: str, list_to_fill: []) -> None:
    """
    :param a:
    :param key_to_search
    :param list_to_fill
    :return:
    """
    for k, v in a.items():
        if isinstance(a[k], dict):
            fill_list_from_dict(a[k], key_to_search, list_to_fill)
        else:
            if k == key_to_search:
                if isinstance(a[k], np.ndarray):
                    list_to_fill.extend(a[k].tolist())
                else:
                    raise ValueError("Type {} is not recognized.".format(type(a[k])))


def get_loader(dataset: Dataset,
               sampler: Optional[SubsetRandomSampler],
               batch_size: int,
               num_workers: int,
               pin_memory: bool,
               drop_last_batch: bool = False,
               shuffle: bool = False,
               ) -> DataLoader:
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             collate_fn=collate_brainy_wrapper,
                             sampler=sampler,
                             drop_last=drop_last_batch,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return data_loader


def collate_brainy_wrapper(batch: list) -> (Tensor, Tensor, Tensor):
    brains, images, labels, img_ids, participants = zip(*batch)

    brains = torch.stack(brains, 0)
    images = torch.stack(images, 0)
    labels = torch.stack(labels, 0)
    img_ids = torch.stack(img_ids, 0)
    participants = torch.stack(participants, 0)

    return brains.float(), images.float(), labels.long(), img_ids.long(), participants.long()


CHANNELS = get_all_channels()


def get_epochs(epochs_path):
    epochs_dict = {}
    for subject in get_all_subjects():
        try:
            epochs = mne.read_epochs(os.path.join(epochs_path,
                                                  "{subject}_task-{events_type}_epo.fif".format(
                                                      subject=subject,
                                                      events_type=EVENTS_TYPE)),
                                     verbose="WARNING"
                                     )
            epochs_dict[subject] = epochs
        except FileNotFoundError:
            raise FileNotFoundError(
                "Data for the subject {} is not found".format(subject))
    return epochs_dict


def read_events_metadata(bids_path, subject, include_events):
    """
    Reads a subject's raw optical density data from BIDS and returns it as a dataframe.
    """
    subject_id = subject[-3:]

    sub_events_path = os.path.join(bids_path, "sub-{}".format(subject_id),
                                   'nirs', "sub-{}_task-{}_events.tsv".format(
            subject_id, include_events))

    return pd.read_csv(sub_events_path, sep="\t")


def read_raw_od(bids_path, subject, include_events):
    """
    Reads a subject's raw optical density data from BIDS and returns it as an MNE object.
    """
    subject_id = subject[-3:]
    bidspath = BIDSPath(
        subject=subject_id,
        task=include_events,
        root=bids_path,
        datatype="nirs",
    )
    # mne_bids does not currently support reading fnirs_od, so we have to manually set the channel types and ignore warnings
    with mne.utils.use_log_level("ERROR"):
        raw_od_bids = read_raw_bids(bidspath).load_data()
        ch_map = {ch: "fnirs_od" for ch in raw_od_bids.ch_names}
        raw_od_bids.set_channel_types(ch_map)
    return raw_od_bids
