import glob
import logging
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from bale.helpers.misc import load_pickle_from
from sklearn.model_selection import LeaveOneGroupOut, BaseCrossValidator, \
    StratifiedGroupKFold
from torch.utils.data import Dataset

from bale import TASK_TYPE_TO_TARGET
from bale.dataset.helpers import get_windowed_features_v2

_NEMO_TUNE_PARTICIPANTS: list[str] = ['105', '124', '108']
_AOMIC_TUNE_PARTICIPANTS: list[str] = ['47', '207', '109']
DATASET_NEMO_NAME = 'nemo'
DATASET_AOMIC_NAME = 'aomic'


@dataclass
class Indices:
    train_indices: list[list[int]] = field(default_factory=list)
    test_indices: list[list[int]] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)
    image_ids: list[list[int]] = field(default_factory=list)

    def fill(self,
             train: list[int],
             test: list[int],
             participant: str,
             image_id: list[int]) -> None:
        self.train_indices.append(train)
        self.test_indices.append(test)
        self.participants.append(participant)
        self.image_ids.append(image_id)


class CVStrategy:
    """
    General class for cross-validation strategy.
    """

    def __init__(self, strategy: BaseCrossValidator,
                 groups: list,
                 users: pd.Series,
                 y: list,
                 condition_groups: list):
        self.strategy = strategy
        self.groups = groups
        self.users = users
        self.y = y
        self.condition_groups = condition_groups

        self.train_indices_ = []
        self.test_indices_ = []
        self.test_users_ = []
        self.image_ids_ = []
        self.indices: Indices = Indices()
        self.folds = []

    def create_train_and_test_indices(self, dataset: str, tune: bool) -> None:
        """

        :return: Train and test indices, and users.
        """
        logging.info(f"Strategy: {self.strategy.__str__()}")
        for train_, test_ in self.strategy.split(
                X=list(range(len(self.y))),
                y=self.users.map(lambda x: str(x)) + [str(i) for i in self.y] if isinstance(self,
                                                                                            CVStrategyStratifiedGroupFoldNoImageRepeat) else self.y,
                groups=self.groups):
            if tune and isinstance(self, CVStrategyStratifiedGroupFoldNoImageRepeat):
                # Only 'tuned' participants must be in a test set
                for i in test_:
                    if dataset == DATASET_NEMO_NAME and str(self.users[i]) not in _NEMO_TUNE_PARTICIPANTS:
                        test_ = np.delete(test_, np.where(test_ == i))
                    if dataset == DATASET_AOMIC_NAME and str(self.users[i]) not in _AOMIC_TUNE_PARTICIPANTS:
                        test_ = np.delete(test_, np.where(test_ == i))
            logging.info("---Populating indices for the fold---")
            self.populate_indices(train_, test_)

        if tune and isinstance(self, CVStrategyStratifiedGroupFoldNoImageRepeat):
            t = set([str(self.users[i]) for i in [item for row in self.test_indices_ for item in row]])
            logging.info(f"Test participants: {t}")
            if dataset == DATASET_NEMO_NAME:
                assert len(t.difference(set(_NEMO_TUNE_PARTICIPANTS))) == 0
            if dataset == DATASET_AOMIC_NAME:
                assert len(t.difference(set(_AOMIC_TUNE_PARTICIPANTS))) == 0

        assert len(self.train_indices_) == len(self.test_indices_) == len(self.test_users_)

        for fold in range(len(self.test_indices_)):
            assert len(set([self.condition_groups[i] for i in
                            self.test_indices_[fold]]).intersection(
                set([self.condition_groups[i] for i in
                     self.train_indices_[fold]]))) == 0

        for id_, _ in enumerate(self.test_indices_):
            self.indices.fill(train=self.train_indices_[id_],
                              test=self.test_indices_[id_],
                              participant=self.test_users_[id_],
                              image_id=self.image_ids_[id_])
            if tune and isinstance(self, CVStrategyLOGOUserNoImageRepeat):
                if dataset == DATASET_NEMO_NAME and str(self.test_users_[id_]) in _NEMO_TUNE_PARTICIPANTS:
                    self.folds.append(id_)
                elif dataset == DATASET_AOMIC_NAME and str(self.test_users_[id_]) in _AOMIC_TUNE_PARTICIPANTS:
                    self.folds.append(id_)
                else:
                    logging.info(f"Skipping fold {id_}, as the participant is not in the tune set")
                    continue
            else:
                # if the strategy is "independent" we have to ensure that the "tuned" participant is not in the test set
                if isinstance(self, CVStrategyLOGOUserNoImageRepeat):
                    if dataset == DATASET_NEMO_NAME and str(self.test_users_[id_]) in _NEMO_TUNE_PARTICIPANTS:
                        logging.info(f"Skipping fold {id_}, as the participant is in the tune set")
                        continue
                    elif dataset == DATASET_AOMIC_NAME and str(self.test_users_[id_]) in _AOMIC_TUNE_PARTICIPANTS:
                        logging.info(f"Skipping fold {id_}, as the participant is in the tune set")
                        continue
                self.folds.append(id_)

        logging.info("Folds: {}".format(len(self.folds)))

    def populate_indices(self, train_indices: list[int], test_indices: list[int]) -> None:

        if isinstance(self, CVStrategyLOGOUserNoImageRepeat) and len(set(self.y)) > 2:
            seen_labels = []
            new_test_indices = []
            for i in test_indices:
                if self.y[i] not in seen_labels:
                    seen_labels.append(self.y[i])
                    new_test_indices.append(i)
            test_indices = new_test_indices
            test_images = [self.condition_groups[i] for i in test_indices]
            train_indices = [i for i in train_indices if self.condition_groups[i] not in test_images]

        test_images = [self.condition_groups[i] for i in test_indices]
        logging.info("Test images: {}".format(test_images))
        logging.info("Test labels: {}".format([self.y[i] for i in test_indices]))
        logging.info("Train images: {}".format([self.condition_groups[i] for i in train_indices]))
        logging.info("Number of train images: {}".format(len(train_indices)))
        new_train = [i for i in train_indices if
                     self.condition_groups[i] not in test_images]
        logging.info(
            "Train images after assuring correct split: {}".format([self.condition_groups[i] for i in new_train]))
        logging.info("Number of train images after assuring correct split: {}".format(len(new_train)))

        for i in set(self.y):
            logging.info("Number of samples in test set for class {} is {}".
                         format(i, np.sum(np.array(self.y)[test_indices] == i)))

        n_train = len(set([self.y[i] for i in new_train]))
        n_test = len(set([self.y[i] for i in test_indices]))
        assert n_train == n_test

        # Test users:
        test_users = "_".join(
            set([str(self.users[i]) for i in test_indices]))

        # Save indices:
        self.train_indices_.append(new_train)
        self.test_indices_.append(test_indices)
        self.test_users_.append(test_users)
        self.image_ids_.append(
            [self.condition_groups[i] for i in test_indices])


class CVStrategyLOGOUserNoImageRepeat(CVStrategy):
    def __init__(self,
                 groups: pd.Series,
                 users: pd.Series,
                 y: list,
                 condition_group: list,
                 strategy: BaseCrossValidator = LeaveOneGroupOut()):
        super(CVStrategyLOGOUserNoImageRepeat, self).__init__(strategy=strategy,
                                                              users=users,
                                                              groups=groups,
                                                              y=y,
                                                              condition_groups=condition_group)


class CVStrategyStratifiedGroupFoldNoImageRepeat(CVStrategy):
    def __init__(self,
                 groups: list,
                 users: pd.Series,
                 y: list,
                 condition_group: list,
                 splits: int):
        strategy = StratifiedGroupKFold(n_splits=splits)
        super(CVStrategyStratifiedGroupFoldNoImageRepeat, self).__init__(
            strategy=strategy,
            users=users,
            groups=groups,
            y=y,
            condition_groups=condition_group)


class BrainyDataset(Dataset):
    """
    General class for all brain datasets.
    """

    def __init__(self, data_cv_strategy: str,
                 features: list[torch.Tensor] | list[np.ndarray],
                 labels: list[int],
                 images_ids: list[int] | np.ndarray,
                 images: list[torch.Tensor],
                 user_ids: pd.Series,
                 num_classes: int, ):
        """

        :param data_cv_strategy:
        :param features:
        :param labels:
        :param images_ids:
        :param images:
        :param user_ids:
        :param num_classes:
        """

        # Must be set by classes that inherit this class:
        self.features: list[torch.Tensor] | list[np.ndarray] = features  # shape (n , chs, t)
        self.labels: list[int] = labels
        self.images_ids: list[int] = images_ids
        self.images: list[torch.Tensor] = images  # shape (1, z)
        self.user_ids: pd.Series = user_ids
        self.classes: int = num_classes

        if isinstance(self, FNIRSDataset):
            windows = 12  # each window captures 1 second
        # elif isinstance(self, EmotionDataset):
        #     windows = 4  # each window captures 0.1 second
        # elif isinstance(self, EEGAttractDataset):
        #     windows = 7  # each window captures 0.1 second
        elif isinstance(self, AOMICDataset):
            windows = None
        else:
            raise ValueError("Dataset is not recognised.")
        if windows is not None:
            self.features = get_windowed_features_v2(self.features,
                                                     windows=windows)
        else:
            self.features = np.stack(self.features, axis=0)

        logging.info("Samples in dataset: {}".format(len(self.labels)))
        self.cv_strategy: CVStrategy = self.get_cv_strategy(data_cv_strategy)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        y = torch.tensor(y, dtype=torch.float)
        image_ids = torch.tensor(self.images_ids[index], dtype=torch.long)
        user_ids = torch.tensor(self.user_ids[index], dtype=torch.long)

        return torch.FloatTensor(x), self.images[index], y, image_ids, user_ids

    def __len__(self):
        return len(self.labels)

    def get_cv_strategy(self, strategy: str) -> CVStrategy:
        cv_strategy = {
            'subject-independent': CVStrategyLOGOUserNoImageRepeat(
                groups=self.user_ids,
                users=self.user_ids,
                y=self.labels,
                condition_group=self.images_ids),
            'subject-dependent': CVStrategyStratifiedGroupFoldNoImageRepeat(
                splits=5,
                groups=self.images_ids,
                users=self.user_ids,
                y=self.labels,
                condition_group=self.images_ids),
        }

        return cv_strategy[strategy]


class FNIRSDataset(BrainyDataset):
    def __init__(self,
                 data_path,
                 tune: bool,
                 data_cv_strategy: str):

        self.data_path = data_path
        self.remove_tuned_participants = not tune
        self.X, self.y, self.channels, self.epoch_ids, self.metadata = self.load_dataset()

        self.images = []
        images_paths = glob.glob(self.data_path._str + '/images/*.*',
                                 recursive=True)
        image_names = dict(
            zip([int(i.split(".JPG")[0][-4:]) for i in images_paths],
                images_paths))

        self.images_ids = []
        for epoch_id in self.epoch_ids:
            key = self.metadata[self.metadata['epoch'] == epoch_id][
                "img_num"].item()
            value = image_names.get(key, None)
            if value is not None:
                self.images_ids.append(key)
                image_embeddings = torch.load(
                    self.data_path._str + "/images/embeddings/{}.pt".format(
                        key))
                self.images.append(image_embeddings)

        # Adjust the metadata:
        self.metadata = self.metadata[
            self.metadata['epoch'].isin(self.epoch_ids)]
        assert (self.metadata['epoch'] == self.epoch_ids).all() == True

        self.features, self.labels, self.user_ids = self.create_batch_dataset()

        super(FNIRSDataset, self).__init__(
            data_cv_strategy=data_cv_strategy,
            features=self.features,
            labels=self.labels,
            user_ids=pd.Series(self.user_ids),
            images=self.images,
            images_ids=self.images_ids,
            num_classes=2
        )

    def create_batch_dataset(self):
        data_list = [self.X[user] for user in self.X.keys()]
        user_ids_list = [[user] * self.X[user].shape[0] for user in
                         self.X.keys()]
        labels_list = [self.y[user] for user in self.y.keys()]

        batch_data = [d for sublist in data_list for d in sublist]
        user_ids = [int(d.split("-")[1]) for sublist in user_ids_list for d in
                    sublist]
        labels_data = [d for sublist in labels_list for d in sublist]

        labels_data = [TASK_TYPE_TO_TARGET[v] for v in labels_data]

        return batch_data, labels_data, user_ids

    def load_dataset(self):
        X = load_pickle_from(self.data_path / "X.pkl")
        y = load_pickle_from(self.data_path / "y.pkl")
        epoch_ids = load_pickle_from(self.data_path / "epoch_ids.pkl")
        fnirs_metadata = load_pickle_from(self.data_path / "metadata.pkl")

        if self.remove_tuned_participants:
            logging.info("Skipping participants that are in the tune set")
            # Remove data from X, y, epoch_ids if key in _NEMO_PARTICIPANTS
            X = {k: v for k, v in X.items() if k.split('-')[1] not in _NEMO_TUNE_PARTICIPANTS}
            y = {k: v for k, v in y.items() if k.split('-')[1] not in _NEMO_TUNE_PARTICIPANTS}
            epoch_ids = {k: v for k, v in epoch_ids.items() if k.split('-')[1] not in _NEMO_TUNE_PARTICIPANTS}
            fnirs_metadata['subject_temp'] = fnirs_metadata['subject'].str.replace("sub-", "")
            fnirs_metadata = fnirs_metadata[~fnirs_metadata['subject_temp'].isin(_NEMO_TUNE_PARTICIPANTS)]
            fnirs_metadata = fnirs_metadata.drop('subject_temp', axis=1).reset_index(drop=True)
        channels = load_pickle_from(self.data_path / "channels.pkl")
        hbo_channels = [label for idx, label in enumerate(channels) if
                        'hbo' in label]
        flat_epoch_ids = [item for sublist in
                          [epoch_ids[i] for i in epoch_ids.keys()]
                          for item in sublist]

        hbo_idx_chs = [idx for idx, label in enumerate(hbo_channels) if
                       'hbo' in label]
        for key in X.keys():
            X[key] = X[key][:, hbo_idx_chs, :]

        return X, y, hbo_channels, flat_epoch_ids, fnirs_metadata


class AOMICDataset(BrainyDataset):
    def __init__(self,
                 data_path,
                 tune: bool,
                 data_cv_strategy: str
                 ):

        self.data_path = data_path
        self.remove_tuned_participants = not tune
        self.epochs, self.metadata = self.load_dataset()

        self.images = []
        images_paths = glob.glob(self.data_path._str + '/images/embeddings/*.*',
                                 recursive=True)
        image_names = dict(
            zip([os.path.basename(i).split("-Face")[0] for i in images_paths],
                images_paths))

        self.images_ids = []
        for _, row in self.metadata.iterrows():
            image_id = row['ethnicity'] + '_' + row['ADFES_id'] + '-' + row[
                'trial_type'].capitalize()
            self.images_ids.append(image_id)
            self.images.append(torch.load(image_names.get(image_id)))

        self.images_ids = pd.factorize(self.images_ids)[0]
        self.features = self.epochs

        self.metadata['label'], _ = self.metadata['trial_type'].factorize()
        self.labels = self.metadata['label'].values.tolist()
        self.user_ids = self.metadata['subject_id'].map(
            lambda x: int(x.lstrip('sub-')))

        super(AOMICDataset, self).__init__(
            data_cv_strategy=data_cv_strategy,
            features=self.features,
            labels=self.labels,
            images=self.images,
            images_ids=self.images_ids,
            user_ids=self.user_ids,
            num_classes=5)

    def load_dataset(self):
        events = pd.read_csv(os.path.join(self.data_path, 'events.csv'))

        def custom_round(x):
            if x - math.floor(x) > 0.01:
                return math.ceil(x)
            else:
                return math.floor(x)

        events['frame_start'] = events['onset'].values / 0.75  # 0.75 is TR
        events['frame_start'] = events['frame_start'].apply(custom_round)
        events['frame_end'] = events[
                                  'frame_start'] + 8  # use 9 (9*0.75=6.75 seconds) volumes
        events['frame_start'] += 2  # Start from 1.5 seconds

        pattern = r'sub-(\d+)'
        epochs = []
        metadata = []

        recordings_dir = os.path.join(self.data_path, 'recordings')
        for filename in os.listdir(recordings_dir):
            if filename.endswith('.npy'):
                file_path = os.path.join(recordings_dir, filename)
                try:
                    # Recordings
                    data = np.load(file_path)

                    # Metadata
                    match = re.search(pattern, filename)
                    subject_id = match.group(0)
                    if self.remove_tuned_participants and str(int(
                            subject_id.replace("sub-", ""))) in _AOMIC_TUNE_PARTICIPANTS:
                        logging.info(f"Skipping participant {subject_id}")
                        continue
                    subject_metadata = events[
                        events['subject_id'] == subject_id]

                    # Skip the first three trials that have an identical stimulus. See the original publication for more details.
                    # The repetition of the same stimulus was done in order to make it possible to evaluate the possible effects of stimulus repetition.
                    subject_metadata = subject_metadata[3:]

                    for _, row in subject_metadata.iterrows():
                        start = int(row['frame_start']) - 1
                        end = int(row['frame_end'])
                        epoch = data[start:end, :]
                        epochs.append(epoch)
                    metadata.append(subject_metadata)

                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        metadata = pd.concat(metadata).reset_index(drop=True)

        return epochs, metadata
