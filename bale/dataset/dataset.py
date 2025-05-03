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


# class EEGAttractDataset(BrainyDataset):
#     """
#     Holds EEG data for the Attractiveness dataset.
#     """
#
#     def __init__(self,
#                  data_path: str,
#                  config: dict):
#         self.task = "attract"
#         self.time_start = 0.25
#         self.time_end = 0.95
#         self.n_electrodes = 64
#         self.data_path = data_path
#
#         self.features, self.labels, self.user_ids, self.images_ids, self.metadata = self.create_batch_dataset()
#         self.images = self.load_images()
#         self.classes = 2
#
#         super(EEGAttractDataset, self).__init__(
#             data_cv_strategy=config["data_cv_strategy"])
#
#     def load_images(self) -> list[torch.Tensor]:
#         """
#         Loads images.
#         :return: A list of images.
#         """
#         id_mapping = pd.read_excel(self.data_path / "image_markers.xlsx",
#                                    header=0,
#                                    usecols=["imageID", "Marker"],
#                                    sheet_name="lowlevel")
#         images = []
#         for image_id in self.images_ids:
#             image_name = id_mapping[id_mapping["Marker"] == image_id][
#                 "imageID"].item()
#             image_embeddings = torch.load(
#                 self.data_path._str + "/images/embeddings/{}.pt".format(
#                     image_name.split('.')[0]))
#             images.append(image_embeddings)
#
#         return images
#
#     def select_time_range(self, data: dict) -> dict:
#         """
#         Select only a particular time range.
#         :param data: Data.
#         :return: A filtered data.
#         """
#         for k in data.keys():
#             time = data[k]["time"]
#             s = min(range(len(time)),
#                     key=lambda i: abs(time[i] - self.time_start))
#             e = min(range(len(time)),
#                     key=lambda i: abs(time[i] - self.time_end))
#             data[k]["features"] = data[k]["features"][..., s:e]
#             data[k]["time"] = data[k]["time"][s:e]
#
#         return data
#
#     def calculate_image_statistics(self,
#                                    metadata: pd.DataFrame) -> pd.DataFrame:
#         id_mapping = pd.read_excel(self.data_path / "image_markers.xlsx",
#                                    header=0,
#                                    usecols=["imageIDCorrected", "Marker"],
#                                    sheet_name="lowlevel")
#
#         high_cols = [col for col in metadata.columns if
#                      'img_idx_all_pp_HI' in col]
#         low_cols = [col for col in metadata.columns if
#                     'img_idx_all_pp_LO' in col]
#
#         def get_participant_images(d: pd.DataFrame, row_index: int) -> list[
#             int]:
#             return d.iloc[d.index == row_index].values.flatten().tolist()
#
#         def map_imgs_to_gender_attract(d: pd.DataFrame,
#                                        participant_imgs: list[int]) -> list[
#             str]:
#             return d.loc[d['Marker'].isin(participant_imgs)][
#                 "imageIDCorrected"].tolist()
#
#         def count_images(images: list[str], gender_filter: Literal['f', 'm']):
#             return len([i for i in images if gender_filter in i])
#
#         n_pref_high = list()
#         n_pref_low = list()
#         n_dispref_high = list()
#         n_dispref_low = list()
#
#         for participant in metadata.index:
#             high_values = get_participant_images(metadata[high_cols],
#                                                  participant)
#             low_values = get_participant_images(metadata[low_cols], participant)
#
#             high_gender = map_imgs_to_gender_attract(id_mapping, high_values)
#             low_gender = map_imgs_to_gender_attract(id_mapping, low_values)
#
#             pref_gender = metadata.loc[
#                 metadata.index == participant, "PreferredGender"].item()
#             dispref_gender = cast(Literal['f', 'm'],
#                                   'm' if pref_gender == "female" else 'f')
#
#             n_pref_high.append(count_images(high_gender, pref_gender[0]))
#             n_pref_low.append(count_images(low_gender, pref_gender[0]))
#             n_dispref_high.append(count_images(high_gender, dispref_gender))
#             n_dispref_low.append(count_images(low_gender, dispref_gender))
#
#         metadata["n_pref_high"] = n_pref_high
#         metadata["n_pref_low"] = n_pref_low
#         metadata["n_dispref_high"] = n_dispref_high
#         metadata["n_dispref_low"] = n_dispref_low
#
#         return metadata
#
#     def check_labels(self,
#                      labels: list[int],
#                      images: list[int],
#                      preferred_gender: dict[str],
#                      users: list[int]) -> list[int]:
#         """
#         img   1-40  female attractive
#         img  41-80  female unattractive
#         img  81-120 female intermediate
#         img 121-160 male attractive
#         img 161-200 male unattractive
#         img 201-240 male intermediate
#         :param labels: The real labels.
#         :param images: The indices of images.
#         :return:
#         """
#
#         indices_for_removal = list()
#
#         attractive_images = [*range(1, 41, 1)] + [*range(121, 161, 1)]
#         unattractive_images = [*range(41, 81, 1)] + [*range(161, 201, 1)]
#         female_intermediate = [*range(81, 121, 1)]
#         male_intermediate = [*range(201, 240, 1)]
#
#         def get_attractiveness_label(image):
#             label_ = ""
#             if image in attractive_images:
#                 label_ = "attractive"
#             elif image in unattractive_images:
#                 label_ = "unattractive"
#             elif image in female_intermediate:
#                 label_ = "female intermediate"
#             elif image in male_intermediate:
#                 label_ = "male intermediate"
#             return label_
#
#         current_user = ""
#         for idx, (label, image, user) in enumerate(zip(labels, images, users)):
#             user_pref = str(user) + " " + preferred_gender[user]
#             if current_user != user_pref:
#                 current_user = user_pref
#             warning_message = "({}) Label is {}, however image is {} ({})".format(
#                 current_user,
#                 label,
#                 get_attractiveness_label(
#                     image), image)
#             if label == 1:
#                 if image in unattractive_images or (
#                         image in female_intermediate and preferred_gender[
#                     user] == "male") or (
#                         image in male_intermediate and preferred_gender[
#                     user] == "female"):
#                     logging.warning(warning_message)
#                     indices_for_removal.append(idx)
#             elif label == 0:
#                 if image in attractive_images or (
#                         image in female_intermediate and preferred_gender[
#                     user] == "male") or (
#                         image in male_intermediate and preferred_gender[
#                     user] == "female"):
#                     logging.warning(warning_message)
#                     indices_for_removal.append(idx)
#             else:
#                 raise ValueError("Label {} is not recognized".format(label))
#
#         return indices_for_removal
#
#     def _load_high_low_data(self) -> (dict, dict, pd.DataFrame):
#         """
#         Loads raw matlab data.
#         :return: Dict (group1) containing the samples belonging to a positive, dict (group2)
#         containing samples of the negative class, and the metadata.
#         """
#
#         # Metadata:
#         high_demog = pd.read_csv(self.data_path / "Demographics_HiRatedIdx.csv")
#         low_demog = pd.read_csv(self.data_path / "Demographics_LoRatedIdx.csv")
#         metadata = pd.merge(high_demog, low_demog,
#                             on=["Subject", "Age", "OwnAttractivenessRating",
#                                 "Sex",
#                                 "SexualOrientation", "PreferredGender"])
#         metadata = self.calculate_image_statistics(metadata)
#
#         high_pref_X = self.load_matlab_data(
#             "high_ratings_per_pp_preferred_gender.mat")
#         high_disp_X = self.load_matlab_data(
#             "high_ratings_per_pp_dispreferred_gender.mat")
#         low_pref_X = self.load_matlab_data(
#             "low_ratings_per_pp_preferred_gender.mat")
#         low_disp_X = self.load_matlab_data(
#             "low_ratings_per_pp_dispreferred_gender.mat")
#
#         # Parsed data:
#         high_disp_X, high_pref_X, low_disp_X, low_pref_X = list(
#             map(self.parse_attract_data,
#                 [high_disp_X, high_pref_X, low_disp_X, low_pref_X],
#                 [False, True, False, True],
#                 [True, True, False, False],
#                 [metadata, metadata, metadata, metadata]))
#
#         # Features: dict {participant: EEG features}
#         if self.task in [AttractTasks.TASK_7.value, AttractTasks.TASK_8.value]:
#             group1 = high_pref_X
#             group2 = low_pref_X
#         elif self.task in [AttractTasks.TASK_ATTRACT.value]:
#             group1 = merge_two_dicts(high_pref_X, high_disp_X)
#             group2 = merge_two_dicts(low_pref_X, low_disp_X)
#         else:
#             group1 = merge_two_dicts(high_pref_X, low_pref_X)
#             group2 = merge_two_dicts(high_disp_X, low_disp_X)
#
#         return group1, group2, metadata
#
#     def _load_all_ratings_data(self):
#
#         # Metadata:
#         metadata = pd.read_csv(self.data_path / "Demographics_Idx.csv")
#
#         # Raw data:
#         pref_X = self.load_matlab_data(
#             "all_ratings_per_pp_preferred_gender.mat")
#         disp_X = self.load_matlab_data(
#             "all_ratings_per_pp_dispreferred_gender.mat")
#
#         # Parsed data:
#         disp_X, pref_X = list(
#             map(self.parse_attract_data, [disp_X, pref_X],
#                 [False, True], [None, None]))
#
#         return pref_X, disp_X, metadata
#
#     def load_dataset(self) -> (dict, dict, pd.DataFrame):
#         """
#         Loads a dataset from matlab files.
#         :return: Features, labels, and metadata.
#         """
#
#         if self.task not in [AttractTasks.TASK_5.value,
#                              AttractTasks.TASK_6.value]:
#             group1, group2, metadata = self._load_high_low_data()
#         else:
#             raise ValueError("Do not use the task {}".format(self.task))
#             # group1, group2, metadata = self._load_all_ratings_data()
#
#         logging.info("Checking the number of samples in two groups ...")
#
#         # Targets: dict {participant: targets}
#         group1_n_samples = self.create_targets(group1, target=1)
#         group2_n_samples = self.create_targets(group2, target=0)
#
#         X = merge_two_dicts(group1, group2)
#         y = merge_two_dicts(group1_n_samples, group2_n_samples)
#
#         # Filtering on time:
#         X = self.select_time_range(X)
#
#         return X, y, metadata
#
#     def create_targets(self, a: dict,
#                        target: Union[Literal[0], Literal[1]]) -> dict:
#         """
#         Creates labels for the targets.
#         :param a: dict containing features and image_ids.
#         :param target: a value for the target.
#         :return: A dictionary containing the labels.
#         """
#         targets = defaultdict()
#         for k, v in a.items():
#             if target == 0:
#                 targets[k] = np.zeros((v['image_ids'].shape[0]))
#             else:
#                 targets[k] = np.ones((v['image_ids'].shape[0]))
#
#         return targets
#
#     def _exclude_participants(self, is_preferred: bool,
#                               metadata: pd.DataFrame) -> list:
#         """
#         Marks the participants that should not be considered.
#         This selection is conditioned on the task itself.
#         :param is_preferred: A boolean flag to mark if data relates to a preferred or non-preferred
#         gender.
#         :return: A list of participants that should be discarded.
#         """
#         if self.task in [AttractTasks.TASK_1.value, AttractTasks.TASK_3.value,
#                          AttractTasks.TASK_5.value]:
#             preferred_gender = ["female"] if is_preferred else ["male"]
#         elif self.task in [AttractTasks.TASK_2.value, AttractTasks.TASK_4.value,
#                            AttractTasks.TASK_6.value]:
#             preferred_gender = ["male"] if is_preferred else ["female"]
#         elif self.task in [AttractTasks.TASK_7.value]:
#             preferred_gender = ["female"]
#         elif self.task in [AttractTasks.TASK_8.value]:
#             preferred_gender = ["male"]
#         elif self.task in [AttractTasks.TASK_34.value,
#                            AttractTasks.TASK_ATTRACT.value]:
#             preferred_gender = ["female", "male", "both"]
#         else:
#             raise ValueError("A task is not recognized")
#
#         participants = \
#             metadata[~metadata["PreferredGender"].isin(preferred_gender)][
#                 "Subject"].tolist()
#         participants = list(map(str, participants))
#
#         return participants
#
#     def parse_attract_data(self, data: np.ndarray,
#                            is_preferred: bool,
#                            is_high: bool,
#                            metadata: pd.DataFrame) -> dict:
#         """
#         Parses data into a dictionary having the 'features', 'image_ids', 'electrodes', and
#         'time' keys.
#         :param data: A Numpy array containing raw data.
#         :param is_preferred: A boolean flag to mark if data relates to a preferred or non-preferred
#         gender.
#         :param: is_high:
#         :return: Parsed data containing features, image ids, electrodes, and time.
#         """
#
#         attract_data = defaultdict()
#         if self.task in [AttractTasks.TASK_3.value, AttractTasks.TASK_4.value,
#                          AttractTasks.TASK_34.value] and (
#                 (is_preferred and not is_high) or (
#                 not is_preferred and is_high)):
#             logging.info(
#                 "Data having HI={} and Pref={} won't be considered for the task {}".format(
#                     is_high,
#                     is_preferred,
#                     self.task))
#         elif self.task in [AttractTasks.TASK_7.value,
#                            AttractTasks.TASK_8.value] and not is_preferred:
#             logging.info(
#                 "Data having Pref={} won't be considered for the task {}".format(
#                     is_preferred,
#                     self.task))
#         else:
#             excluded_participants = self._exclude_participants(is_preferred,
#                                                                metadata)
#             for participant_iter in range(len(data['subject'])):
#                 participant_id = str(
#                     int(data['subject'][participant_iter].item()))
#                 if participant_id in excluded_participants:
#                     continue
#                 else:
#                     features = np.stack(data['trial'][participant_iter])
#                     images = data['trialinfo'][participant_iter]
#                     electrodes = data['label'][participant_iter]
#                     electrodes = [electrodes[i][0] for i in
#                                   range(len(electrodes))]
#                     time = data["time"][participant_iter][0].tolist()
#                     attract_data[participant_id] = {'features': features,
#                                                     'image_ids': images,
#                                                     'electrodes': electrodes,
#                                                     'time': time}
#
#         return attract_data
#
#     def load_matlab_data(self, file_name: str, is_mat73: bool = True) -> dict:
#         """
#         Loads a .mat file.
#         :param file_name: The name of a file.
#         :param is_mat73: A boolean flag that states whether data is in matlab 7.3 format.
#         :return: A loaded object as a dict.
#         """
#         if is_mat73:
#             mat_file = mat73.loadmat(self.data_path / file_name)
#         else:
#             logging.warning("mat73 is not used!")
#             mat_file = sio.loadmat(self.data_path / file_name)
#         mat_data = mat_file[
#             file_name.split(".mat")[
#                 0]]  # get data by key (the file name without .mat)
#         return mat_data.squeeze() if not is_mat73 else mat_data
#
#     def create_batch_dataset(self) -> (list, list, list, list, pd.DataFrame):
#         """
#         Creates batch data by converting a dictionary to a list of values.
#         :return:
#         """
#
#         X, y, metadata = self.load_dataset()
#
#         data_list = [X[user]['features'] for user in X.keys()]
#         user_ids_list = [[user] * X[user]['features'].shape[0] for user in
#                          X.keys()]
#         image_ids_list = [X[user]['image_ids'] for user in X.keys()]
#         labels_list = [y[user] for user in y.keys()]
#
#         batch_data = [d for sublist in data_list for d in sublist]
#         image_ids = [int(d) for sublist in image_ids_list for d in sublist]
#         user_ids = [int(d) for sublist in user_ids_list for d in sublist]
#         labels_data = [int(d) for sublist in labels_list for d in sublist]
#
#         logging.info("Distribution of targets: {}".format(
#             sum([abs(y[user].sum() - (len(y[user]) - y[user].sum())) for user in
#                  y.keys()]) / len(
#                 y)))
#
#         indices_for_removal = self.check_labels(labels=labels_data,
#                                                 images=image_ids,
#                                                 preferred_gender=
#                                                 metadata.set_index('Subject')[
#                                                     'PreferredGender'].to_dict(),
#                                                 users=user_ids)
#
#         # batch_data = [batch_data[i] for i in range(len(batch_data)) if i not in indices_for_removal]
#         # labels_data = [labels_data[i] for i in range(len(labels_data)) if
#         #                i not in indices_for_removal]
#         # user_ids = [user_ids[i] for i in range(len(user_ids)) if i not in indices_for_removal]
#         # image_ids = [image_ids[i] for i in range(len(image_ids)) if i not in indices_for_removal]
#         #
#         # # Select only the images that are attractive or unattractive:
#         # attractive_images = [*range(1, 41, 1)] + [*range(121, 161, 1)]
#         # unattractive_images = [*range(41, 81, 1)] + [*range(161, 201, 1)]
#         # relevant_images = attractive_images + unattractive_images
#         # mask = [True if i in relevant_images else False for i in image_ids]
#         #
#         # batch_data = [batch_data[i] for i in range(len(mask)) if mask[i] is True]
#         # labels_data = [labels_data[i] for i in range(len(mask)) if mask[i] is True]
#         # user_ids = [user_ids[i] for i in range(len(mask)) if mask[i] is True]
#         # image_ids = [image_ids[i] for i in range(len(mask)) if mask[i] is True]
#         #
#         # # Set a target that depends on the image label:
#         # labels_data = [1 if i in attractive_images else 0 for i in image_ids]
#
#         return batch_data, labels_data, user_ids, image_ids, metadata
#
#     def select_electrodes_data(self, data: dict,
#                                electrodes: list = None) -> dict:
#         """
#         Selects only particular electrodes.
#         :param data: Data
#         :param electrodes: Electrodes.
#         :return: Data containing only the selected electrodes.
#         """
#         if electrodes is not None:
#             for participant in data.keys():
#                 electrodes_idx = [i for i, e in
#                                   enumerate(data[participant]['electrodes']) if
#                                   e in electrodes]
#                 data[participant]['features'] = data[participant]['features'][:,
#                                                 electrodes_idx, :]
#         return data
#

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
        participants_metadata = pd.read_csv(
            os.path.join(self.data_path, 'participants.tsv'), sep='\t'
        )

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

        # participants_metadata = participants_metadata[
        #     participants_metadata['participant_id'].isin(metadata['subject_id'].unique())]
        # participants_metadata['sex'].value_counts(dropna=False).sort_index()

        return epochs, metadata

# class EmotionDataset(BrainyDataset):
#     def __init__(self,
#                  data_path,
#                  config):
#
#         self.data_path = data_path
#         self.epochs, self.metadata = self.load_dataset()
#         self.binarise_data()
#
#         self.images = []
#         images_paths = glob.glob(self.data_path._str + '/images/embeddings/*.*',
#                                  recursive=True)
#         image_names = dict(
#             zip([os.path.basename(i) for i in images_paths], images_paths))
#
#         self.images_ids = []
#         for _, row in self.metadata.iterrows():
#             self.images_ids.append(row['image_id'])
#             self.images.append(torch.load(image_names.get(row['image'])))
#
#         self.features = self.epochs
#         self.labels = self.metadata['label'].values.tolist()
#         self.user_ids = self.metadata['participant'].values.tolist()
#
#         super(EmotionDataset, self).__init__(
#             data_cv_strategy=config["data_cv_strategy"])
#
#     def load_dataset(self):
#         epochs = []
#         metadata = []
#
#         recordings_dir = os.path.join(self.data_path, 'recordings')
#         for filename in os.listdir(recordings_dir):
#             if filename.endswith('.set'):
#                 file_path = os.path.join(recordings_dir, filename)
#
#                 try:
#                     epoched_data = mne.read_epochs_eeglab(
#                         file_path).apply_baseline((-0.2, 0))
#                     cropped_data = epoched_data.crop(tmin=0.4, tmax=0.8).pick(
#                         'eeg')
#                     eeg_data = cropped_data.get_data()
#
#                     # Metadata
#                     participant = [int(filename.split('ar.')[0][3:])] * len(
#                         eeg_data)
#                     events = cropped_data.events[:, 0]
#                     event_id = cropped_data.events[:, 2]
#
#                     # Image labels
#                     image_event = dict(
#                         (v, k) for k, v in cropped_data.event_id.items())
#                     pattern = re.compile(r"\((\d+)\)")
#                     label = [int(pattern.findall(image_event[id])[0][1]) for id
#                              in event_id]
#
#                     # Images
#                     pattern = re.compile(r"\((\d+)\)")
#                     image_id = [pattern.findall(image_event[id])[0] for id
#                                 in event_id]
#
#                     image = [self.get_image_filename(image_id[i]) for i in
#                              range(len(image_id))]
#
#                     data = {
#                         'participant': participant,
#                         'events': events,
#                         'label': label,
#                         'image_id': [int(i) for i in image_id],
#                         'image': image,
#                     }
#
#                     data = pd.DataFrame(data)
#
#                     metadata.append(data)
#                     epochs.append(eeg_data)
#
#                 except Exception as e:
#                     print(f"Error loading {filename}: {e}")
#
#         epochs = np.vstack(epochs)
#         metadata = pd.concat(metadata).reset_index(drop=True)
#
#         return epochs, metadata
#
#     def get_image_filename(self, image_id: str):
#
#         filename = ''
#         subject = image_id[0]
#         morphing_level = image_id[1]
#
#         if subject == '1':
#             filename += 'SF1-'
#         elif subject == '2':
#             filename += 'SF3-'
#         elif subject == '3':
#             filename += 'SM2-'
#         elif subject == '4':
#             filename += 'SM3-'
#         else:
#             raise ValueError('Invalid image id: {}'.format(image_id))
#
#         if morphing_level == '1':
#             filename += 'fe0-ha100'
#         elif morphing_level == '2':
#             filename += 'fe30-ha70'
#         elif morphing_level == '3':
#             filename += 'fe40-ha60'
#         elif morphing_level == '4':
#             filename += 'fe50-ha50'
#         elif morphing_level == '5':
#             filename += 'fe60-ha40'
#         elif morphing_level == '6':
#             filename += 'fe70-ha30'
#         elif morphing_level == '7':
#             filename += 'fe100-ha0'
#         else:
#             raise ValueError(
#                 'Invalid morphing level: {}'.format(morphing_level))
#
#         return filename + '.pt'
#
#     def binarise_data(self):
#         indices = self.metadata.index[
#             self.metadata['label'].isin([1, 7])].tolist()
#         self.metadata = self.metadata.iloc[indices].reset_index(drop=True)
#         self.metadata.loc[self.metadata["label"] == 1, "label"] = 1
#         self.metadata.loc[self.metadata["label"] == 7, "label"] = 0
#         self.epochs = self.epochs[indices]
