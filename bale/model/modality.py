import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Union, Literal

import numpy as np
import torch
from torch import nn, Tensor

from bale import DEVICE


class BrainNetwork(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 h,
                 n_blocks=4,
                 norm_type='ln',
                 act_first=False):
        super().__init__()
        self.out_dim = out_dim
        self.n_blocks = n_blocks

        norm_func = partial(nn.BatchNorm1d,
                            num_features=h) if norm_type == 'bn' else partial(
            nn.LayerNorm, normalized_shape=h)
        act_fn = partial(nn.ReLU,
                         inplace=True) if norm_type == 'bn' else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)
        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h),
            *[item() for item in act_and_norm],
            nn.Dropout(0.5),
        )
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                *[item() for item in act_and_norm],
                nn.Dropout(0.15)
            ) for _ in range(n_blocks)
        ])

        self.projector = nn.Sequential(
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, self.out_dim),
        )

        self.lin0 = self.lin0.to(DEVICE)
        self.mlp = self.mlp.to(DEVICE)
        self.projector = self.projector.to(DEVICE)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)  # (batch_size, brain_signals)
        x = self.lin0(x)  # bs, h
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x

        return self.projector(x)


class BrainMLPEncoder(nn.Module):
    """
    Brain encoder/classifier that is based on ATCNet.
    """

    def __init__(self,
                 in_dim: int,
                 identifier: str,
                 projection_dim: int,
                 mlp_hidden_size: int,
                 ) -> None:
        super(BrainMLPEncoder, self).__init__()

        # Image encoder:
        self.encoder = BrainNetwork(in_dim=in_dim,
                                    h=mlp_hidden_size,
                                    out_dim=projection_dim)
        self.feature_dim = 1024
        self.identifier: str = identifier

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        return x


class CLIPImageEncoder(nn.Module):
    def __init__(self, identifier: str, projection_dim: int) -> None:
        super(CLIPImageEncoder, self).__init__()
        self.identifier: str = identifier
        self.feature_dim = 1024
        self.projection_dim = projection_dim

        self.projector = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, projection_dim),
        )

        self.projector = self.projector.to(DEVICE)

    def forward(self, x: Tensor) -> Tensor:
        # x = x.to(DEVICE)
        # self.projector = self.projector.to(DEVICE)
        return self.projector(x)


class BrainMLPClassifier(nn.Module):
    def __init__(self,
                 in_dim: int,
                 identifier: str,
                 projection_dim: int,
                 classes: int,
                 mlp_hidden_size: int) -> None:
        super(BrainMLPClassifier, self).__init__()
        self.encoder = BrainNetwork(in_dim=in_dim,
                                    h=mlp_hidden_size,
                                    out_dim=projection_dim)
        self.identifier: str = identifier
        self.linear = nn.Linear(self.encoder.out_dim, classes, bias=True)

        self.linear = self.linear.to(DEVICE)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(self.encoder(x))

        return x

    def predict_each_class(self, x: Tensor) -> (list[np.ndarray[int]], Tensor):
        logits = self(x)

        probabilities = torch.softmax(logits, dim=1)
        _, predicted_class = torch.max(probabilities, 1)
        predicted_class = predicted_class.to('cpu').detach().numpy()

        return predicted_class, probabilities


# class BrainLRClassifier(ModalityClassifier):
#     def __init__(self, seed: int):
#         super(BrainLRClassifier, self).__init__()
#         self.model = LogisticRegression(random_state=seed, multi_class="ovr",
#                                         max_iter=1000)  # TODO: model
#         self.labeller = MultiLabelBinarizer()
#         self.scaler = StandardScaler()
#
#     def fit(self,
#             dataset: BrainyDataset,
#             train_idx: list[int]):
#         data = self.prepare_data(dataset)
#         # data_scaled = self.scaler.fit_transform(data[train_idx])
#
#         target = [dataset.labels[i] for i in train_idx]
#         self.model.fit(data[train_idx], target)
#         self.labeller.fit(np.expand_dims(target, axis=1))
#
#     def test(self,
#              dataset: BrainyDataset,
#              test_idx: list[int]):
#         data = self.prepare_data(dataset)[test_idx]
#         # data_scaled = self.scaler.transform(data[test_idx])
#         y = [dataset.labels[i] for i in test_idx]
#
#         if len(set(y)) > 2:
#             y_hat = [y_pred for y_pred in self.model.predict(data)]
#             y_hat = self.labeller.transform(np.expand_dims(y_hat, axis=1))
#             roc_auc = roc_auc_score(y, y_hat, multi_class='ovr')
#         else:
#             y_hat = self.model.predict_proba(data)[:, 1]
#             roc_auc = roc_auc_score(y, y_hat)
#
#         return roc_auc
#         # wandb.log(dict(supervised_brain_test_rocauc=roc_auc))
#
#     @staticmethod
#     def prepare_data(dataset: BrainyDataset):
#         return dataset.features.reshape(dataset.features.shape[0], -1)
#
#     def predict(self, x: Tensor) -> list[float]:
#         x = x.reshape(x.shape[0], -1)
#
#         predictions = [y_pred for y_pred in self.model.predict(x.cpu())]
#
#         # predictions = self.model.predict_proba(x.cpu())[:, 1].tolist()
#
#         return predictions
#
#     def predict_each_class(self, x: Tensor) -> list[float]:
#         x = x.reshape(x.shape[0], -1)
#
#         predictions = self.model.predict_proba(x.cpu())
#         prob0 = predictions[:, 0]
#         prob1 = predictions[:, 1]
#         if predictions.shape[1] == 5:
#             prob2 = predictions[:, 2]
#             prob3 = predictions[:, 3]
#             prob4 = predictions[:, 4]
#         else:
#             prob2 = [0] * len(prob0)
#             prob3 = [0] * len(prob0)
#             prob4 = [0] * len(prob0)
#
#         return [prob0, prob1, prob2, prob3, prob4]
#
#
# class ImageLRClassifier(ModalityClassifier):
#     def __init__(self, seed: int):
#         super(ImageLRClassifier, self).__init__()
#         self.model = LogisticRegression(random_state=seed,
#                                         multi_class="ovr",
#                                         max_iter=1000)
#         self.labeller = MultiLabelBinarizer()
#
#     def fit(self,
#             dataset: BrainyDataset,
#             train_idx: list[int]):
#         data = self.prepare_data(dataset)
#
#         target = [dataset.labels[i] for i in train_idx]
#         self.model.fit(data[train_idx], target)
#         self.labeller.fit(np.expand_dims(target, axis=1))
#
#     def test(self,
#              dataset: BrainyDataset,
#              test_idx: list[int]):
#         data = self.prepare_data(dataset)[test_idx]
#         y = [dataset.labels[i] for i in test_idx]
#         if len(set(y)) > 2:
#             y_hat = [y_pred for y_pred in self.model.predict(data)]
#             y_hat = self.labeller.transform(np.expand_dims(y_hat, axis=1))
#             roc_auc = roc_auc_score(y, y_hat, multi_class='ovr')
#         else:
#             y_hat = self.model.predict_proba(data)[:, 1]
#             roc_auc = roc_auc_score(y, y_hat)
#
#         return roc_auc
#         # wandb.log(dict(supervised_image_test_rocauc=roc_auc))
#
#     @staticmethod
#     def prepare_data(dataset: BrainyDataset):
#         data = copy.deepcopy(dataset.images)
#         data = torch.stack(data, dim=0).squeeze()
#         return np.stack(data.detach().numpy(), axis=0)
#
#     def predict(self, x: Tensor) -> list[float]:
#         if x.ndim == 3:
#             x = x.squeeze()
#
#         predictions = [y_pred for y_pred in self.model.predict(x.cpu())]
#         # predictions = self.model.predict_proba(x.cpu())[:, 1].tolist()
#
#         return predictions


@dataclass
class ModalityModels:
    # Image encoders
    image_encoder: CLIPImageEncoder = field(init=False)
    image_encoder_base: CLIPImageEncoder = field(init=False)
    image_encoder_control: CLIPImageEncoder = field(init=False)

    # Brain encoders.
    brain_encoder: BrainMLPEncoder = field(init=False)
    brain_encoder_scl: BrainMLPEncoder = field(init=False)
    brain_encoder_base: BrainMLPEncoder = field(init=False)
    brain_encoder_control: BrainMLPEncoder = field(init=False)

    # Brain classifiers.
    brain_classifier: BrainMLPClassifier = field(init=False)

    def create_models(self,
                      seed: int,
                      dataset: Union[Literal["nemo"], Literal["aomic"]],
                      projection_dim: int,
                      mlp_hidden_size: int):

        if dataset == "nemo":
            self.__set_encoders(seed, in_dim=24 * 12, classes=2,
                                projection_dim=projection_dim,
                                mlp_hidden_size=mlp_hidden_size)
        elif dataset == "emotion":
            self.__set_encoders(seed, in_dim=33 * 4, classes=2,
                                projection_dim=projection_dim,
                                mlp_hidden_size=mlp_hidden_size)
        elif dataset == "attract":
            self.__set_encoders(seed, in_dim=64 * 7, classes=2,
                                projection_dim=projection_dim,
                                mlp_hidden_size=mlp_hidden_size)
        elif dataset == "aomic":
            self.__set_encoders(seed, in_dim=1024 * 7, classes=5,
                                projection_dim=projection_dim,
                                mlp_hidden_size=mlp_hidden_size)  # TODO: flattening
        else:
            raise ValueError("Dataset {} is not recognized.".format(dataset))

    def __set_encoders(self,
                       seed: int,
                       in_dim: int,
                       classes: int,
                       projection_dim: int,
                       mlp_hidden_size: int):

        self.image_encoder = CLIPImageEncoder(identifier="CLIP-image-encoder",
                                              projection_dim=projection_dim)

        self.brain_encoder = BrainMLPEncoder(in_dim=in_dim,
                                             identifier="MLP-brain-encoder",
                                             projection_dim=projection_dim,
                                             mlp_hidden_size=mlp_hidden_size,
                                             )
        self.brain_classifier = BrainMLPClassifier(in_dim=in_dim,
                                                   identifier="MLP-brain-classifier",
                                                   classes=classes,
                                                   projection_dim=projection_dim,
                                                   mlp_hidden_size=mlp_hidden_size,
                                                   )
        self.image_encoder = self.image_encoder.to(DEVICE)

        self.brain_encoder = self.brain_encoder.to(DEVICE)
        self.brain_encoder.encoder = self.brain_encoder.encoder.to(DEVICE)
        self.brain_encoder.encoder.lin0 = self.brain_encoder.encoder.lin0.to(DEVICE)
        self.brain_encoder.encoder.mlp = self.brain_encoder.encoder.mlp.to(DEVICE)
        self.brain_encoder.encoder.projector = self.brain_encoder.encoder.projector.to(DEVICE)

        self.brain_classifier = self.brain_classifier.to(DEVICE)
        self.brain_classifier.encoder = self.brain_classifier.encoder.to(DEVICE)
        self.brain_classifier.linear = self.brain_classifier.linear.to(DEVICE)
