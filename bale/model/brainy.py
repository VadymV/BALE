import logging
import pickle
from dataclasses import dataclass, field
from typing import Any, Tuple, List, Dict, Optional, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import balanced_accuracy_score
from torch import Tensor
from torch.utils.data import DataLoader

from bale.model.helpers import count_trainable_parameters, \
    get_brainy_predictions, get_rankings
from bale.model.loss import ClipLoss, SigmoidContrastiveLoss, MSELoss
from bale.model.modality import CLIPImageEncoder, BrainMLPEncoder, BrainMLPClassifier

_EVALUATE_OUTPUT = List[Dict[str, float]]  # 1 dict per DataLoader


@dataclass
class BrainyPredictions:
    target: list[int] = field(default_factory=list)
    preds_brainclas: list[int] = field(default_factory=list)
    preds_scl: list[int] = field(default_factory=list)
    preds_base: list[int] = field(default_factory=list)
    preds_control: list[int] = field(default_factory=list)
    preds_image: list[float] = field(default_factory=list)
    preds_brain: list[float] = field(default_factory=list)
    preds_brain0: list[float] = field(default_factory=list)
    preds_brain1: list[float] = field(default_factory=list)
    preds_brain2: list[float] = field(default_factory=list)
    preds_brain3: list[float] = field(default_factory=list)
    preds_brain4: list[float] = field(default_factory=list)
    preds_mlp_brain: list[float] = field(default_factory=list)
    preds_mlp_brain0: list[float] = field(default_factory=list)
    preds_mlp_brain1: list[float] = field(default_factory=list)
    preds_mlp_brain2: list[float] = field(default_factory=list)
    preds_mlp_brain3: list[float] = field(default_factory=list)
    preds_mlp_brain4: list[float] = field(default_factory=list)
    image_idx: list[int] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)

    def fill(self,
             target: list[int],
             preds_brainclas: list[int],
             preds_scl: list[int],
             preds_base: list[int],
             preds_control: list[int],
             preds_image: list[float],
             preds_brain: list[float],
             preds_brain0: list[float],
             preds_brain1: list[float],
             preds_brain2: list[float],
             preds_brain3: list[float],
             preds_brain4: list[float],
             preds_mlp_brain: list[float],
             preds_mlp_brain0: list[float],
             preds_mlp_brain1: list[float],
             preds_mlp_brain2: list[float],
             preds_mlp_brain3: list[float],
             preds_mlp_brain4: list[float],
             image_idx: list[int],
             participant: list[str]) -> None:
        self.target.extend(target)
        self.preds_brainclas.extend(preds_brainclas)
        self.preds_scl.extend(preds_scl)
        self.preds_base.extend(preds_base)
        self.preds_control.extend(preds_control)
        self.preds_image.extend(preds_image)
        self.preds_brain.extend(preds_brain)
        self.preds_brain0.extend(preds_brain0)
        self.preds_brain1.extend(preds_brain1)
        self.preds_brain2.extend(preds_brain2)
        self.preds_brain3.extend(preds_brain3)
        self.preds_brain4.extend(preds_brain4)
        self.preds_mlp_brain.extend(preds_mlp_brain)
        self.preds_mlp_brain0.extend(preds_mlp_brain0)
        self.preds_mlp_brain1.extend(preds_mlp_brain1)
        self.preds_mlp_brain2.extend(preds_mlp_brain2)
        self.preds_mlp_brain3.extend(preds_mlp_brain3)
        self.preds_mlp_brain4.extend(preds_mlp_brain4)
        self.image_idx.extend(image_idx)
        self.participants.extend(participant)

    def save(self, file_path: Optional[str]) -> None:
        if file_path is not None:
            with open(file_path, 'wb') as file:
                pickle.dump(self, file)

    def load(self, file_path: str):
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def to_data_frame(self, fold: int, seed: int) -> pd.DataFrame:
        data = {"target": self.target,
                "preds_brainclas": self.preds_brainclas,
                "preds_base": self.preds_base,
                "preds_control": self.preds_control,
                "preds_image": self.preds_image,
                "preds_brain": self.preds_brain,
                "preds_brain0": self.preds_brain0,
                "preds_brain1": self.preds_brain1,
                "preds_brain2": self.preds_brain2,
                "preds_brain3": self.preds_brain3,
                "preds_brain4": self.preds_brain4,
                "preds_mlp_brain": self.preds_mlp_brain,
                "preds_mlp_brain0": self.preds_mlp_brain0,
                "preds_mlp_brain1": self.preds_mlp_brain1,
                "preds_mlp_brain2": self.preds_mlp_brain2,
                "preds_mlp_brain3": self.preds_mlp_brain3,
                "preds_mlp_brain4": self.preds_mlp_brain4,
                "img_idx": self.image_idx,
                "participants": self.participants,
                "fold": [fold for _ in range(len(self.image_idx))],
                "seed": [seed for _ in range(len(self.image_idx))],
                }
        df = pd.DataFrame(data)

        return df


class BrainyModel(nn.Module):
    def __init__(self,
                 image_encoder: CLIPImageEncoder,
                 brain_encoder: BrainMLPEncoder,
                 loss: Union[ClipLoss, MSELoss, SigmoidContrastiveLoss],
                 identifier: str,
                 brainy_permute_pairs: bool = False,
                 ) -> None:
        super(BrainyModel, self).__init__()

        self.image_encoder = image_encoder
        self.brain_encoder = brain_encoder
        self.tau = nn.Parameter(torch.ones([]) * np.log(5))
        self.b = nn.Parameter(torch.ones([]) * -5)
        self.loss = loss,
        self.identifier = identifier
        self.brainy_permute_pairs = brainy_permute_pairs

    def encode_image(self, image_input: Tensor) -> Tensor:
        """
        Encodes an image into a projection space.
        :param image_input: An image representation.
        :return: An encoded image in a projection space.
        """
        return self.image_encoder(image_input).squeeze(1)

    def encode_brain(self, brain_input: Tensor) -> Tensor:
        """
        Encodes brain data into a projection space.
        :param brain_input: Brain signals representation.
        :return: Encoded brain signals in a projection space.
        """
        brain_projection = self.brain_encoder(brain_input)

        return brain_projection.squeeze(1)

    def forward(self, brain_signals: Tensor, images: Tensor) -> (
            Tensor, Tensor):
        """
        Forward method.
        :param brain_signals: Brain data.
        :param images: Images.
        :return:
        """
        brain_features = self.encode_brain(brain_signals)
        image_features = self.encode_image(images)

        # Normalized features.
        brain_features = brain_features / brain_features.norm(dim=1,
                                                              keepdim=True)
        image_features = image_features / image_features.norm(dim=1,
                                                              keepdim=True)

        # cosine similarity as logits
        logits_per_image = (image_features @ brain_features.t())
        logits_per_brain = logits_per_image.t()

        return logits_per_image, logits_per_brain


class BrainyTrainer(pl.LightningModule):

    def __init__(self,
                 model: BrainyModel,
                 epochs: int,
                 steps_per_epoch: int,
                 callbacks: Optional[Union[List[Callback], Callback]] = None,
                 lr: float = 3e-4,
                 weight_decay: float = 0.0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 wandb_logger: WandbLogger = None,
                 limit_batches: Optional[Union[int, float]] = None,
                 ):
        super().__init__()

        self.model: BrainyModel = model
        self.callbacks = callbacks
        self.wandb_logger = wandb_logger
        self.limit_batches = limit_batches

        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        self.lr = lr
        self.weight_decay = weight_decay

        self.devices = devices
        self.accelerator = accelerator

        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

        self.count_trainable_parameters()

    def count_trainable_parameters(self):
        logging.info("Brainy:")
        count_trainable_parameters(self.model, log_non_trainable_params=True)

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader = None,
            *args,
            **kwargs) -> Any:
        trainer = pl.Trainer(devices=self.devices,
                             callbacks=self.callbacks,
                             accelerator=self.accelerator,
                             max_epochs=self.epochs,
                             log_every_n_steps=1,
                             logger=self.wandb_logger,
                             limit_train_batches=self.limit_batches,
                             limit_val_batches=self.limit_batches,
                             limit_test_batches=self.limit_batches,
                             num_sanity_val_steps=0,
                             enable_checkpointing=False,
                             *args,
                             **kwargs)
        return trainer.fit(self, train_loader, val_loader)

    def test(self, test_loader: DataLoader, *args,
             **kwargs) -> _EVALUATE_OUTPUT:
        r'''
        Args:
            test_loader (DataLoader): Iterable DataLoader for traversing the test data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
        '''
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             inference_mode=False,
                             logger=self.wandb_logger,
                             limit_train_batches=self.limit_batches,
                             limit_val_batches=self.limit_batches,
                             limit_test_batches=self.limit_batches,
                             *args,
                             **kwargs)
        return trainer.test(self, test_loader)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.model(x, z)

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        brain, image, y, _, _ = batch
        if self.model.brainy_permute_pairs:
            brain = brain[torch.randperm(brain.size()[0])]
            image = image[torch.randperm(image.size()[0])]
        image_embeddings, brain_embeddings = self(brain, image)
        loss = self.model.loss[0](image_embeddings, brain_embeddings, self.model.tau, self.model.b)

        # log to prog_bar
        self.log("{}_train_loss_step".format(self.model.identifier),
                 self.train_loss(loss),
                 on_epoch=False,
                 logger=False,
                 on_step=True)

        self.train_loss.update(loss)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("{}_train_loss".format(self.model.identifier),
                 self.train_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)

        self.train_loss.reset()

    def validation_step(self, batch: Tuple[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        brain, image, y, _, _ = batch
        image_embeddings, brain_embeddings = self(brain, image)
        loss = self.model.loss[0](image_embeddings, brain_embeddings, self.model.tau, self.model.b)

        # log to prog_bar
        self.log("{}_val_loss_step".format(self.model.identifier),
                 self.val_loss(loss),
                 on_epoch=False,
                 logger=False,
                 on_step=True)

        self.val_loss.update(loss)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("{}_val_loss".format(self.model.identifier),
                 self.val_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)

        self.val_loss.reset()

    def test_step(self, batch: Tuple[torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        brain, image, y, _, _ = batch
        image_embeddings, brain_embeddings = self(brain, image)
        loss = self.model.loss[0](image_embeddings, brain_embeddings, self.model.tau, self.model.b)

        self.test_loss.update(loss)
        return loss

    def on_test_epoch_end(self) -> None:
        self.log("{}_test_loss".format(self.model.identifier),
                 self.test_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=False)

        self.test_loss.reset()

    def configure_optimizers(self):
        parameters = list(self.model.parameters())
        trainable_parameters = list(
            filter(lambda p: p.requires_grad, parameters))
        optimizer = torch.optim.AdamW(trainable_parameters,
                                      lr=self.lr,
                                      weight_decay=self.weight_decay)
        monitor_metric = '{}_val_loss'.format(self.model.identifier)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch
        )
        scheduler = [
            {
                'scheduler': lr_scheduler,
                'monitor': monitor_metric,
            }]
        return [optimizer], scheduler


def generate_supervised_rankings(model: BrainMLPClassifier,
                                 loader: DataLoader,
                                 device: str,
                                 seed: int,
                                 fold: int) -> pd.DataFrame:
    """
    Generates rankings using brain classifier.
    :param model:
    :param loader:
    :param device:
    :param seed:
    :param fold:
    :return:
    """
    all_rankings = []

    with torch.no_grad():
        id = 0
        for batch_idx, (
                brain_data, images, labels, img_idx, participants_list) in enumerate(
            loader):
            brain_data, images, labels = brain_data.to(device), images.to(
                device), labels.to(
                device)

            if id > 0:
                raise ValueError()

            predicted_classes, probabilities = model.predict_each_class(brain_data)
            probabilities = probabilities.to('cpu').detach().numpy()

            true_image_indices = []
            predicted_image_indices = []
            true_labels = []
            predicted_labels = []
            ranks = []
            queries = []
            participants = []
            similarity_scores = []

            for idx in range(len(predicted_classes)):

                highest_probabilities = np.array([probabilities[i, j] for i, j in enumerate(predicted_classes, 0)])
                indices = np.argsort(highest_probabilities)[::-1]

                temp_predicted_image_indices = []
                minus_rank = 0
                for rank, index in enumerate(indices, 1):
                    if img_idx[index].item() in temp_predicted_image_indices:
                        minus_rank += 1
                        continue
                    true_labels.append(labels[idx].item())
                    predicted_labels.append(predicted_classes[idx].item())
                    true_image_indices.append(img_idx[idx].item())
                    predicted_image_indices.append(img_idx[index].item())
                    ranks.append(rank - minus_rank)
                    queries.append(img_idx[idx].item())
                    participants.append(participants_list[idx].item())
                    similarity_scores.append(highest_probabilities[index].item())

                    temp_predicted_image_indices.append(img_idx[index].item())

            df = pd.DataFrame(list(
                zip(queries, ranks, true_image_indices, predicted_image_indices, true_labels, predicted_labels,
                    participants, similarity_scores)),
                columns=['query', 'rank', 'true_image', 'predicted_image', 'true_label', 'predicted_label',
                         'participant', 'similarity_score'])

            df['seed'] = seed
            df['fold'] = fold

            all_rankings.append(df)
            id += 1

    return pd.concat(all_rankings, axis=0)


def generate_clas_rankings(model: BrainyModel,
                           loader: DataLoader,
                           device: str,
                           seed: int,
                           fold: int) -> pd.DataFrame:
    """
    Generates rankings.
    :param loader: Data loader.
    :param device: Device
    :return: Rankings.
    """
    all_rankings = []

    with torch.no_grad():
        id = 0
        for batch_idx, (
                brain_data, images, labels, img_idx, participants) in enumerate(
            loader):
            brain_data, images, labels = brain_data.to(device), images.to(
                device), labels.to(
                device)

            if id > 0:
                raise ValueError()

            brain_embeddings = model.encode_brain(brain_data)
            image_embeddings = model.encode_image(images)

            rankings = get_rankings(
                brain_embeddings=brain_embeddings,
                image_embeddings=image_embeddings,
                labels=labels,
                img_idx=img_idx,
                participant_list=participants.tolist(),
                seed_run=seed,
                fold=fold)

            all_rankings.append(rankings)
            id += 1

    return pd.concat(all_rankings, axis=0)


def calc_accuracy_supervised(model: BrainMLPClassifier,
                             loader: DataLoader,
                             device: str) -> (float, float):
    """
    Calculates accuracy for the supervised model.
    :param model:
    :param loader:
    :param device:
    :return:
    """
    model.eval()

    target = []
    preds = []
    with torch.no_grad():
        for batch_idx, (brain_data, images, labels, _, _) in enumerate(loader):
            brain_data, images, labels = brain_data.to(device), images.to(device), labels.to(device)
            predicted_classes, _ = model.predict_each_class(brain_data)
            target.extend(labels)
            preds.extend(predicted_classes)

    target = [i.item() for i in target]
    preds = [i.item() for i in preds]
    acc = balanced_accuracy_score(target, preds)

    logging.info("{} Accuracy: {}".format(model.identifier, acc))

    return acc


def calc_accuracy_bale(model: BrainyModel,
                       loader: DataLoader,
                       device: str) -> (float, float):
    """
    Calculates accuracy for the BALE model.
    :param model:
    :param loader:
    :param device:
    :return:
    """
    model.eval()

    target = []
    preds = []
    loss = []
    with torch.no_grad():
        for batch_idx, (brain_data, images, labels, _, _) in enumerate(loader):
            brain_data, images, labels = brain_data.to(device), images.to(
                device), labels.to(
                device)

            brain_embeddings = model.encode_brain(brain_data)
            image_embeddings = model.encode_image(images)

            logits_per_image, logits_per_brain = model(brain_data, images)
            loss_f = ClipLoss()
            loss.append(loss_f(logits_per_image, logits_per_brain, model.tau, model.b).detach().cpu())

            brainy_predictions, true_labels = get_brainy_predictions(
                brain_embeddings=brain_embeddings,
                image_embeddings=image_embeddings,
                labels=labels,
                top_k=1)

            target.extend(true_labels)
            preds.extend(brainy_predictions)

    acc = balanced_accuracy_score(target, preds)
    loss = sum(np.array(loss)) / len(loss)

    logging.info("{} Accuracy: {}".format(model.identifier, acc))

    return acc, loss
