import logging
from typing import Any, Dict, List, Tuple, Union, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from bale.model.helpers import create_metrics
from bale.model.modality import BrainMLPClassifier

_EVALUATE_OUTPUT = List[Dict[str, float]]  # 1 dict per DataLoader

log = logging.getLogger(__name__)


class ClassifierTrainer(pl.LightningModule):
    r'''
        A generic trainer class for EEG classification.

        .. code-block:: python

            trainer = ClassifierTrainer(model)
            trainer.fit(train_loader, val_loader)
            trainer.test(test_loader)

        Args:
            model (nn.Module): The classification model, and the dimension of its output should be equal to the number of categories in the dataset. The output layer does not need to have a softmax activation function.
            num_classes (int, optional): The number of categories in the dataset. If :obj:`None`, the number of categories will be inferred from the attribute :obj:`num_classes` of the model. (defualt: :obj:`None`)
            lr (float): The learning rate. (default: :obj:`0.001`)
            weight_decay (float): The weight decay. (default: :obj:`0.0`)
            devices (int): The number of devices to use. (default: :obj:`1`)
            accelerator (str): The accelerator to use. Available options are: 'cpu', 'gpu'. (default: :obj:`"cpu"`)
            metrics (list of str): The metrics to use. Available options are: 'precision', 'recall', 'aucroc', 'accuracy'. (default: :obj:`["aucroc"]`)

        .. automethod:: fit
        .. automethod:: test
    '''

    def __init__(self,
                 model: BrainMLPClassifier,
                 num_classes: int = 1,
                 lr: float = 3e-4,
                 weight_decay: float = 0.0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 wandb_logger: WandbLogger = None,
                 metrics=None,
                 brain_modality: bool = True,
                 limit_batches: Optional[Union[int, float]] = None,
                 callbacks: Optional[Union[List[Callback], Callback]] = None,
                 ):

        super().__init__()
        if metrics is None:
            metrics = ["aucroc", "precision", "recall"]

        self.model: BrainMLPClassifier = model
        self.modality_index = 0 if brain_modality else 1
        self.modality_map = {0: "brain", 1: "image"}
        self.wandb_logger = wandb_logger
        self.limit_batches = limit_batches
        self.callbacks = callbacks

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay

        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics

        self.ce_fn = nn.CrossEntropyLoss()

        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

        self.train_metrics = create_metrics(task="multiclass", metric_list=metrics, num_classes=num_classes)
        self.val_metrics = create_metrics(task="multiclass", metric_list=metrics, num_classes=num_classes)
        self.test_metrics = create_metrics(task="multiclass", metric_list=metrics, num_classes=num_classes)

    def fit(self,
            train_loader: DataLoader,
            max_epochs: int,
            val_loader: DataLoader = None,
            *args,
            **kwargs) -> Any:
        r'''
        Args:
            train_loader (DataLoader): Iterable DataLoader for traversing the training data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            val_loader (DataLoader): Iterable DataLoader for traversing the validation data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            max_epochs (int): Maximum number of epochs to train the model. (default: :obj:`300`)
        '''
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             max_epochs=max_epochs,
                             logger=self.wandb_logger,
                             log_every_n_steps=10,
                             num_sanity_val_steps=0,
                             limit_train_batches=self.limit_batches,
                             limit_val_batches=self.limit_batches,
                             limit_test_batches=self.limit_batches,
                             callbacks=self.callbacks,
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
                             num_sanity_val_steps=0,
                             limit_train_batches=self.limit_batches,
                             limit_val_batches=self.limit_batches,
                             limit_test_batches=self.limit_batches,
                             *args,
                             **kwargs)
        return trainer.test(self, test_loader)

    def predict(self, loader: DataLoader, *args,
                **kwargs) -> _EVALUATE_OUTPUT:
        r'''
        Args:
            loader (DataLoader): Iterable DataLoader for traversing the test data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
        '''
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             inference_mode=False,
                             logger=self.wandb_logger,
                             num_sanity_val_steps=0,
                             limit_train_batches=self.limit_batches,
                             limit_val_batches=self.limit_batches,
                             limit_test_batches=self.limit_batches,
                             *args,
                             **kwargs)
        return trainer.predict(model=self, dataloaders=loader)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:

        x = batch[self.modality_index]
        y = batch[2]
        y_hat = self(x)

        loss = self.ce_fn(y_hat, y)

        # log to prog_bar
        self.log("{}_train_loss".format(self.model.identifier),
                 self.train_loss(loss),
                 prog_bar=False,
                 on_epoch=False,
                 logger=False,
                 on_step=True)

        # for i, metric_value in enumerate(self.train_metrics.values()):
        #     self.log(f"{self.modality_map[self.modality_index]}_train_{self.metrics[i]}",
        #              metric_value(y_hat, y),
        #              prog_bar=False,
        #              on_epoch=False,
        #              logger=False,
        #              on_step=True)

        self.train_loss.update(loss)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("{}_train_loss".format(self.modality_map[self.modality_index]),
                 self.train_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        # for i, metric_value in enumerate(self.train_metrics.values()):
        #     self.log(f"{self.modality_map[self.modality_index]}_train_{self.metrics[i]}",
        #              metric_value.compute(),
        #              prog_bar=False,
        #              on_epoch=True,
        #              on_step=False,
        #              logger=True)
        # wandb.log({f"{self.modality_index}_train_{self.metrics[i]}": metric_value.compute()})

        # print the metrics
        # str = "\n[Train] "
        # for key, value in self.trainer.logged_metrics.items():
        #     if key.startswith("train_"):
        #         str += f"{key}: {value:.3f} "
        # logging.info(str + '\n')

        # reset the metrics
        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_step(self, batch: Tuple[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:

        x = batch[self.modality_index]
        y = batch[2]
        y_hat = self(x)

        loss = self.ce_fn(y_hat, y)

        self.val_loss.update(loss)
        self.val_metrics.update(y_hat, y)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("{}_val_loss".format(self.model.identifier),
                 self.val_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        # for i, metric_value in enumerate(self.val_metrics.values()):
        #     self.log(f"{self.modality_map[self.modality_index]}_val_{self.metrics[i]}",
        #              metric_value.compute(),
        #              prog_bar=False,
        #              on_epoch=True,
        #              on_step=False,
        #              logger=True)
        # wandb.log({f"{self.modality_index}_val_{self.metrics[i]}": metric_value.compute()})

        # print the metrics
        # str = "\n[Val] "
        # for key, value in self.trainer.logged_metrics.items():
        #     if key.startswith("val_"):
        #         str += f"{key}: {value:.3f} "
        # logging.info(str + '\n')

        self.val_loss.reset()
        self.val_metrics.reset()

    def test_step(self, batch: Tuple[torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        x = batch[self.modality_index]
        y = batch[2]
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        self.test_loss.update(loss)
        self.test_metrics.update(y_hat, y)
        return loss

    def on_test_epoch_end(self) -> None:
        self.log("{}_test_loss".format(self.model.identifier),
                 self.test_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=False)
        for i, metric_value in enumerate(self.test_metrics.values()):
            self.log(f"{self.model.identifier}_test_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)
            # wandb.log({f"{self.modality_index}_test_{self.metrics[i]}": metric_value.compute()})

        # print the metrics
        str = "\n[Test] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("test_"):
                str += f"{key}: {value:.3f} "
        logging.info(str + '\n')

        self.test_loss.reset()
        self.test_metrics.reset()

    def configure_optimizers(self):
        parameters = list(self.model.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        optimizer = torch.optim.AdamW(trainable_parameters,
                                      lr=self.lr,
                                      weight_decay=self.weight_decay)
        scheduler = [
            {
                'scheduler': CosineAnnealingLR(optimizer, T_max=10),
            }]
        return [optimizer], scheduler

    def predict_step(self,
                     batch: Tuple[torch.Tensor],
                     batch_idx: int,
                     dataloader_idx: int = 0):
        x = batch[self.modality_index]
        y = batch[2]
        y_hat = self(x)
        return {'y_hat': y_hat, 'y': y}

    def generate_predictions(self, dataloader):
        output = self.predict(dataloader)[0]
        target = output['y']
        logits = output['y_hat']

        probabilities = torch.softmax(logits, dim=1)
        _, predicted_class = torch.max(probabilities, 1)

        return target, predicted_class, probabilities
