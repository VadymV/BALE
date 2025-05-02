from __future__ import print_function

import argparse
import logging
import os
import shutil
from pathlib import Path
from pprint import pformat, pprint
from typing import Union, Literal

import pandas as pd
import ray
import torch
import yaml
from pytorch_lightning import Callback
from ray import train
from ray import tune
from torch.utils.data import DataLoader, Subset
from yaml import SafeLoader

from bale import DEVICE, PROJECT_PATH, get_dataset_key, NUM_WORKERS, PIN_MEMORY
from bale.dataset import DATASETS
from bale.dataset.dataset import BrainyDataset
from bale.dataset.helpers import get_loader
from bale.helpers.misc import set_seed, set_logging
from bale.model.brainy import BrainyModel, BrainyTrainer, calc_accuracy_bale, generate_clas_rankings, \
    generate_supervised_rankings, calc_accuracy_supervised
from bale.model.classification_trainer import ClassifierTrainer
from bale.model.helpers import log_all_parameters
from bale.model.loss import ClipLoss, MSELoss
from bale.model.modality import ModalityModels, CLIPImageEncoder, BrainMLPEncoder, BrainMLPClassifier


class CheckpointCleanupCallback(Callback):
    def on_train_end(self, trainer, pl_module):
        checkpoint_dir = trainer.checkpoint_callback.dirpath
        if checkpoint_dir and os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            print(f"Deleted all checkpoints in {checkpoint_dir} due to early stopping.")


def _initialize_bale_model(image_encoder: CLIPImageEncoder,
                           brain_encoder: BrainMLPEncoder,
                           model_type: Union[
                               Literal["clip"], Literal["mse"], Literal["control"]] = "clip",
                           log: bool = False) -> BrainyModel:
    """
    Initializes the BALE model.
    :param image_encoder: Image encoder
    :param brain_encoder: Brain encoder
    :param model_type: Model type
    :param log: A boolean flag stating whether to print the named model parameters
    :return: BALE model.
    """

    logging.info("Initializing BALE model")
    image_encoder.train()
    brain_encoder.train()

    brainy_permute_pairs = True if model_type == "control" else False
    loss = get_loss(model_type=model_type)

    model = BrainyModel(
        image_encoder=image_encoder,
        brain_encoder=brain_encoder,
        loss=loss,
        brainy_permute_pairs=brainy_permute_pairs,
        identifier=model_type)
    model = model.to(DEVICE)
    if log:
        log_all_parameters(model)

    return model


def create_dataset(data_dir: Path,
                   dataset_name: Union[Literal["nemo"], Literal["aomic"]],
                   is_tune: bool,
                   data_cv_strategy: str) -> BrainyDataset:
    """
    Creates a dataset
    :param data_dir:
    :param dataset_name:
    :param is_tune:
    :param data_cv_strategy:
    :return:
    """
    logging.info("Dataset path: {}".format(data_dir))
    dataset_func = DATASETS[
        get_dataset_key(dataset=dataset_name)]
    dataset = dataset_func(data_path=data_dir,
                           tune=True if data_cv_strategy == "subject-independent" else is_tune, # TODO: This is a hack
                           data_cv_strategy=data_cv_strategy
                           )

    dataset.cv_strategy.create_train_and_test_indices(dataset=dataset_name,
                                                      tune=is_tune)

    return dataset


def create_data_sets(data_dir: Path,
                     configuration: dict,
                     fold: int,
                     dataset: Union[BrainyDataset, None] = None) -> tuple[Subset, Subset]:
    """
    Creates train and test data sets
    :param data_dir: The path to the dataset
    :param configuration: Configuration
    :param fold: Fold identifier
    :param dataset:
    :return: Train and test data sets
    """
    if dataset is None:
        dataset = create_dataset(data_dir,
                                 dataset_name=configuration["dataset"],
                                 is_tune=configuration["tune"],
                                 data_cv_strategy=configuration["data_cv_strategy"])
    participants = dataset.cv_strategy.indices.participants[fold]
    configuration["participants"] = participants

    train_idx = dataset.cv_strategy.indices.train_indices[fold]
    test_idx = dataset.cv_strategy.indices.test_indices[fold]

    logging.info("Test user(s) is(are) {}".format(participants))
    logging.info("Train samples: {}".format(len(train_idx)))
    logging.info("Test samples: {}".format(len(test_idx)))

    train_d = Subset(dataset, train_idx)
    test_d = Subset(dataset, test_idx)

    configuration["classes"] = dataset.classes

    return train_d, test_d


def get_loss(model_type: Union[Literal["clip"], Literal["mse"], Literal["control"]] = "clip"):
    """
    Selects the loss function
    :param model_type: Model type
    :return: The loss
    """
    if model_type == "clip" or model_type == "control":
        return ClipLoss()
    elif model_type == "mse":
        return MSELoss()


def evaluate_supervised(model: BrainMLPClassifier,
                        data_loader: DataLoader):
    """
    Evaluates the BALE model
    :param model: The BALE model
    :param data_loader: Dataloader
    """
    model = model.eval()
    model = model.to(DEVICE)

    acc = calc_accuracy_supervised(model=model,
                                   loader=data_loader,
                                   device=DEVICE)

    logging.info(f"{model.identifier} Accuracy: {acc}")

    return acc


def evaluate_bale(model: BrainyModel,
                  data_loader: DataLoader):
    """
    Evaluates the BALE model
    :param model: The BALE model
    :param data_loader: Dataloader
    :param is_ray_report:
    """
    model.eval()
    model = model.to(DEVICE)

    acc, loss = calc_accuracy_bale(model=model,
                                   loader=data_loader,
                                   device=DEVICE)

    logging.info(f"{model.identifier} Loss: {loss} Accuracy: {acc}")

    return loss, acc


def save_rankings(data: pd.DataFrame,
                  save_dir: str,
                  identifier: str):
    """
    Saves the generated rankings
    :param data: The rankings data
    :param save_dir: A directory to save the data
    :param identifier: Unique identifier of the file to be saved
    """
    file_path = os.path.join(save_dir, identifier)
    logging.info("Saving data to {}".format(file_path))
    data.to_csv(file_path, sep=",", index=False)


def create_modality_models(seed: int, dataset: Union[Literal["nemo"], Literal["aomic"]],
                           projection_dim: int, mlp_hidden_size: int) -> ModalityModels:
    """
    Creates the modality models
    :param seed:
    :param dataset:
    :param projection_dim:
    :param mlp_hidden_size:
    :return:
    """
    modality_models: ModalityModels = ModalityModels()
    modality_models.create_models(seed=seed,
                                  dataset=dataset,
                                  projection_dim=projection_dim,
                                  mlp_hidden_size=mlp_hidden_size)

    return modality_models


def get_supervised_trainer(model: BrainMLPClassifier,
                           learning_rate: float,
                           weight_decay: float,
                           limit_batches: float,
                           classes: int) -> ClassifierTrainer:
    """
    Creates a supervised trainer
    :param model:
    :param learning_rate:
    :param weight_decay:
    :param limit_batches:
    :param classes:
    :return:
    """

    supervised_trainer = ClassifierTrainer(
        model=model,
        brain_modality=True,
        accelerator=DEVICE,
        num_classes=classes,
        lr=learning_rate,
        weight_decay=weight_decay,
        limit_batches=limit_batches)

    return supervised_trainer


def run_tune_supervised(param_config: dict,
                        configuration: dict,
                        dataset: BrainyDataset) -> float:
    """
    Runs supervised learning
    :param param_config:
    :param configuration: Configuration
    """
    # noinspection DuplicatedCode
    train_subset, test_subset = create_data_sets(data_dir=configuration["features_path"],
                                                 configuration=configuration,
                                                 fold=configuration["fold"],
                                                 dataset=dataset)

    model = create_modality_models(seed=configuration["seed"],
                                   dataset=configuration["dataset"],
                                   projection_dim=param_config["projection_dim"],
                                   mlp_hidden_size=param_config["mlp_h_dim"]).brain_classifier

    train_loader = get_loader(dataset=train_subset,
                              sampler=None,
                              shuffle=True,
                              batch_size=param_config["batch_size"],
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY,
                              drop_last_batch=False)

    # noinspection DuplicatedCode
    trainer = get_supervised_trainer(model=model,
                                     learning_rate=configuration["learning_rate"],
                                     weight_decay=configuration["weight_decay"],
                                     limit_batches=configuration["limit_batches"],
                                     classes=configuration["classes"])

    test_loader = get_loader(dataset=test_subset,
                             sampler=None,
                             num_workers=NUM_WORKERS,
                             pin_memory=PIN_MEMORY,
                             batch_size=len(test_subset))

    logging.info("Tuning the {} classifier".format(model.__class__.__name__))
    trainer.fit(train_loader=train_loader,
                max_epochs=param_config["epochs"])

    acc = evaluate_supervised(model=model, data_loader=test_loader)

    return acc


# noinspection DuplicatedCode
def run_train_supervised(configuration: dict,
                         dataset: BrainyDataset):
    """
    Runs supervised learning
    :param dataset:
    :param configuration: Configuration
    """
    train_subset, test_subset = create_data_sets(data_dir=configuration["features_path"],
                                                 configuration=configuration,
                                                 fold=configuration["fold"],
                                                 dataset=dataset)

    model = create_modality_models(seed=configuration["seed"],
                                   dataset=configuration["dataset"],
                                   projection_dim=configuration["supervised_projection_dim"],
                                   mlp_hidden_size=configuration["supervised_mlp_h_dim"]).brain_classifier

    train_loader = get_loader(dataset=train_subset,
                              sampler=None,
                              shuffle=True,
                              batch_size=configuration["supervised_batch_size"],
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY,
                              drop_last_batch=False)

    trainer = get_supervised_trainer(model=model,
                                     learning_rate=configuration["learning_rate"],
                                     weight_decay=configuration["weight_decay"],
                                     limit_batches=configuration["limit_batches"],
                                     classes=configuration["classes"])

    test_loader = get_loader(dataset=test_subset,
                             sampler=None,
                             num_workers=NUM_WORKERS,
                             pin_memory=PIN_MEMORY,
                             batch_size=len(test_subset))

    logging.info("Training the {} classifier".format(model.__class__.__name__))
    trainer.fit(train_loader=train_loader,
                max_epochs=configuration["supervised_epochs"])

    # Brain rankings
    model.eval()
    model = model.to(DEVICE)
    rankings = generate_supervised_rankings(model=model,
                                            loader=test_loader,
                                            device=DEVICE,
                                            seed=configuration["seed"],
                                            fold=configuration["fold"])
    identifier = "rankings-{}-{}-{}-{}".format("supervised",
                                               configuration["project_name"],
                                               configuration["seed"],
                                               configuration["fold"])
    save_rankings(rankings,
                  save_dir=configuration["save_dir"],
                  identifier=identifier)


def get_bale_trainer(model: BrainyModel,
                     epochs: int,
                     learning_rate: float,
                     weight_decay: float,
                     limit_batches: float,
                     steps: int) -> BrainyTrainer:
    """
    Creates a bale trainer
    :param model:
    :param epochs:
    :param learning_rate:
    :param weight_decay:
    :param limit_batches:
    :param steps:
    :return:
    """
    clas_trainer = BrainyTrainer(model=model,
                                 steps_per_epoch=steps,
                                 epochs=epochs,
                                 accelerator=DEVICE,
                                 lr=learning_rate,
                                 weight_decay=weight_decay,
                                 limit_batches=limit_batches)

    return clas_trainer


def cleanup():
    checkpoint_dir = ray.train.get_context().get_trial_dir()
    if os.path.exists(checkpoint_dir):
        for file in os.listdir(checkpoint_dir):
            if file.endswith(".ckpt"):
                os.remove(os.path.join(checkpoint_dir, file))
                logging.info(f"Deleted checkpoint: {file}")


# noinspection DuplicatedCode
def run_bale_train(configuration: dict,
                   dataset: BrainyDataset):
    """
    Runs contrastive learning
    :param dataset:
    :param configuration: Configuration
    """
    train_subset, test_subset = create_data_sets(data_dir=configuration["features_path"],
                                                 configuration=configuration,
                                                 fold=configuration["fold"],
                                                 dataset=dataset)

    modality_models = create_modality_models(seed=configuration["seed"],
                                             dataset=configuration["dataset"],
                                             projection_dim=configuration["bale_projection_dim"],
                                             mlp_hidden_size=configuration["bale_mlp_h_dim"])
    model: BrainyModel = _initialize_bale_model(
        image_encoder=modality_models.image_encoder,
        brain_encoder=modality_models.brain_encoder,
        model_type=configuration["model_type"])

    train_loader = get_loader(dataset=train_subset,
                              sampler=None,
                              shuffle=True,
                              batch_size=configuration["bale_batch_size"],
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY,
                              drop_last_batch=False)

    trainer = get_bale_trainer(model=model,
                               epochs=configuration["bale_epochs"],
                               learning_rate=configuration["learning_rate"],
                               weight_decay=configuration["weight_decay"],
                               limit_batches=configuration["limit_batches"],
                               steps=train_loader.dataset.__len__())

    logging.info("BALE training.")
    trainer.fit(train_loader=train_loader)

    test_loader = get_loader(dataset=test_subset,
                             sampler=None,
                             num_workers=NUM_WORKERS,
                             pin_memory=PIN_MEMORY,
                             batch_size=len(test_subset))

    # BALE rankings
    model.eval()
    model = model.to(DEVICE)
    rankings = generate_clas_rankings(model=model,
                                      loader=test_loader,
                                      device=DEVICE,
                                      seed=configuration["seed"],
                                      fold=configuration["fold"])
    identifier = "rankings-{}-{}-{}-{}".format(model.identifier,
                                               configuration["project_name"],
                                               configuration["seed"],
                                               configuration["fold"])
    save_rankings(rankings,
                  save_dir=configuration["save_dir"],
                  identifier=identifier)

    evaluate_bale(model=model, data_loader=test_loader)


def run_tune_bale(param_config: dict,
                  configuration: dict,
                  dataset: BrainyDataset):
    """
    Runs hyperparameter search
    :param param_config: Hyperparameters
    :param configuration: Configuration
    """
    # noinspection DuplicatedCode
    train_subset, test_subset = create_data_sets(data_dir=configuration["features_path"],
                                                 configuration=configuration,
                                                 fold=configuration["fold"],
                                                 dataset=dataset)

    modality_models = create_modality_models(seed=configuration["seed"],
                                             dataset=configuration["dataset"],
                                             projection_dim=param_config["projection_dim"],
                                             mlp_hidden_size=param_config["mlp_h_dim"])

    # noinspection DuplicatedCode
    model: BrainyModel = _initialize_bale_model(
        image_encoder=modality_models.image_encoder,
        brain_encoder=modality_models.brain_encoder,
        model_type=configuration["model_type"])

    train_loader = get_loader(dataset=train_subset,
                              sampler=None,
                              shuffle=True,
                              batch_size=param_config["batch_size"],
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY,
                              drop_last_batch=False)
    test_loader = get_loader(dataset=test_subset,
                             sampler=None,
                             num_workers=NUM_WORKERS,
                             pin_memory=PIN_MEMORY,
                             batch_size=len(test_subset))

    trainer = BrainyTrainer(model=model,
                            steps_per_epoch=train_loader.dataset.__len__(),
                            epochs=param_config["epochs"],
                            accelerator=DEVICE,
                            lr=configuration["learning_rate"],
                            weight_decay=configuration["weight_decay"],
                            limit_batches=configuration["limit_batches"])
    logging.info("BALE tuning.")
    trainer.fit(train_loader=train_loader)

    loss, acc = evaluate_bale(model=model, data_loader=test_loader)

    return loss, acc


def ray_tune_supervised(configuration: dict,
                        tune_results: pd.DataFrame,
                        dataset: BrainyDataset) -> pd.DataFrame:
    """
    Runs hyperparameter search
    :param configuration: Configuration
    :param tune_results: Data frame containing the results
    :return: The data frame containing the results for hyperparameters that lead to the best accuracy
    """

    configuration["model_type"] = "supervised"
    ray_tune_config = {
        "batch_size": tune.randint(lower=32, upper=configuration["train_samples"]).sample(),
        "projection_dim": tune.randint(lower=32, upper=1025).sample(),
        "mlp_h_dim": tune.randint(lower=32, upper=4096).sample(),
        "epochs": tune.randint(lower=10, upper=300).sample(),
    }
    acc = run_tune_supervised(param_config=ray_tune_config,
                              configuration=configuration,
                              dataset=dataset)

    tune_results = tune_results.append({
        "batch_size": ray_tune_config["batch_size"],
        "projection_dim": ray_tune_config["projection_dim"],
        "mlp_h_dim": ray_tune_config["mlp_h_dim"],
        "epochs": ray_tune_config["epochs"],
        "accuracy": acc,
    }, ignore_index=True)

    return tune_results


def ray_tune_bale(configuration: dict,
                  tune_results: pd.DataFrame,
                  dataset: BrainyDataset) -> pd.DataFrame:
    """
    Runs hyperparameter search
    :param configuration: Configuration
    :param tune_results: Data frame containing the results
    :return: The data frame containing the results for hyperparameters that lead to the best accuracy
    """

    configuration["model_type"] = "clip"
    ray_tune_config = {
        "batch_size": tune.randint(lower=32, upper=configuration["train_samples"]).sample(),
        "projection_dim": tune.randint(lower=32, upper=1025).sample(),
        "mlp_h_dim": tune.randint(lower=32, upper=4096).sample(),
        "epochs": tune.randint(lower=10, upper=300).sample(),
    }
    loss, acc = run_tune_bale(param_config=ray_tune_config,
                              configuration=configuration,
                              dataset=dataset)

    tune_results = tune_results.append({
        "batch_size": ray_tune_config["batch_size"],
        "projection_dim": ray_tune_config["projection_dim"],
        "mlp_h_dim": ray_tune_config["mlp_h_dim"],
        "epochs": ray_tune_config["epochs"],
        "loss": loss,
        "accuracy": acc,
    }, ignore_index=True)

    return tune_results


def get_best_hyperparameters_bale(tune_results: pd.DataFrame,
                                  save_dir: str,
                                  file_name: str):
    """
    Finds the best hyperparameters
    :param tune_results: Data frame containing the results
    """
    # Average results across all hyperparameters
    tune_results = tune_results.groupby(
        ["batch_size", "projection_dim", "mlp_h_dim", "epochs"]).mean().reset_index()
    best_trial = tune_results.loc[tune_results["accuracy"].idxmax()]
    best_trial.to_csv(os.path.join(save_dir, file_name))
    print(f"Best hyperparameters (BALE): {best_trial}")


def get_best_hyperparameters_supervised(tune_results: pd.DataFrame,
                                        save_dir: str,
                                        file_name: str):
    """
    Finds the best hyperparameters
    :param tune_results: Data frame containing the results
    """
    # Average results across all hyperparameters
    tune_results = tune_results.groupby(
        ["batch_size", "projection_dim", "mlp_h_dim", "epochs"]).mean().reset_index()
    best_trial = tune_results.loc[tune_results["accuracy"].idxmax()]
    best_trial.to_csv(os.path.join(save_dir, file_name))
    print(f"Best hyperparameters (supervised): {best_trial}")


def run(configuration: dict) -> None:
    """
    Runs the experiment.
    :param configuration: Configuration
    """
    logging.info(pformat(configuration))
    logging.basicConfig(level=logging.INFO)
    tune_results_bale = pd.DataFrame(columns=["batch_size",
                                              "projection_dim",
                                              "mlp_h_dim",
                                              "epochs",
                                              "loss",
                                              "accuracy"])
    tune_results_supervised = pd.DataFrame(columns=["batch_size",
                                                    "epochs",
                                                    "projection_dim",
                                                    "mlp_h_dim",
                                                    "accuracy"])
    tmp_dir = configuration["tmp_folder_cluster"] if torch.cuda.is_available() else \
        configuration["tmp_folder_local"]
    configuration["tmp_folder"] = tmp_dir

    logging.info("Setting the seed {}".format(configuration["seed"]))
    set_seed(configuration["seed"])

    data_path = Path("{project_path}/data/{dataset}".format(
        project_path=PROJECT_PATH.__str__(),
        dataset=configuration["dataset"]))

    dataset = create_dataset(data_path,
                             dataset_name=configuration["dataset"],
                             is_tune=configuration["tune"],
                             data_cv_strategy=configuration["data_cv_strategy"])
    for fold_id, fold in enumerate(dataset.cv_strategy.folds, 1):
        logging.info(f"Fold: {fold} out of {len(dataset.cv_strategy.folds)}")
        configuration["train_samples"] = len(dataset.cv_strategy.train_indices_[fold])
        configuration["fold"] = fold
        save_dir = configuration[
            "preds_save_dir_cluster"] if torch.cuda.is_available() else \
            configuration["preds_save_dir_local"]
        configuration["save_dir"] = save_dir
        configuration["features_path"] = data_path

        if configuration["mode"] == "tune":
            if configuration["tune_model"] == "bale":
                if configuration["tune_fold_id"] == fold_id:
                    logging.info("Running hyperparameter tuning for BALE")
                    for i in range(50):
                        logging.info(f"Iteration {i}")
                        tune_results_bale = ray_tune_bale(configuration=configuration,
                                                          tune_results=tune_results_bale,
                                                          dataset=dataset)
                    get_best_hyperparameters_bale(tune_results_bale,
                                                  configuration["save_dir"],
                                                  file_name=configuration['project_name'] + f"_best_hyperparameters_bale_{fold_id}.csv")

            if configuration["tune_model"] == "supervised":
                if configuration["tune_fold_id"] == fold_id:
                    logging.info("Running hyperparameter tuning for Brain (supervised)")
                    for i in range(50):
                        logging.info(f"Iteration {i}")
                        tune_results_supervised = ray_tune_supervised(configuration=configuration,
                                                                      tune_results=tune_results_supervised,
                                                                      dataset=dataset)
                    get_best_hyperparameters_supervised(tune_results_supervised,
                                                        configuration["save_dir"],
                                                        file_name=configuration[
                                                                      'project_name'] + f"_best_hyperparameters_supervised_{fold_id}.csv")

        if configuration["mode"] == "train":
            for model_type in ["clip", "mse", "control"]:
                configuration["model_type"] = model_type
                run_bale_train(configuration=configuration, dataset=dataset)
            configuration["model_type"] = "supervised"
            run_train_supervised(configuration=configuration, dataset=dataset)


if __name__ == '__main__':
    set_logging(PROJECT_PATH.__str__())
    logging.info("Device is {}".format(DEVICE))

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default='nemo',
        help="Dataset: 'nemo' or 'aomic'."
    )
    parser.add_argument(
        "-cv",
        "--cross_validation",
        type=str,
        default='subject-dependent',
        help="CV strategy: 'subject-dependent' or 'subject-independent'."
    )

    parser.add_argument(
        "-mode",
        "--mode",
        type=str,
        default='train',
        help="Mode: 'train' or 'tune'."
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=1,
        help="Seed."
    )

    parser.add_argument(
        "-tf",
        "--tune_fold_id",
        type=int,
        default=1,
        help="Fold to use for hyperparameter tuning."
    )

    parser.add_argument(
        "-tm",
        "--tune_model",
        type=str,
        default="bale",
        help="Model to tune: 'bale' or 'supervised'."
    )

    parser.add_argument(
        "-local",
        "--local",
        action="store_true",
        help="Local mode."
    )

    args = parser.parse_args()
    pprint(args)

    config_path = os.path.join(PROJECT_PATH / "scripts/config.yaml")

    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)
    config["project_name"] += "-" + args.dataset
    config["project_name"] += "-" + args.cross_validation
    config["dataset"] = args.dataset
    config["data_cv_strategy"] = args.cross_validation
    config["mode"] = args.mode
    config["seed"] = args.seed
    config["tune"] = False if config['mode'] == "train" else True
    config["local"] = args.local
    config["tune_fold_id"] = args.tune_fold_id
    config["tune_model"] = args.tune_model

    d = config["dataset"]
    s = config["data_cv_strategy"].split('-')[1]

    config["bale_epochs"] = config[f"bale_epochs_{d}_{s}"]
    config["supervised_epochs"] = config[f"supervised_epochs_{d}_{s}"]

    config["bale_batch_size"] = config[f"bale_batch_size_{d}_{s}"]
    config["supervised_batch_size"] = config[f"supervised_batch_size_{d}_{s}"]

    config["bale_projection_dim"] = config[f"bale_projection_dim_{d}_{s}"]
    config["supervised_projection_dim"] = config[f"supervised_projection_dim_{d}_{s}"]

    config["bale_mlp_h_dim"] = config[f"bale_mlp_h_dim_{d}_{s}"]
    config["supervised_mlp_h_dim"] = config[f"supervised_mlp_h_dim_{d}_{s}"]

    run(config)
