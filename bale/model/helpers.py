import logging
from typing import List, Optional

import pandas as pd
import torch
import torch.nn.functional as functional
import torchmetrics
from torch import nn, Tensor
from torchmetrics import MetricCollection


def create_metrics(metric_list: List[str], num_classes: Optional[int],
                   task: str = 'binary') -> MetricCollection:
    """

    :param metric_list:
    :param num_classes:
    :return:
    """
    allowed_metrics = ['precision', 'recall', 'accuracy', "aucroc", "f1"]

    for metric in metric_list:
        if metric not in allowed_metrics:
            raise ValueError(
                f"{metric} is not allowed. Please choose 'precision', 'recall', 'f1_score', 'accuracy'"
            )
    metric_dict = {
        'accuracy':
            torchmetrics.Accuracy(task=task,
                                  num_classes=num_classes,
                                  top_k=1),
        'precision':
            torchmetrics.Precision(task=task,
                                   num_classes=num_classes),
        'recall':
            torchmetrics.Recall(task=task,
                                num_classes=num_classes),
        'aucroc':
            torchmetrics.AUROC(task=task,
                               num_classes=num_classes),
        'f1':
            torchmetrics.F1Score(task=task,
                                 num_classes=num_classes)
    }
    metrics = [metric_dict[name] for name in metric_list]
    return MetricCollection(metrics)


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


def log_all_parameters(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        logging.info("Parameter: {}.".format(n))


def count_trainable_parameters(model: nn.Module, log_non_trainable_params: bool = True) -> None:
    """
    Counts the number of trainable parameters.
    :param model:
    :param log_trainable_params:
    :return:
    """
    if log_non_trainable_params:
        for n, p in model.named_parameters():
            if not p.requires_grad:
                logging.info("{} is not trainable.".format(n))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("In total {} trainable parameters".format(n_params))


def get_brainy_predictions(brain_embeddings: torch.Tensor,
                           image_embeddings: torch.Tensor,
                           labels: torch.Tensor,
                           top_k: int = 1) -> (Tensor, Tensor):
    """
    Calculates a Brainy accuracy. If a match, then True, otherwise - False.
    :param top_k:
    :param brain_embeddings: Brain embeddings.
    :param image_embeddings: Image embeddings.
    :param labels:
    :return: An accuracy score.
    """

    brain_embeddings_n = brain_embeddings / brain_embeddings.norm(dim=1, keepdim=True)
    image_embeddings_n = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)

    true_labels = []
    predictions = []

    for idx in range(brain_embeddings_n.shape[0]):
        # For each brain signal:

        brain_embedding_n = brain_embeddings_n[idx, ...].reshape(1, -1)
        dot_similarity = brain_embedding_n @ image_embeddings_n.T
        similarity = (100.0 * dot_similarity).softmax(dim=-1)

        # Get the image that is predicted as the most similar:
        values, indices = similarity[0].topk(top_k)
        for value, index in zip(values, indices):
            true_labels.append(labels[idx].item())
            predictions.append(labels[index].item())
    return predictions, true_labels


def get_rankings(brain_embeddings: torch.Tensor,
                 image_embeddings: torch.Tensor,
                 labels: torch.Tensor,
                 img_idx: torch.Tensor,
                 participant_list: list,
                 seed_run: int,
                 fold: int) -> (Tensor, Tensor):
    """
    Calculates a Brainy accuracy. If a match, then True, otherwise - False.
    :param brain_embeddings: Brain embeddings.
    :param image_embeddings: Image embeddings.
    :param labels:
    :param img_idx: Image indices.
    :param participant_list: A list of participants.
    :return: A data frame containing a ranking list of all matched images.
    """

    brain_embeddings_n = brain_embeddings / brain_embeddings.norm(dim=1, keepdim=True)
    image_embeddings_n = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)

    true_image_indices = []
    predicted_image_indices = []
    true_labels = []
    predicted_labels = []
    ranks = []
    queries = []
    participants = []
    similarity_scores = []

    for idx in range(brain_embeddings_n.shape[0]):
        brain_embedding_n = brain_embeddings_n[idx, ...].reshape(1, -1)
        dot_similarity = brain_embedding_n @ image_embeddings_n.T
        similarity = (100.0 * dot_similarity).softmax(dim=-1)

        values, indices = similarity[0].topk(img_idx.size()[0])
        temp_predicted_image_indices = []
        minus_rank = 0
        for rank, (value, index) in enumerate(zip(values, indices), 1):
            if img_idx[index].item() in temp_predicted_image_indices:
                minus_rank += 1
                continue
            true_labels.append(labels[idx].item())
            predicted_labels.append(labels[index].item())
            true_image_indices.append(img_idx[idx].item())
            predicted_image_indices.append(img_idx[index].item())
            ranks.append(rank - minus_rank)
            queries.append(img_idx[idx].item())
            participants.append(participant_list[idx])
            similarity_scores.append(value.item())

            temp_predicted_image_indices.append(img_idx[index].item())

    df = pd.DataFrame(list(
        zip(queries, ranks, true_image_indices, predicted_image_indices, true_labels, predicted_labels, participants, similarity_scores)),
        columns=['query', 'rank', 'true_image', 'predicted_image', 'true_label', 'predicted_label',
                 'participant', 'similarity_score'])

    df['seed'] = seed_run
    df['fold'] = fold

    return df
