import glob
import glob
import logging
import os
from pathlib import Path
from typing import Literal, List

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon, ttest_rel, shapiro
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torchmetrics.retrieval import RetrievalMRR, RetrievalPrecision, \
    RetrievalRecall, RetrievalNormalizedDCG, RetrievalMAP, RetrievalAUROC
from torchmetrics.retrieval.base import RetrievalMetric
from tqdm import tqdm

from bale import PROJECT_PATH
from bale.dataset.dataset import DATASET_NEMO_NAME, DATASET_AOMIC_NAME
from bale.helpers.misc import set_seed, set_logging, run_permutation_test
from bale.visual.vis import plot_group_class_predictions, \
    plot_corr_models_facet_grid

_DATASETS = ["nemo", "aomic"]
_MODELS = ["clip", "mse", "control", "supervised"]
_EVALUATION_STRATEGIES = ["dependent", "independent"]
_RANKING_LIMIT = 5


def _stat_test(a, b, test: Literal["wilcoxon", "ttest_rel"]):
    if test == "ttest_rel":
        print("Checking normality assumption (ttest_rel): {}".format(
            shapiro(a - b)))
        t, p = ttest_rel(a, b)
    elif test == "wilcoxon":
        t, p = wilcoxon(x=a - b, y=None)
    else:
        raise ValueError

    return t, p


def read_rankings(data_path: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(data_path, '**', 'rankings-*'),
                      recursive=True)
    logging.info("{} ranking files are found".format(len(files)))
    rankings = pd.DataFrame()

    for csv_file in files:
        if (csv_file.endswith("-43") or csv_file.endswith("-194") or \
            csv_file.endswith("-98")) and "dependent" in csv_file:
            logging.info("Skipping {}".format(csv_file))
            continue
        df = pd.read_csv(csv_file)
        file_name = Path(csv_file).stem
        df["model"] = file_name.split("-")[1]
        df["dataset"] = file_name.split("-")[3]
        if file_name.split("-")[3] == DATASET_NEMO_NAME:
            df["group"] = df["true_label"].map({0: 'negative', 1: 'positive'})
        if file_name.split("-")[3] == DATASET_AOMIC_NAME:
            df["group"] = df["true_label"].map(
                {0: 'pride', 1: 'anger', 2: 'joy', 3: 'neutral', 4: 'contempt'})
        df["strategy"] = file_name.split("-")[5]
        df = df.reset_index(drop=True)
        rankings = pd.concat([rankings, df])

    rankings = rankings.reset_index(drop=True)
    size = rankings.groupby(['dataset', 'strategy']).size().reset_index(
        name='row_count')
    logging.info(size)
    return rankings


def prepare_data_class_predictions(data: pd.DataFrame,
                                   grouping_variables: List[str]):
    runs = data.groupby(grouping_variables)
    result = {"fraction_clip": [],
              "fraction_supervised": [],
              "fraction_mse": [],
              "fraction_control": [],
              'group': [],
              'dataset': []}

    for _, d in runs:
        y_hat_clip = d[d["model"] == 'clip']['predicted_label'].values
        y_hat_supervised = d[d["model"] == 'supervised'][
            'predicted_label'].values
        y_hat_mse = d[d["model"] == 'mse']['predicted_label'].values
        y_hat_control = d[d["model"] == 'control']['predicted_label'].values

        y_clip = d[d["model"] == 'clip']['true_label'].values
        y_supervised = d[d["model"] == 'supervised']['true_label'].values
        y_mse = d[d["model"] == 'mse']['true_label'].values
        y_control = d[d["model"] == 'control']['true_label'].values

        result["fraction_clip"].append(accuracy_score(y_clip, y_hat_clip))
        result["fraction_supervised"].append(
            accuracy_score(y_supervised, y_hat_supervised))
        result["fraction_mse"].append(accuracy_score(y_mse, y_hat_mse))
        result["fraction_control"].append(
            accuracy_score(y_control, y_hat_control))
        result["group"].append(d['group'].unique().item())
        result["dataset"].append(d["dataset"].iloc[0])

    result = pd.DataFrame.from_dict(result)

    return result


def prepare_data_correlation(data: pd.DataFrame):
    runs = data.groupby(['model', 'dataset', 'query'])
    result = {"acc_dependent": [], 'acc_independent': [], 'dataset': [],
              'model': []}

    for _, d in runs:
        # Dependent
        y_hat_dependent = d[d["strategy"] == 'dependent'][
            'predicted_label'].values
        y_dependent = d[d["strategy"] == 'dependent']['true_label'].values
        result["acc_dependent"].append(
            accuracy_score(y_dependent, y_hat_dependent))
        # Independent
        y_hat_independent = d[d["strategy"] == 'independent'][
            'predicted_label'].values
        y_independent = d[d["strategy"] == 'independent']['true_label'].values
        result["acc_independent"].append(
            accuracy_score(y_independent, y_hat_independent))

        result["dataset"].append(d["dataset"].iloc[0])
        result["model"].append(d["model"].iloc[0])

    result = pd.DataFrame.from_dict(result)

    return result


def calculate_retrieval_metric(preds: torch.Tensor, target: torch.Tensor,
                               indexes: torch.Tensor, f: RetrievalMetric):
    retrieval_metric = f(preds=preds, target=target, indexes=indexes).item()

    return retrieval_metric


def compute_affective_decoding_results(data: pd.DataFrame) -> pd.DataFrame:
    results = {'strategy': [], 'dataset': [], 'model': [], 'score': []}
    for strategy in _EVALUATION_STRATEGIES:
        for dataset in _DATASETS:
            for model in _MODELS:
                d = data[(data["dataset"] == dataset) &
                         (data["strategy"] == strategy) &
                         (data["model"] == model)]
                print(
                    "\nAffective decoding results for {} {} {}.".format(
                        dataset, strategy, model))
                runs = d.groupby(['fold', 'seed'])
                scores = []
                for run in runs:
                    score = balanced_accuracy_score(run[1]['true_label'].values,
                                                    run[1][
                                                        'predicted_label'].values)
                    scores.append(score)
                    results['strategy'].append(strategy)
                    results['dataset'].append(dataset)
                    results['model'].append(model)
                    results['score'].append(score)

                print("Accuracy: {:.3f} +/- {:.3f} SE".format(
                    sum(scores) / len(scores),
                    np.std(scores) / np.sqrt(len(scores))))

    return pd.DataFrame(results)


def compute_statistical_significance(data: pd.DataFrame):
    for strategy in _EVALUATION_STRATEGIES:
        for dataset in _DATASETS:
            d = data[
                (data["dataset"] == dataset) &
                (data["strategy"] == strategy)]
            print(
                "\nPermutation test for {} {}: {} and {}".format(
                    dataset,
                    strategy,
                    "clip",
                    "mse"))
            run_permutation_test(a=d[d["model"] == "clip"]['score'],
                                 b=d[d["model"] == "mse"]['score'])
            print(
                "\nPermutation test for {} {}: {} and {}".format(
                    dataset,
                    strategy,
                    "clip",
                    "control"))
            run_permutation_test(a=d[d["model"] == "clip"]['score'],
                                 b=d[d["model"] == "control"]['score'])

            print(
                "\nPermutation test for {} {}: {} and {}".format(
                    dataset,
                    strategy,
                    "supervised",
                    "clip"))
            run_permutation_test(a=d[d["model"] == "supervised"]['score'],
                                 b=d[d["model"] == "clip"]['score'])


def compute_effective_ranking(data: pd.DataFrame, file_path, save: bool):
    ranking_results = []
    for strategy in _EVALUATION_STRATEGIES:
        for dataset in _DATASETS:
            for model in _MODELS:
                if model == 'supervised':
                    continue
                print(f"Affective ranking for {dataset}, {strategy}, {model}")
                rank_data = data[(data["dataset"] == dataset)
                                 & (data["model"] == model)
                                 & (data["strategy"] == strategy)
                                 & (data["rank"] <= _RANKING_LIMIT)]
                group_columns = ['seed', 'fold', 'participant', 'query']
                runs = rank_data.groupby(group_columns)
                retrieval_metrics = {'seed': [], 'depth': [], 'MRR': [],
                                     'P': [],
                                     'R': [], 'NDCG': [], 'MAP': [], 'AUC': []}

                for _, run in tqdm(enumerate(runs), total=runs.ngroups):
                    d = run[1]
                    if d.shape[0] != _RANKING_LIMIT:
                        raise ValueError("Must be {} results per query".format(
                            _RANKING_LIMIT))

                    # Retrieval result is correct when the predicted label is the same as the true label:
                    d['target'] = np.where(
                        d['true_label'] == d['predicted_label'], True,
                        False)

                    preds = torch.tensor(d["similarity_score"].values,
                                         dtype=torch.float64)
                    target = torch.tensor(d["target"].values)
                    indexes = torch.zeros_like(target, dtype=torch.long)

                    for depth in range(1, _RANKING_LIMIT + 1):
                        # Metrics:
                        mrr = RetrievalMRR(top_k=depth)
                        p = RetrievalPrecision(top_k=depth)
                        map = RetrievalMAP(top_k=depth)
                        r = RetrievalRecall(top_k=depth)
                        g = RetrievalNormalizedDCG(top_k=depth)
                        auc = RetrievalAUROC(top_k=depth)

                        mean_reciprocal_rank = calculate_retrieval_metric(
                            preds=preds,
                            target=target,
                            indexes=indexes,
                            f=mrr)
                        precision = calculate_retrieval_metric(preds=preds,
                                                               target=target,
                                                               indexes=indexes,
                                                               f=p)
                        map_score = calculate_retrieval_metric(preds=preds,
                                                               target=target,
                                                               indexes=indexes,
                                                               f=map)
                        recall = calculate_retrieval_metric(preds=preds,
                                                            target=target,
                                                            indexes=indexes,
                                                            f=r)
                        ndcg = calculate_retrieval_metric(preds=preds,
                                                          target=target,
                                                          indexes=indexes, f=g)
                        auroc = calculate_retrieval_metric(preds=preds,
                                                           target=target,
                                                           indexes=indexes,
                                                           f=auc)

                        retrieval_metrics['seed'].append(d["seed"].iloc[0])
                        retrieval_metrics['depth'].append(depth)
                        retrieval_metrics['MRR'].append(mean_reciprocal_rank)
                        retrieval_metrics['P'].append(precision)
                        retrieval_metrics['R'].append(recall)
                        retrieval_metrics['NDCG'].append(ndcg)
                        retrieval_metrics['MAP'].append(map_score)
                        retrieval_metrics['AUC'].append(auroc)

                retrieval_metrics = pd.DataFrame(retrieval_metrics)
                retrieval_metrics['dataset'] = dataset
                retrieval_metrics['model'] = model
                retrieval_metrics['strategy'] = strategy

                latex_data = retrieval_metrics.groupby(["depth"]).apply(
                    lambda x: x.mean(numeric_only=True)).round(3)
                latex_data.drop(columns=["depth"], inplace=True)
                metrics = ['P', 'MAP', 'MRR', 'AUC']
                depths = [1, 3, 5]
                values = []
                for metric in metrics:
                    for depth in depths:
                        values.append(latex_data.loc[depth][metric])
                print(" & ".join([str(v) for v in values]))

                ranking_results.append(retrieval_metrics)

    all_rankings = pd.concat(ranking_results)
    if save:
        all_rankings.to_csv(file_path, index=False)

    return all_rankings


if __name__ == '__main__':
    set_logging(PROJECT_PATH.__str__())
    set_seed(1)
    output_folder = "BALE-results"
    only_plots = True
    load = True

    if load:
        predictions = pd.read_csv(
            os.path.join(PROJECT_PATH, output_folder, "predictions.csv"))
        rankings = pd.read_csv(
            os.path.join(PROJECT_PATH, output_folder, "rankings.csv"))
    else:
        rankings = read_rankings(os.path.join(PROJECT_PATH, output_folder))

        predictions = rankings[rankings["rank"] == 1]
        predictions.to_csv(
            os.path.join(PROJECT_PATH, output_folder, "predictions.csv"))

    if not only_plots:
        decoding_results = compute_affective_decoding_results(data=predictions)
        compute_statistical_significance(data=decoding_results)
        _ = compute_effective_ranking(data=rankings,
                                      file_path=os.path.join(PROJECT_PATH,
                                                             output_folder,
                                                             "rankings.csv"),
                                      save=False if load else True)

    # Plot correlation between participant-dependent and participant-independent models across images
    d_nemo = prepare_data_correlation(
        predictions[predictions['dataset'] == DATASET_NEMO_NAME])
    d_aomic = prepare_data_correlation(
        predictions[predictions['dataset'] == DATASET_AOMIC_NAME])
    plot_corr_models_facet_grid(data=(d_nemo, d_aomic),
                                figures_path=os.path.join(PROJECT_PATH,
                                                          output_folder,
                                                          "figures",
                                                          f"correlation.pdf"))

    # Plot class predictions
    d_dependent = prepare_data_class_predictions(
        predictions[predictions['strategy'] == "dependent"],
        grouping_variables=['seed', 'dataset', 'group'])
    d_independent = prepare_data_class_predictions(
        predictions[predictions['strategy'] == "independent"],
        grouping_variables=['seed', 'dataset', 'group'])
    plot_group_class_predictions(
        figures_path=os.path.join(PROJECT_PATH, output_folder, "figures",
                                  f"analysis_fraction_nemo.pdf"),
        d_dependent=d_dependent[(d_dependent['dataset'] == DATASET_NEMO_NAME)],
        d_independent=d_independent[
            (d_independent['dataset'] == DATASET_NEMO_NAME)],
        x_label="Valence")
    plot_group_class_predictions(
        figures_path=os.path.join(PROJECT_PATH, output_folder, "figures",
                                  f"analysis_fraction_aomic.pdf"),
        d_dependent=d_dependent[(d_dependent['dataset'] == DATASET_AOMIC_NAME)],
        d_independent=d_independent[
            (d_independent['dataset'] == DATASET_AOMIC_NAME)],
        x_label="Emotion")
