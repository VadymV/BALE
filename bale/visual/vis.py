import copy
import logging
import time
from itertools import combinations
from typing import Literal, List

import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.stats as stats
import seaborn as sns
import torch
from adjustText import adjust_text
from matplotlib import pyplot as plt, transforms, ticker
from matplotlib.ticker import FormatStrFormatter
from torchmetrics.classification import BinaryPrecision, BinaryRecall

from bale.dataset.dataset import DATASET_NEMO_NAME, DATASET_AOMIC_NAME
from bale.helpers.misc import run_permutation_test

sns.set_theme()
sns.set_theme(font_scale=1.4)


def create_thresholds_data(raw_data_: pd.DataFrame,
                           modality: Literal["preds_brain", "preds_image"],
                           fraction: float = 0.5) -> list[pd.DataFrame]:
    """
    Creates thresholds data.
    :param raw_data: Predictions.
    :param modality: Modality.
    :param fraction: Fraction.
    :return: List of data frames.
    """
    raw_data = copy.deepcopy(raw_data_)
    runs = raw_data.groupby(['fold', 'seed'])

    data = []
    for run in runs:
        preds_brainclas = run[1]["preds_brainclas"].tolist()
        preds_modality = run[1][modality].tolist()
        target = torch.LongTensor(run[1]["target"].tolist())
        group = run[0]

        run_result = {"group": [], "p": [], "r": [], "t": [], "n": []}
        for t in np.arange(0.0, 1.0, 0.01):
            preds_modality_t = [1 if x > t else 0 for x in preds_modality]
            if np.mean(preds_modality_t) < fraction:
                preds = [int(int(i) and int(k)) for i, k in zip(preds_brainclas, preds_modality_t)]
                preds = torch.LongTensor(preds)

                precision = BinaryPrecision()
                recall = BinaryRecall()

                run_result["p"].append(precision(preds, target).item())
                run_result["r"].append(recall(preds, target).item())
                run_result["t"].append(t)
                run_result["n"].append(fraction)
                run_result["group"].append(group)
                break

        run_data = pd.DataFrame(data=run_result)
        data.append(run_data)
    return data


def create_thresholds_data_all(raw_data: pd.DataFrame,
                               modality: Literal["preds_brain", "preds_image"]) -> list[pd.DataFrame]:
    runs = raw_data.groupby(['fold', 'seed'])

    data = []
    for run in runs:
        preds_brainclas = run[1]["preds_brainclas"].tolist()
        preds_modality = run[1][modality].tolist()
        target = torch.LongTensor(run[1]["target"].tolist())

        run_result = {"p": [], "r": [], "t": [], "n": []}
        for t in np.arange(0.0, 1.0, 0.01):
            preds_modality_t = [1 if x > t else 0 for x in preds_modality]
            preds = [int(int(i) and int(k)) for i, k in zip(preds_brainclas, preds_modality_t)]
            preds = torch.LongTensor(preds)

            precision = BinaryPrecision()
            recall = BinaryRecall()
            n = torch.sum(preds).item() / preds.shape[0]

            run_result["p"].append(precision(preds, target).item())
            run_result["r"].append(recall(preds, target).item())
            run_result["t"].append(t)
            run_result["n"].append(n)

        run_data = pd.DataFrame(data=run_result)
        data.append(run_data)
    return data


def plot_thresholds(prepared_data: list[pd.DataFrame],
                    figures_path: str):
    thresholds = prepared_data[0].shape[0]
    cardinality = len(prepared_data)
    threshold_values = prepared_data[0].t.values

    shape = (thresholds, cardinality)
    precision = np.zeros(shape)
    recall = np.zeros(shape)
    n_flagged = np.zeros(shape)

    for i, df in enumerate(prepared_data):
        precision[:, i] = df.p.values
        recall[:, i] = df.r.values
        n_flagged[:, i] = df.n.values

    precision_ = np.mean(precision, axis=1)
    recall_ = np.mean(recall, axis=1)
    n_flagged_ = np.mean(n_flagged, axis=1)

    def get_ci(data):
        lower = []
        upper = []
        for r in range(data.shape[0]):
            d = data[r, :]
            interval = st.t.interval(alpha=0.90, df=len(d) - 1, loc=np.mean(d), scale=st.sem(d))
            lower.append(interval[0])
            upper.append(interval[1])
        lower = np.array(lower)
        upper = np.array(upper)

        return lower, upper

    precision_interval = get_ci(precision)
    recall_interval = get_ci(recall)
    n_flagged_interval = get_ci(n_flagged)

    a4_dims = (15, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    fig.tight_layout()

    plt.plot(threshold_values, precision_, color='blue', label='precision')
    plt.plot(threshold_values, recall_, color='green', label='recall')
    plt.plot(threshold_values, n_flagged_, color='red', label='flagged')

    plt.fill_between(threshold_values, precision_interval[0],
                     precision_interval[1], color='blue',
                     alpha=0.2)
    plt.fill_between(threshold_values, recall_interval[0],
                     recall_interval[1], color='green',
                     alpha=0.2)
    plt.fill_between(threshold_values, n_flagged_interval[0],
                     n_flagged_interval[1], color='red',
                     alpha=0.2)

    plt.ylim([-0.05, 0.8])
    plt.xlabel('Threshold')
    plt.ylabel('Metric value')
    plt.legend(loc="lower left", labels=["Precision", "Recall", "Fraction of predicted positives"])
    # plt.legend().get_texts()[0].set_text('Precision')

    # ax = plt.twinx()
    # ax.plot(threshold_values, n_flagged_, color='red', label='flagged')
    # ax.fill_between(threshold_values, n_flagged_interval[0],
    #                 n_flagged_interval[1], color='red',
    #                 alpha=0.2)

    # ax.legend(loc="lower left", labels=["Fraction of predicted positives"])
    # plt.ylabel('Fraction of predicted positives', rotation=270, labelpad=30)
    # plt.ylim([-0.05, 0.8])
    # plt.grid()

    # plt.title("Zero-shot image prediction at different thresholds.")
    plt.savefig(figures_path,
                format="pdf",
                bbox_inches="tight")
    plt.close()


def plot_bar(save_path, x, y, x_label, y_label, hue_label=None,
             y_lim=None, title=None, average: List[int] = None, colours: List[str] = None, hue=None,
             hue_categorical=False,
             legend_title=None, rotate=True, order=None, errorbar=None,
             x_ticks_fontsize: int = None):
    a4_dims = (15, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)

    sns.barplot(ax=ax, x=x, y=y, errorbar=errorbar, hue=hue, order=order, dodge=True,
                palette="YlOrBr" if hue is not None and not hue_categorical else None,
                color='b' if hue is None and hue_categorical else None)

    if hue is not None and not hue_categorical:
        norm = plt.Normalize(hue.min(), hue.max())
        sm = plt.cm.ScalarMappable(cmap="YlOrBr", norm=norm)
        cbar = ax.figure.colorbar(sm, ax=ax)
        cbar.set_label(hue_label, rotation=270)
        cbar.ax.get_yaxis().labelpad = 30
        ax.get_legend().remove()
    ax.set_xlabel(xlabel=x_label)
    ax.set_ylabel(ylabel=y_label)

    if rotate:
        for item in ax.get_xticklabels():
            item.set_rotation(45)
    else:
        ax.set_xlabel(xlabel=x_label, labelpad=90)
    if y_lim is not None:
        plt.ylim(y_lim)
    if average is not None:
        if len(average) != len(colours):
            logging.error("The number of average values and colours must match.")
        for i in range(len(average)):
            plt.axhline(y=average[i], color=colours[i], linestyle='--')
            trans = transforms.blended_transform_factory(
                ax.get_yticklabels()[0].get_transform(), ax.transData)
            ax.text(0, average[i], "{:.2f}".format(average[i]), color=colours[i], transform=trans,
                    ha="right", va="center")
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    if legend_title is not None:
        ax.legend(title=legend_title, loc="lower right")

    # counts = []
    # for category in ax.get_xticklabels():
    #     count = x[x == category._text].shape[0]
    #     counts.append(count)
    #
    # for i, v in enumerate(counts):
    #     ax.text(i, 0.4, str(v), ha='center', color='black')

    if x_ticks_fontsize is not None:
        plt.xticks(fontsize=x_ticks_fontsize)
    plt.title(title)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")

    plt.close()


def set_text(ax_, x_, y_, category_, plot_text_size_):
    x = copy.deepcopy(x_)
    y = copy.deepcopy(y_)
    category = copy.deepcopy(category_)

    texts = []
    for x_, y_, s in zip(x, y, category):
        texts.append(
            ax_.text(x_, y_, s, fontsize=plot_text_size_)
        )
    adjust_text(texts,
                ax=ax_,
                x=x,
                y=y,
                expand_text=(1.01, 1.2),
                expand_points=(1.01, 1.2),
                force_text=(0.01, 0.5),
                force_points=(0.01, 0.5),
                arrowprops=dict(arrowstyle='-', color='gray', alpha=.5))


def plot_2d_representation(figures_path: str, file: str):
    sns.set(font_scale=3.4, style="ticks")
    a4_dims = (15, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    fig.tight_layout()
    data = np.load(figures_path / "{}.npy".format(file))
    sns.heatmap(data)

    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=360)
    ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=360)

    for id, tick in enumerate(ax.get_xticklabels()):
        if id % 3 != 0:
            tick.set_visible(False)

    for id, tick in enumerate(ax.get_yticklabels()):
        if id % 2 != 0:
            tick.set_visible(False)

    ax.set(xlabel="Time domain")
    ax.set(ylabel="Electrode")

    plt.savefig(figures_path / "{}.pdf".format(file),
                format="pdf",
                bbox_inches="tight")
    plt.close()


def plot_correlation_results(data: pd.DataFrame,
                             figures_path: str,
                             x_label: str,
                             y_label: str,
                             labels: list[str],
                             average: bool = False,
                             plot_text: bool = False,
                             plot_text_col: str = None,
                             plot_text_size: int = 14):
    a4_dims = (15, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    fig.tight_layout()

    d_pos = data[data["target"] == 1]
    d_neg = data[data["target"] == 0]
    tau_pos, p_value_pos = stats.kendalltau(d_pos["x"],
                                            d_pos["y"])
    tau_neg, p_value_neg = stats.kendalltau(d_neg["x"],
                                            d_neg["y"])

    if plot_text:
        assert plot_text_col is not None
        ax = sns.scatterplot(ax=ax, data=data, x="x", y="y", s=20)
        set_text(ax, data["x"].tolist(), data["y"].tolist(), data[plot_text_col].tolist(), plot_text_size)

    ax = sns.regplot(ax=ax,
                     x="x",
                     y="y",
                     data=d_pos,
                     label=labels[0],
                     color='b')
    sns.regplot(ax=ax,
                x="x",
                y="y",
                data=d_neg,
                label=labels[1],
                color='g')

    plt.ylim([-0.1, 1.1])
    plt.text(data["x"].mean() + 0.2, -0.06, 'Kendall’s Tau = {v:.2f}, p < {p:.2e}.'.format(v=tau_pos, p=p_value_pos),
             fontstyle='italic', color='b')
    plt.text(data["x"].min(), -0.06, 'Kendall’s Tau = {v:.2f}, p < {p:.2e}.'.format(v=tau_neg, p=p_value_neg),
             fontstyle='italic', color='g')

    if average:
        plt.axvline(data["x"].mean(), color='r', ls="--")

    plt.legend(loc="upper right", labels=labels)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    # plt.grid()

    plt.savefig(figures_path,
                format="pdf",
                bbox_inches="tight")
    plt.close()


def plot_correlation_model(figures_path: str,
                           x,
                           y,
                           x_label: str,
                           y_label: str,
                           xlim,
                           text_col=None,
                           x_pos_text=None,
                           y_pos_text=None,
                           ):
    a4_dims = (15, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    fig.tight_layout()

    ax = sns.scatterplot(ax=ax, x=x, y=y)
    if text_col is not None:
        set_text(ax, x, y, text_col, plot_text_size_=20)
    ax = sns.regplot(ax=ax,
                     x=x,
                     y=y,
                     color='r',
                     scatter_kws={'s': 30})

    plt.ylim([0.2, 0.97])
    plt.xlim(xlim)
    plt.axvline(np.mean(x), color='b', ls="--")
    plt.axhline(np.mean(y), color='y', ls="--")
    tau_pos, p_value_pos = stats.kendalltau(x, y)
    plt.text(x_pos_text, y_pos_text, 'Kendall’s Tau = {v:.2f}, p < {p:.2e}.'.format(v=tau_pos, p=p_value_pos),
             fontstyle='italic')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(figures_path,
                format="pdf",
                bbox_inches="tight")
    plt.close()


def plot_rankings(data: pd.DataFrame,
                  save_path: str,
                  depth: int,
                  strategy: str,
                  ):
    # Prepared data
    if len(data["model"].unique()) > 4:
        raise ValueError("Can process at most four models.")
    prepared_d = data.drop(columns=["seed"], errors="ignore")
    prepared_d = prepared_d[prepared_d["strategy"] == strategy]
    prepared_d = prepared_d[prepared_d["depth"] <= depth]
    prepared_d = prepared_d.melt(id_vars=["depth", "dataset", "model", "strategy"], value_name="Value",
                                 var_name="Metric")

    # Set up figure
    sns.set_theme(font_scale=1.9, style="ticks")
    a4_dims = (15, 8.27)
    fig, axs = plt.subplots(figsize=a4_dims, nrows=2, ncols=4)
    fig.tight_layout()

    # Plot (0, 0):
    d = prepared_d[(prepared_d["dataset"] == "nemo") & (prepared_d["Metric"] == "NDCG")]
    sns.pointplot(
        data=d, x="depth", y="Value", hue="model", ax=axs[0, 0], errorbar="se", markers=["*", "^", "v", "."],
        markersize=15, linestyles=["-", "--", "-.", ":"]
    )

    # Plot (0, 1):
    d = prepared_d[(prepared_d["dataset"] == "nemo") & (prepared_d["Metric"] == "P")]
    sns.pointplot(
        data=d, x="depth", y="Value", hue="model", ax=axs[0, 1], errorbar="se", markers=["*", "^", "v", "."],
        markersize=15, linestyles=["-", "--", "-.", ":"]
    )

    # Plot (0, 2):
    d = prepared_d[(prepared_d["dataset"] == "nemo") & (prepared_d["Metric"] == "MAP")]
    sns.pointplot(
        data=d, x="depth", y="Value", hue="model", ax=axs[0, 2], errorbar="se", markers=["*", "^", "v", "."],
        markersize=15, linestyles=["-", "--", "-.", ":"]
    )

    # Plot (0, 3):
    d = prepared_d[(prepared_d["dataset"] == "nemo") & (prepared_d["Metric"] == "R")]
    sns.pointplot(
        data=d, x="depth", y="Value", hue="model", ax=axs[0, 3], errorbar="se", markers=["*", "^", "v", "."],
        markersize=15, linestyles=["-", "--", "-.", ":"]
    )

    # Plot (1, 0):
    d = prepared_d[(prepared_d["dataset"] == "aomic") & (prepared_d["Metric"] == "NDCG")]
    sns.pointplot(
        data=d, x="depth", y="Value", hue="model", ax=axs[1, 0], errorbar="se", markers=["*", "^", "v", "."],
        markersize=15, linestyles=["-", "--", "-.", ":"]
    )

    # Plot (1, 1):
    d = prepared_d[(prepared_d["dataset"] == "aomic") & (prepared_d["Metric"] == "P")]
    sns.pointplot(
        data=d, x="depth", y="Value", hue="model", ax=axs[1, 1], errorbar="se", markers=["*", "^", "v", "."],
        markersize=15, linestyles=["-", "--", "-.", ":"]
    )

    # Plot (1, 2):
    d = prepared_d[(prepared_d["dataset"] == "aomic") & (prepared_d["Metric"] == "MAP")]
    sns.pointplot(
        data=d, x="depth", y="Value", hue="model", ax=axs[1, 2], errorbar="se", markers=["*", "^", "v", "."],
        markersize=15, linestyles=["-", "--", "-.", ":"]
    )

    # Plot (1, 3):
    d = prepared_d[(prepared_d["dataset"] == "aomic") & (prepared_d["Metric"] == "R")]
    sns.pointplot(
        data=d, x="depth", y="Value", hue="model", ax=axs[1, 3], errorbar="se", markers=["*", "^", "v", "."],
        markersize=15, linestyles=["-", "--", "-.", ":"]
    )

    for idx, ax in enumerate(fig.get_axes(), 1):
        ax.grid()
        ax.set_ylim([0.0, 1.03])
        ax.set(ylabel=None)

        if idx > 1:
            ax.get_legend().remove()

        if idx <= 4:
            ax.set(xlabel=None)
            ax.set(xticklabels=[])

        if idx == 1:
            ax.set_title("NDCG")
        if idx == 2:
            ax.set_title("Precision")
        if idx == 3:
            ax.set_title("MAP")
        if idx == 4:
            ax.set_title("Recall")

        if idx == 5:
            ax.set_xlabel("NDCG@K")
        if idx == 6:
            ax.set_xlabel("Precision@K")
        if idx == 7:
            ax.set_xlabel("MAP@K")
        if idx == 8:
            ax.set_xlabel("Recall@K")

        if idx != 1 and idx != 5:
            ax.set(yticklabels=[])

        if idx == 4:
            ax2 = ax.twinx()
            ax2.set(yticklabels=[])
            ax2.get_yaxis().set_ticks([])
            ax2.set_ylabel('NEMO', labelpad=10)
        if idx == 8:
            ax2 = ax.twinx()
            ax2.set(yticklabels=[])
            ax2.get_yaxis().set_ticks([])
            ax2.set_ylabel('AOMIC', labelpad=10)

        ax.set(yticks=np.arange(0.0, 1.03, 0.15))

        xticks = ax.xaxis.get_major_ticks()
        for i in range(len(xticks)):
            if i == 6 or i == 8 or i == 9:
                xticks[i].set_visible(False)

    axs[0, 0].legend(bbox_to_anchor=(5.61, 0), loc='center right', title="Model")

    brain_supervised_name = 'Brain-supervised'
    control_name = 'BALE-control'
    mse_name = 'BALE-MSE'
    clip_name = 'BALE'

    for entry in axs[0, 0].get_legend().get_texts():
        if 'clip' in entry.get_text():
            entry.set_text(clip_name)
        elif 'mse' in entry.get_text():
            entry.set_text(mse_name)
        elif 'supervised' in entry.get_text():
            entry.set_text(brain_supervised_name)
        elif 'control' in entry.get_text():
            entry.set_text(control_name)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.savefig(save_path,
                format="pdf",
                bbox_inches="tight")
    plt.close()


def plot_group_category_predictions(figures_path: str,
                                    category_data_attract: pd.DataFrame,
                                    category_data_nemo: pd.DataFrame):
    a4_dims = (10, 4.27)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=a4_dims)
    fig.tight_layout()
    y_lim = (-1.0, 1.0)

    # AX1
    x = category_data_attract["group"]
    y = category_data_attract["fraction"]
    sns.boxplot(ax=ax1,
                x=x,
                y=y,
                hue=category_data_attract["hue"],
                order=category_data_attract.groupby(["group"])["fraction"].median().sort_values(
                    ascending=False).index.values,
                dodge=False,
                palette="YlOrBr",
                color=None)

    norm = plt.Normalize(category_data_attract["hue"].min(), category_data_attract["hue"].max())
    sm = plt.cm.ScalarMappable(cmap="YlOrBr", norm=norm)
    cbar = ax1.figure.colorbar(sm, ax=ax1)
    cbar.set_label("Image rating", rotation=270)
    cbar.ax.get_yaxis().labelpad = 20
    ax1.get_legend().remove()
    ax1.set_ylim(y_lim)

    # AX2
    x = category_data_nemo["group"]
    y = category_data_nemo["fraction"]
    sns.boxplot(ax=ax2,
                x=x,
                y=y,
                hue=category_data_nemo["hue"],
                order=category_data_nemo.groupby(["group"])["fraction"].median().sort_values(
                    ascending=False).index.values,
                dodge=False,
                palette="YlOrBr",
                color=None)

    norm = plt.Normalize(category_data_nemo["hue"].min(), category_data_nemo["hue"].max())
    sm = plt.cm.ScalarMappable(cmap="YlOrBr", norm=norm)
    cbar = ax2.figure.colorbar(sm, ax=ax2)
    cbar.set_label("Valence", rotation=270)
    cbar.ax.get_yaxis().labelpad = 20
    ax2.get_legend().remove()
    ax2.set_ylim(y_lim)

    # # AX4
    # x = user_data_nemo["group"]
    # y = user_data_nemo["fraction"]
    # sns.boxplot(ax=ax4,
    #             x=x,
    #             y=y,
    #             hue=user_data_nemo["hue"],
    #             order=user_data_nemo.groupby(["group"])["rating"].mean().sort_values(
    #                 ascending=False).index.values,
    #             dodge=False,
    #             palette="YlOrBr",
    #             color=None)
    #
    # norm = plt.Normalize(user_data_nemo["hue"].min(), user_data_nemo["hue"].max())
    # sm = plt.cm.ScalarMappable(cmap="YlOrBr", norm=norm)
    # cbar = ax4.figure.colorbar(sm, ax=ax4)
    # cbar.set_label("Valence", rotation=270)
    # cbar.ax.get_yaxis().labelpad = 20
    # ax4.get_legend().remove()
    # ax4.set_ylim(y_lim)

    # x1, x2 = 0, 1
    # y, h, col = 1.05, 0.02, 'k'
    # ax4.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col, clip_on=False)
    # ax4.text((x1 + x2) * .5, y + h, nemo_p, ha='center', va='bottom', color=col)

    for idx, ax in enumerate(fig.get_axes(), 1):
        # ax.label_outer()

        if idx == 1:
            ax.set_ylabel("Difference")
            ax.set_title("Facial attractiveness")
            ax.set(xlabel=None)
            ticks_loc = ax.get_xticks()
            ax.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
            ax.set_xlabel("Image category")
            ax.get_xaxis().labelpad = 22
            time.sleep(1)
        if idx == 2:
            ax.set_title("NEMO")
            ticks_loc = ax.get_xticks()
            ax.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
            ax.set_xlabel("Image category")
            ax.set(yticklabels=[])
            ax.set(ylabel=None)
            time.sleep(1)

    plt.subplots_adjust(hspace=0.6)
    plt.subplots_adjust(wspace=0.2)

    # # Plot most common attractive images:
    # for d in images.keys():
    #     for p in images[d]:
    #         if p == "top":
    #             y_pos = 0.75
    #         else:
    #             y_pos = 0.15
    #         if "pos" in d:
    #             x_pos = -0.25
    #         else:
    #             x_pos = 0.75
    #         if "attract" in d:
    #             zoom = 0.15
    #         else:
    #             zoom = 0.05
    #         for img in images[d][p]:
    #             imagebox = OffsetImage(img, zoom=zoom)
    #             ab = AnnotationBbox(imagebox, (x_pos, y_pos), frameon=False)
    #             if "attract" in d:
    #                 ax3.add_artist(ab)
    #                 # ax3.scatter(5, 700, s=20000, marker='o', color='red', facecolors='none')
    #             else:
    #                 ax4.add_artist(ab)
    #             x_pos += 0.25
    #             y_pos += 0.05

    plt.savefig(figures_path,
                format="pdf",
                bbox_inches="tight")
    plt.close()


def plot_group_class_predictions(figures_path: str,
                                 d_dependent: pd.DataFrame,
                                 d_independent: pd.DataFrame,
                                 x_label: str):
    """
    Plots group class predictions.
    :param figures_path: Path to save the figure
    :param x_label: Label for x axis
    """
    sns.set_theme(font_scale=2)
    if x_label == "Valence":
        a4_dims = (10.5, 6.5)
        xticklabels_rotation = 30
        y_label_left = " Fraction of correctly classified \n samples"
        y_label_right_dependent ="Participant-\n   dependent    "
        y_label_right_independent = "Participant-\n   independent   "
        h_space = 0.06
        y_label_right_offset = 0.01
        y_label_left_offset = -0.01
    else:
        a4_dims = (10.5, 9.5)
        xticklabels_rotation = 90
        y_label_left = "Fraction of correctly classified samples"
        y_label_right_dependent = "Participant-dependent"
        y_label_right_independent = "Participant-independent"
        h_space = 0.03
        y_label_right_offset = 0.0
        y_label_left_offset = 0.0
    fig, axes = plt.subplots(2, 4, sharex=True, sharey=True, figsize=a4_dims)
    fig.tight_layout()
    fig.text(0.5, -0.1, x_label, ha='center')
    fig.text(0.0 + y_label_left_offset, 0.2, y_label_left, ha='center', rotation=90)
    fig.text(1.0, 0.57 + y_label_right_offset, y_label_right_dependent, ha='center', rotation=270)
    fig.text(1.0, 0.12 + y_label_right_offset, y_label_right_independent, ha='center', rotation=270)
    y_lim = (-0.03, 1.03)

    if d_dependent["group"].unique().shape[0] > 2:
        mapping = {'anger': 1, 'contempt': 2, 'neutral': 3, 'joy': 4, 'pride': 5}
    else:
        mapping = {'negative': 1, 'valence': 2}
    d_dependent.loc[:, 'order'] = d_dependent['group'].map(mapping)
    d_independent.loc[:, 'order'] = d_independent['group'].map(mapping)

    axes[0, 0] = sns.boxplot(ax=axes[0, 0],
                             data=d_dependent,
                             x="group",
                             y="fraction_supervised",
                             hue=d_dependent["dataset"],
                             dodge=False,
                             width=0.5,
                             order=d_dependent.groupby(["group"])["order"].mean().sort_values(
                                 ascending=True).index.values,
                             palette="husl")

    axes[0, 1] = sns.boxplot(ax=axes[0, 1],
                             data=d_dependent,
                             x="group",
                             y="fraction_clip",
                             hue=d_dependent["dataset"],
                             dodge=False,
                             width=0.5,
                             order=d_dependent.groupby(["group"])["order"].mean().sort_values(
                                 ascending=True).index.values,
                             palette="husl")

    axes[0, 2] = sns.boxplot(ax=axes[0, 2],
                             data=d_dependent,
                             x="group",
                             y="fraction_mse",
                             hue=d_dependent["dataset"],
                             dodge=False,
                             width=0.5,
                             order=d_dependent.groupby(["group"])[
                                 "order"].mean().sort_values(
                                 ascending=True).index.values,
                             palette="husl")

    axes[0, 3] = sns.boxplot(ax=axes[0, 3],
                             data=d_dependent,
                             x="group",
                             y="fraction_control",
                             hue=d_dependent["dataset"],
                             dodge=False,
                             width=0.5,
                             order=d_dependent.groupby(["group"])[
                                 "order"].mean().sort_values(
                                 ascending=True).index.values,
                             palette="husl")

    axes[1, 0] = sns.boxplot(ax=axes[1, 0],
                             data=d_independent,
                             x="group",
                             y="fraction_supervised",
                             hue=d_independent["dataset"],
                             dodge=False,
                             width=0.5,
                             order=d_independent.groupby(["group"])["order"].mean().sort_values(
                                 ascending=True).index.values,
                             palette="husl")

    axes[1, 1] = sns.boxplot(ax=axes[1, 1],
                             data=d_independent,
                             x="group",
                             y="fraction_clip",
                             hue=d_independent["dataset"],
                             dodge=False,
                             width=0.5,
                             order=d_independent.groupby(["group"])["order"].mean().sort_values(
                                 ascending=True).index.values,
                             palette="husl")

    axes[1, 2] = sns.boxplot(ax=axes[1, 2],
                             data=d_independent,
                             x="group",
                             y="fraction_mse",
                             hue=d_independent["dataset"],
                             dodge=False,
                             width=0.5,
                             order=d_independent.groupby(["group"])[
                                 "order"].mean().sort_values(
                                 ascending=True).index.values,
                             palette="husl")

    axes[1, 3] = sns.boxplot(ax=axes[1, 3],
                             data=d_independent,
                             x="group",
                             y="fraction_control",
                             hue=d_independent["dataset"],
                             dodge=False,
                             width=0.5,
                             order=d_independent.groupby(["group"])[
                                 "order"].mean().sort_values(
                                 ascending=True).index.values,
                             palette="husl")

    for idx, ax in enumerate(fig.get_axes(), 1):
        ax.set_ylim(y_lim)
        ax.legend().remove()
        if idx > 4:
            ticks_loc = ax.get_xticks()
            ax.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=xticklabels_rotation)
            ax.get_xaxis().labelpad = 10
        x_labels = [tick.get_text() for tick in axes[1, 1].get_xticklabels()]
        label_to_index = {label: idx for idx, label in enumerate(x_labels)}
        ax.label_outer()

        if idx == 1:
            ax.set_ylabel('')
            ax.set_title('Brain-supervised')
            ax.set_xlabel('')
            metric = 'fraction_supervised'
            if len(d_dependent["group"].unique()) == 2:
                pairs = combinations(['negative', 'positive'], 2)
            else:
                pairs = combinations(['joy', 'contempt', 'neutral', 'pride', 'anger'], 2)

        if idx == 2:
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_title('BALE')
            metric = 'fraction_clip'
            if len(d_dependent["group"].unique()) == 2:
                pairs = combinations(['negative', 'positive'], 2)
            else:
                pairs = combinations(['joy', 'contempt', 'neutral', 'pride', 'anger'], 2)

        if idx == 3:
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_title('MSE')
            metric = 'fraction_mse'
            if len(d_dependent["group"].unique()) == 2:
                pairs = combinations(['negative', 'positive'], 2)
            else:
                pairs = combinations(['joy', 'contempt', 'neutral', 'pride', 'anger'], 2)

        if idx == 4:
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_title('Control')
            metric = 'fraction_control'
            if len(d_dependent["group"].unique()) == 2:
                pairs = combinations(['negative', 'positive'], 2)
            else:
                pairs = combinations(['joy', 'contempt', 'neutral', 'pride', 'anger'], 2)

        if idx == 5:
            ax.set_ylabel("")
            ax.set_xlabel('')
            metric = 'fraction_supervised'
            if len(d_independent["group"].unique()) == 2:
                pairs = combinations(['negative', 'positive'], 2)
            else:
                pairs = combinations(['joy', 'contempt', 'neutral', 'pride', 'anger'], 2)

        if idx == 6:
            ax.set_ylabel('')
            ax.set_xlabel('')
            metric = 'fraction_clip'
            if len(d_independent["group"].unique()) == 2:
                pairs = combinations(['negative', 'positive'], 2)
            else:
                pairs = combinations(['joy', 'contempt', 'neutral', 'pride', 'anger'], 2)

        if idx == 7:
            ax.set_ylabel('')
            ax.set_xlabel('')
            metric = 'fraction_mse'
            if len(d_independent["group"].unique()) == 2:
                pairs = combinations(['negative', 'positive'], 2)
            else:
                pairs = combinations(['joy', 'contempt', 'neutral', 'pride', 'anger'], 2)

        if idx == 8:
            ax.set_ylabel('')
            ax.set_xlabel('')
            metric = 'fraction_control'
            if len(d_independent["group"].unique()) == 2:
                pairs = combinations(['negative', 'positive'], 2)
            else:
                pairs = combinations(['joy', 'contempt', 'neutral', 'pride', 'anger'], 2)

        y = 0.65
        for pair in pairs:
            logging.info(f"Pair is {pair}")
            if idx == 1 or idx == 2:
                p = run_permutation_test(d_dependent[d_dependent['group'] == pair[0]][metric],
                                         d_dependent[d_dependent['group'] == pair[1]][metric],
                                         sign='either')
            else:
                p = run_permutation_test(d_independent[d_independent['group'] == pair[0]][metric],
                                         d_independent[d_independent['group'] == pair[1]][metric],
                                         sign='either')
            if p != 'ns':
                x1, x2 = label_to_index[pair[0]], label_to_index[pair[1]]
                h, col = 0.02, 'k'
                ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col, clip_on=False)
                ax.text((x1 + x2) * .5, y - h, p, ha='center', va='bottom', color=col)
                y += 0.06

    plt.subplots_adjust(wspace=0.03, hspace=h_space)
    plt.savefig(figures_path,
                format="pdf",
                bbox_inches="tight")
    plt.close()


def plot_query_predictions(figures_path: str,
                           plot_data: pd.DataFrame):
    """
    Plots group class predictions.
    :param figures_path: Path to save the figure
    :param plot_data: Data for plotting
    """
    a4_dims = (10, 4.27)
    fig, ax = plt.subplots(1, 1, figsize=a4_dims)
    fig.tight_layout()
    y_lim = (-0.03, 0.5)

    ax = sns.boxplot(ax=ax,
                     data=plot_data,
                     x="query",
                     y="precision",
                     # dodge=False,
                     hue='group',
                     order=plot_data.groupby(["query"])["precision"].median().sort_values(ascending=False).index.values,
                     palette=None)

    ax.legend(bbox_to_anchor=(2, 0), loc='center right', title="Model")

    ax.set_ylim(y_lim)
    ticks_loc = ax.get_xticks()
    ax.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.get_xaxis().labelpad = 10
    ax.set_ylabel("Precision@1")
    ax.set_xlabel("Query (brain recording)")

    plt.savefig(figures_path,
                format="pdf",
                bbox_inches="tight")
    plt.close()


def plot_corr_models_facet_grid(data: (pd.DataFrame, pd.DataFrame),
                                figures_path: str):
    sns.set_theme(font_scale=2)
    a4_dims = (21, 9.2)
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, sharex=True, sharey=True, figsize=a4_dims)
    fig.tight_layout()
    fig.text(0.5, 0.0, "Balanced accuracy (Participant-dependent)", ha='center')
    fig.text(0.0, 0.15, "Balanced accuracy (Participant-independent)", ha='center', rotation=90)
    fig.text(1.0, 0.70, "NEMO", ha='center', rotation=270)
    fig.text(1.0, 0.25, "AOMIC", ha='center', rotation=270)
    point_size = 30

    d_nemo, d_aomic = data

    for idx, ax in enumerate(fig.get_axes(), 1):
        ax.legend().remove()
        ax.set_ylim((0.02, 0.99))
        ax.set_xlim((-0.02, 0.9999))
        if idx == 1:
            d = d_nemo[d_nemo["model"] == "supervised"]
            ax.set_title("Brain-supervised")
        if idx == 2:
            d = d_nemo[d_nemo["model"] == "clip"]
            ax.set_title("BALE")
        if idx == 3:
            d = d_nemo[d_nemo["model"] == "mse"]
            ax.set_title("MSE")
        if idx == 4:
            d = d_nemo[d_nemo["model"] == "control"]
            ax.set_title("Control")
        if idx == 5:
            d = d_aomic[d_aomic["model"] == "supervised"]
        if idx == 6:
            d = d_aomic[d_aomic["model"] == "clip"]
        if idx == 7:
            d = d_aomic[d_aomic["model"] == "mse"]
        if idx == 8:
            d = d_aomic[d_aomic["model"] == "control"]

        x = d["acc_dependent"]
        y = d["acc_independent"]
        ax.axvline(np.mean(x), color='b', ls="--")
        ax.axhline(np.mean(y), color='y', ls="--")
        ax = sns.regplot(ax=ax,
                         x=x,
                         y=y,
                         color='r',
                         scatter_kws={'s': point_size})

        ax.label_outer()
        ax.set_xlabel('')
        ax.set_ylabel('')

        if idx <= 4:
            text_position = (0.0, 0.8)
        else:
            text_position = (0.3, 0.8)



        tau_pos, p_value_pos = stats.kendalltau(x, y)
        ax.text(text_position[0], text_position[1],
                'Kendall’s Tau = {v:.2f}, \np < {p:.2e}'.format(v=tau_pos, p=p_value_pos),
                fontstyle='italic', fontsize=18)

    plt.subplots_adjust(hspace=0.03, wspace=0.01)
    plt.savefig(figures_path,
                format="pdf",
                bbox_inches="tight")
    plt.close()


def plot_histogram(figures_path: str,
                   data: pd.DataFrame,
                   x: str,
                   hue: str,
                   subplots: tuple) -> None:
    sns.set(font_scale=0.5)
    a4_dims = (15, 8.27)
    fig, _ = plt.subplots(subplots[0], subplots[1], figsize=a4_dims)
    fig.tight_layout()

    for idx, ax in enumerate(fig.get_axes(), 0):
        if idx < 24:
            ax.set_xlabel('')
        else:
            ax.set_xlabel("Unit activations")
        ax.legend(title='Valence', loc='upper left', labels=['Negative valence', 'Positive valence'])
        # ax.set_xlim((-3, 3))
        d = data[data["unit"] == idx]
        sns.histplot(ax=ax, data=d, x=x, hue=hue, kde=True)

    # plt.legend(title='Valence', loc='upper left', labels=['Negative valence', 'Positive valence'])
    plt.savefig(figures_path,
                format="pdf",
                bbox_inches="tight")
    plt.close()

# run1 = pd.read_csv("C:\\Users\\vcx763\\PycharmProjects\\Brainy\\brainy_run1.csv")
# run2 = pd.read_csv("C:\\Users\\vcx763\\PycharmProjects\\Brainy\\brainy_run2.csv")
# run1 = run1.drop(columns=['Name'])
# run2 = run2.drop(columns=['Name'])
# data = run1.merge(run2, how='inner', on='Tags')
# data.set_index("Tags")
#
# for columns in [["brainy_test_acc", "brain_test_auroc", "image_test_auroc"],
#                 ["brainy_train_acc", "brain_train_auroc", "image_train_auroc"]]:
#
#     d = data[columns]
#     # your plot setup
#     fig, ax = plt.subplots()
#
#     d.plot.bar(rot=0, ax=ax)
#
#     plt.minorticks_on()
#     ax.tick_params(axis='x',which='minor',bottom='off')
#     ax.set_xlabel("Participant")
#     ax.set_ylabel("Accuracy")
#     ax.set_title(("Accuracy per participant"), fontsize=20)
#
#     # use axhline
#     colors = ["blue", "orange", "green"]
#     for i in range(len(columns)):
#         mean = d[columns[i]].mean()
#         ax.axhline(mean, color=colors[i])
#     plt.show()
