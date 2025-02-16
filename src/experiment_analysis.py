import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.lines as lines
current_working_directory = os.getcwd()
SAVEDIR = os.path.dirname(current_working_directory)


def calculate_rmse(true_labels, pred_labels):
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    mse = np.mean((true_labels - pred_labels) ** 2)  #
    rmse = np.sqrt(mse)  #
    return rmse


def exact_accuracy(true_labels, pred_labels):
    success_count = 0
    for true, pred in zip(true_labels, pred_labels):
        if true == pred:
            success_count += 1
    accuracy = success_count / len(true_labels)
    return accuracy


def adjacency_accuracy(true_labels, pred_labels):
    success_count = 0
    for true, pred in zip(true_labels, pred_labels):
        if abs(true - pred) <= 1:
            success_count += 1
    accuracy = success_count / len(true_labels)
    return accuracy


if __name__ == '__main__':
    rmse_matrix_dict = dict()
    adjacc_matrix_dict = dict()
    exacc_matrix_dict = dict()

    for expId in ["baseline", "47"]:
        for lang in ["ar", "en", "fr", "hi", "ru"]:
            df_agent_expert = pd.read_csv(f"LLMGeneration/README/{lang}/{lang}_few-shot_{expId}.csv")
            label_all = [1 for _ in range(len(df_agent_expert))] + [2 for _ in range(len(df_agent_expert))] + [3 for _ in
                                                                                                               range(
                                                                                                                   len(df_agent_expert))]
            pred_all = df_agent_expert["pred_1"].tolist() + df_agent_expert["pred_2"].tolist() + df_agent_expert[
                "pred_3"].tolist()
            correlation, p_value = spearmanr(label_all, pred_all)
            autometric_dict = {
                "Cor: ": round(correlation, 2),
                "p_value": round(p_value, 5),
                "AdjAcc": round(adjacency_accuracy(label_all, pred_all) * 100, 2),
                "ExAcc": round(exact_accuracy(label_all, pred_all) * 100, 2),
                "RMSE": round(calculate_rmse(label_all, pred_all), 2),
            }
            print(lang, autometric_dict)
            rmse_matrix, adjacc_matrix, exacc_matrix = [], [], []
            for source in [4, 5, 6]:
                tmp_rmse, tmp_adjacc, tmp_exacc = [], [], []
                df_ = df_agent_expert.loc[df_agent_expert["Rating"] == source]
                for target in [1, 2, 3]:
                    pred_ = df_[f"pred_{target}"].tolist()
                    label_ = [target for _ in range(len(pred_))]
                    tmp_rmse.append(round(calculate_rmse(label_, pred_), 2))
                    tmp_adjacc.append(round(adjacency_accuracy(label_, pred_) * 100, 2))
                    tmp_exacc.append(round(exact_accuracy(label_, pred_) * 100, 2))
                rmse_matrix.append(tmp_rmse)
                adjacc_matrix.append(tmp_adjacc)
                exacc_matrix.append(tmp_exacc)
            rmse_matrix_dict[f"{lang}_{expId}"] = np.array(rmse_matrix)
            adjacc_matrix_dict[f"{lang}_{expId}"] = np.array(adjacc_matrix)
            exacc_matrix_dict[f"{lang}_{expId}"] = np.array(exacc_matrix)
            print(np.array(adjacc_matrix))
            print()

    lang2desc = {
        "ar": "Arabic", "en": "English", "fr": "French", "hi": "Hindi", "ru": "Russian"
    }

    """ RMSE """
    for lang in ["ar", "en", "fr", "hi", "ru"]:
        baseline = rmse_matrix_dict[f"{lang}_baseline"]
        agent_expert = rmse_matrix_dict[f"{lang}_47"]

        vmin = min(baseline.min(), agent_expert.min())
        vmax = max(baseline.max(), agent_expert.max())

        fig, axes = plt.subplots(1, 2, figsize = (10, 5), dpi = 1000, gridspec_kw = {'width_ratios': [1, 1.25]})

        x_labels = ['A1', 'A2', 'B1']
        y_labels = ['B2', 'C1', 'C2']

        cmap = "Blues"
        sns.heatmap(baseline, annot = True, ax = axes[0], cbar = False, cmap = cmap, xticklabels = x_labels,
                    yticklabels = y_labels, vmin = vmin, vmax = vmax, annot_kws = {'size': 15}, fmt = '.2f')
        axes[0].set_title('One-step LLM', fontsize = 12, fontweight = 'bold')
        axes[0].set_xlabel('Target', fontsize = 12)
        axes[0].set_ylabel('Source', fontsize = 12)

        heatmap = sns.heatmap(agent_expert, annot = True, ax = axes[1], cmap = cmap, xticklabels = x_labels,
                              yticklabels = y_labels, vmin = vmin, vmax = vmax, annot_kws = {'size': 15}, fmt = '.2f')
        axes[1].set_title('DP-planner + Semantic CoT', fontsize = 12, fontweight = 'bold')
        axes[1].set_xlabel('Target', fontsize = 12)
        # axes[1].set_ylabel('Source', fontsize = 12)

        cbar = heatmap.collections[0].colorbar

        cbar.set_label('RMSE', rotation = 90, labelpad = 15)

        fig.suptitle(f'README-{lang2desc[lang]}', fontsize = 15, fontweight = 'bold')

        plt.subplots_adjust(wspace = 10)  #
        plt.tight_layout()
        plt.subplots_adjust(top = 0.88)

        ax0_pos = axes[0].get_position()
        ax1_pos = axes[1].get_position()
        x_line = (ax0_pos.x1 + ax1_pos.x0) / 2  #
        y_start = ax0_pos.y0
        y_end = ax0_pos.y1

        line = lines.Line2D([x_line - 0.008, x_line - 0.008], [y_start, y_end],
                            transform = fig.transFigure, color = '#999999', linewidth = 2.0, linestyle = "--")
        fig.add_artist(line)

        plt.savefig(os.path.join(SAVEDIR, f"img/README/{lang}/README-{lang}_dp-cot_rmse.jpg"), dpi = 1000,
                    bbox_inches = 'tight')
        plt.savefig(os.path.join(SAVEDIR, f"img/README/{lang}/README-{lang}_dp-cot_rmse.svg"), dpi = 1000,
                    bbox_inches = 'tight', format = 'svg')
        plt.show()
        plt.close()

    """ AdjAcc """
    for lang in ["ar", "en", "fr", "hi", "ru"]:
        baseline = adjacc_matrix_dict[f"{lang}_baseline"]
        agent_expert = adjacc_matrix_dict[f"{lang}_47"]

        vmin = min(baseline.min(), agent_expert.min())
        vmax = max(baseline.max(), agent_expert.max())

        fig, axes = plt.subplots(1, 2, figsize = (10, 5), dpi = 1000, gridspec_kw = {'width_ratios': [1, 1.25]})

        x_labels = ['A1', 'A2', 'B1']
        y_labels = ['B2', 'C1', 'C2']

        cmap = "OrRd"
        sns.heatmap(baseline, annot = True, ax = axes[0], cbar = False, cmap = cmap, xticklabels = x_labels,
                    yticklabels = y_labels, vmin = vmin, vmax = vmax, annot_kws = {'size': 15}, fmt = '.2f')
        axes[0].set_title('One-step LLM', fontsize = 12, fontweight = 'bold')
        axes[0].set_xlabel('Target', fontsize = 12)
        axes[0].set_ylabel('Source', fontsize = 12)

        heatmap = sns.heatmap(agent_expert, annot = True, ax = axes[1], cmap = cmap, xticklabels = x_labels,
                              yticklabels = y_labels, vmin = vmin, vmax = vmax, annot_kws = {'size': 15}, fmt = '.2f')
        axes[1].set_title('DP-planner + Semantic CoT', fontsize = 12, fontweight = 'bold')
        axes[1].set_xlabel('Target', fontsize = 12)
        # axes[1].set_ylabel('Source', fontsize = 12)

        cbar = heatmap.collections[0].colorbar

        cbar.set_label('Adjacency Accuracy', rotation = 90, labelpad = 15)


        fig.suptitle(f'README-{lang2desc[lang]}', fontsize = 15, fontweight = 'bold')

        plt.subplots_adjust(wspace = 10)  #

        plt.tight_layout()
        plt.subplots_adjust(top = 0.88)

        ax0_pos = axes[0].get_position()
        ax1_pos = axes[1].get_position()
        x_line = (ax0_pos.x1 + ax1_pos.x0) / 2  #
        y_start = ax0_pos.y0
        y_end = ax0_pos.y1

        line = lines.Line2D([x_line - 0.008, x_line - 0.008], [y_start, y_end],
                            transform = fig.transFigure, color = '#999999', linewidth = 2.0, linestyle = "--")
        fig.add_artist(line)

        plt.savefig(os.path.join(SAVEDIR, f"img/README/{lang}/README-{lang}_dp-cot_adjacc.jpg"), dpi = 1000,
                    bbox_inches = 'tight')
        plt.savefig(os.path.join(SAVEDIR, f"img/README/{lang}/README-{lang}_dp-cot_adjacc.svg"), dpi = 1000,
                    bbox_inches = 'tight', format = 'svg')
        plt.show()
        plt.close()

    """ ExaAcc """
    for lang in ["ar", "en", "fr", "hi", "ru"]:
        baseline = exacc_matrix_dict[f"{lang}_baseline"]
        agent_expert = exacc_matrix_dict[f"{lang}_47"]

        vmin = min(baseline.min(), agent_expert.min())
        vmax = max(baseline.max(), agent_expert.max())

        fig, axes = plt.subplots(1, 2, figsize = (10, 5), dpi = 1000, gridspec_kw = {'width_ratios': [1, 1.25]})

        x_labels = ['A1', 'A2', 'B1']
        y_labels = ['B2', 'C1', 'C2']

        cmap = "OrRd"
        sns.heatmap(baseline, annot = True, ax = axes[0], cbar = False, cmap = cmap, xticklabels = x_labels,
                    yticklabels = y_labels, vmin = vmin, vmax = vmax, annot_kws = {'size': 15}, fmt = '.2f')
        axes[0].set_title('One-step LLM', fontsize = 12, fontweight = 'bold')
        axes[0].set_xlabel('Target', fontsize = 12)
        axes[0].set_ylabel('Source', fontsize = 12)

        heatmap = sns.heatmap(agent_expert, annot = True, ax = axes[1], cmap = cmap, xticklabels = x_labels,
                              yticklabels = y_labels, vmin = vmin, vmax = vmax, annot_kws = {'size': 15}, fmt = '.2f')
        axes[1].set_title('DP-planner + Semantic CoT', fontsize = 12, fontweight = 'bold')
        axes[1].set_xlabel('Target', fontsize = 12)
        # axes[1].set_ylabel('Source', fontsize = 12)

        cbar = heatmap.collections[0].colorbar

        cbar.set_label('Exact Accuracy', rotation = 90, labelpad = 15)

        fig.suptitle(f'README-{lang2desc[lang]}', fontsize = 15, fontweight = 'bold')

        plt.subplots_adjust(wspace = 10)  #

        plt.tight_layout()

        plt.subplots_adjust(top = 0.88)

        ax0_pos = axes[0].get_position()
        ax1_pos = axes[1].get_position()
        x_line = (ax0_pos.x1 + ax1_pos.x0) / 2  #
        y_start = ax0_pos.y0
        y_end = ax0_pos.y1

        line = lines.Line2D([x_line - 0.008, x_line - 0.008], [y_start, y_end],
                            transform = fig.transFigure, color = '#999999', linewidth = 2.0, linestyle = "--")
        fig.add_artist(line)

        plt.savefig(os.path.join(SAVEDIR, f"img/README/{lang}/README-{lang}_dp-cot_exaacc.jpg"), dpi = 1000,
                    bbox_inches = 'tight')
        plt.savefig(os.path.join(SAVEDIR, f"img/README/{lang}/README-{lang}_dp-cot_exaacc.svg"), dpi = 1000,
                    bbox_inches = 'tight', format = 'svg')
        plt.show()
        plt.close()
