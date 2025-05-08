import numpy as np
from matplotlib import pyplot as plt


def plot_pred_vs_true(ax, pos, pos_pred):
    ax.set(xlabel='x', ylabel='y')
    if pos.size > 0:
        ax.scatter(pos[:, 0], pos[:, 1], c='k', label='true', s=2)
        ax.scatter(pos_pred[:, 0], pos_pred[:, 1], c='r', label='pred', s=2)
        ax.plot(np.stack([pos_pred[:, 0], pos[:, 0]]), np.stack([pos_pred[:, 1], pos[:, 1]]), c='r', alpha=0.5)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))


def plot_errors(*true_pred_name, ax):
    loss_list = []
    ylabels = []
    for true, pred, name in true_pred_name:
        finite_idxs = np.all(np.isfinite(true), axis=1)
        if true[finite_idxs].size > 0:
            loss_list.append(np.sum((true[finite_idxs, :] - pred[finite_idxs, :]) ** 2, axis=1) ** 0.5)
            ylabels.append(name)
    ax.boxplot(loss_list, labels=ylabels)


def update_lims(pos, lims):
    lims[0] = np.nanmin([lims[0], np.nanmin(pos[:, 0])])
    lims[1] = np.nanmax([lims[1], np.nanmax(pos[:, 0])])

    lims[2] = np.nanmin([lims[2], np.nanmin(pos[:, 1])])
    lims[3] = np.nanmax([lims[3], np.nanmax(pos[:, 1])])

    return lims


def expand_lims(lims, p=0.05):
    dx = lims[1] - lims[0]
    dy = lims[3] - lims[2]
    lims = [lims[0] - p * dx, lims[1] + p * dx, lims[2] - p * dy, lims[3] + p * dy]
    return lims


def plot_decoding(*true_pred_name, xlim=None, ylim=None):
    fig, axs = plt.subplots(1, len(true_pred_name) + 1, figsize=(12, 3))

    lims = [np.nan, np.nan, np.nan, np.nan]

    for ax, (pos_true, pos_pred, name) in zip(axs, true_pred_name):
        plot_pred_vs_true(ax=ax, pos=pos_true, pos_pred=pos_pred)

        lims = update_lims(pos_true, lims=lims)
        lims = update_lims(pos_pred, lims=lims)

        ax.set_aspect('equal', 'box')
        ax.set_title(name)

    lims = expand_lims(lims, p=0.05)
    for ax, _ in zip(axs, true_pred_name):
        ax.set(xlim=lims[0:2], ylim=lims[2:4])

    ax = axs[-1]
    plot_errors(*true_pred_name, ax=ax)

    plt.tight_layout()

    for ax in axs[:2]:
        ax.set(xlim=xlim, ylim=ylim)

    return fig, axs


def plot_confusion_matrix(p_train, p_pred_train, p_test, p_pred_test):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    fig, axs = plt.subplots(1, 2, figsize=(8, 8), sharex='all', sharey='all')

    for ax, (p, p_pred) in zip(axs, [(p_train, p_pred_train), (p_test, p_pred_test)]):
        cm = confusion_matrix(p, p_pred > 0.5)

        # Calculate evaluation metrics
        accuracy = np.sum(p.diag(cm)) / np.sum(cm)
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        f1_score = 2 * (precision * recall) / (precision + recall)

        # Print evaluation metrics
        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1-Score: ", f1_score)
        print()

        # Plot confusion matrix
        sns.heatmap(cm / np.sum(cm), annot=True, fmt=".1%", cmap="Greens", cbar=False, square=True, ax=ax)
        ax.set(title="Confusion Matrix", xlabel="Predicted", ylabel="True")

    plt.tight_layout()
    plt.show()
