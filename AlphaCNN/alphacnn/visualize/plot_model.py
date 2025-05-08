import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from alphacnn.response.receptive_fields import split_strf, plot_srf, plot_trf
from alphacnn.visualize.plot_stimulus import plot_frame


def plot_bc_rfs(model, fps=None):
    rfs = model.get_layer('BC-RFs').get_weights()[0].squeeze()

    if rfs[0].ndim == 3:
        return plot_bc_3d_rfs(rfs.T, fps=fps)
    else:
        return plot_bc_2d_rfs(rfs.T)


def plot_bc_3d_rfs(rfs, fps=None):
    assert rfs[0].ndim == 3

    splits = [split_strf(rf) for rf in rfs]
    vabsmax = np.max([np.max(np.abs(srf)) for srf, trf in splits])

    fig, axs = plt.subplots(2, len(rfs), figsize=(2 * len(rfs), 2.5), gridspec_kw=dict(height_ratios=(2, 1)))
    for ax_col, (srf, trf) in zip(axs.T, splits):
        plot_srf(srf, ax=ax_col[0], vabsmax=vabsmax, cb=True)
        ax_col[0].set(xticks=[], yticks=[])
        plot_trf(trf, ax=ax_col[1], fps=fps)
        sns.despine(ax=ax_col[1])

    plt.tight_layout()
    return fig, axs


def plot_bc_2d_rfs(rfs):
    assert rfs[0].ndim == 2

    vabsmax = np.max(np.abs(rfs))

    fig, axs = plt.subplots(1, 2, figsize=(3 * rfs.shape[0], 3))
    for ax, rf in zip(axs, rfs):
        srf, trf = split_strf(rf)
        plot_srf(srf, ax=ax, vabsmax=vabsmax, cb=True)

    plt.tight_layout()
    return fig, axs


def plot_rgc_synapses(model):
    from alphacnn.response.receptive_fields import plot_srf

    names = [lay.name.split('-')[1] for lay in model.layers if 'RGC' in lay.name and 'input' in lay.name]
    n_bc_layers = model.get_layer(f'RGC-{names[0]}-input').get_weights()[0].squeeze().shape[-1]

    fig, axs = plt.subplots(len(names), 2, figsize=(3 * n_bc_layers, 3 * len(names)))

    for name, ax_row in zip(names, axs):
        synaptic_weights = model.get_layer(f'RGC-{name}-input').get_weights()[0].squeeze()

        for ax, w in zip(ax_row, synaptic_weights.T):
            ax.set_title(name)
            plot_srf(w[:, :].T, ax=ax, vabsmax=np.max(np.abs(w)), cb=True)

    plt.tight_layout()
    return fig, axs


def plot_simulation(results, n_parts=None, i_part=0):
    if 'Stimulus' in results:
        if results['Stimulus'].ndim != 4:
            raise ValueError(f"Stimulus must be 4D, but is {results['Stimulus'].ndim}D")

    if n_parts is None:
        for k, v in results.items():
            if v.ndim > 2:
                if 'poisson' in k.lower() and v.shape[0] == 1:
                    n_parts = v.shape[1]
                else:
                    n_parts = v.shape[0]
                break
            else:
                print('skip', k, v.ndim)

    if n_parts is None:
        raise ValueError(f"n_parts is None")

    if i_part >= n_parts:
        raise ValueError(f"i_part must be less than {n_parts}, but is {i_part}")

    # There are some layers (Poisson) with redundant first dimensions
    results_i = {k: v[i_part] for k, v in results.items()}
    plot_single_forward_pass(results_i)


def plot_single_forward_pass(results):
    rgc_names = [name.split('-')[1] for name in results.keys() if 'RGC' in name and 'input' in name]

    fig, axs = plt.subplots(4 + len(rgc_names), 2, figsize=(5, 15))

    if 'Stimulus' in results or 'StimulusRescaled' in results:
        stimulus = results['StimulusRescaled'] if 'StimulusRescaled' in results else results['Stimulus']

        if stimulus.ndim != 3:
            raise ValueError(f"Stimulus of single forwardpass must be 3D, but is {stimulus.ndim}D")
        ax = axs[0, 0]
        ax.set(title=f'stimulus first frame')
        plot_frame(frame=stimulus[:, :, 0], ax=ax, add_colorbar=True, vmin=np.min(stimulus), vmax=np.max(stimulus))

        ax = axs[0, 1]
        ax.set(title=f'stimulus last frame')
        plot_frame(frame=stimulus[:, :, -1], ax=ax, add_colorbar=True, vmin=np.min(stimulus), vmax=np.max(stimulus))

    for j, name in enumerate(['RFs', 'noise', 'rect']):
        for i, surround in enumerate(['WS', 'SS']):
            ax = axs[j + 1, i]
            ax.set(title=f'BC-{name} {surround}')
            if f'BC-{name}' not in results:
                continue
            plot_frame(frame=results[f'BC-{name}'][:, :, i], ax=ax, add_colorbar=True,
                       vmin=0. if name == 'rect' else np.min(results[f'BC-{name}']),
                       vmax=np.max(results[f'BC-{name}']))

    for j, side in enumerate(rgc_names):
        for i, name in enumerate(['input', 'noise']):
            ax = axs[j + 4, i]
            ax.set(title=f'RGC-{side}-{name}')
            if f'RGC-{side}-{name}' not in results:
                continue
            plot_frame(frame=results[f'RGC-{side}-{name}'][0:, :], ax=ax, add_colorbar=True,
                       vmin=np.min(results[f'RGC-{side}-{name}']),
                       vmax=np.max(results[f'RGC-{side}-{name}']))

    plt.tight_layout()
