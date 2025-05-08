from alphacnn.database.encoder_schema import *
from alphacnn.database.encoder_schema import connect_to_database


def plot_simulation(key=None, bc_config=1):
    key = ((RGCSynapticInputs & (key or dict())).proj()).fetch(format='frame').sample(1).reset_index().iloc[0].to_dict()

    bc_key = key.copy()
    bc_key['bc_srf_config_id'] = key['bc_srf_config_id_' + str(bc_config)]
    bc_key['bc_rect_config_id'] = key['bc_rect_config_id_' + str(bc_config)]

    stimulus_id, video, target_pos = (Stimulus & key).fetch1('stimulus_id', 'video', 'target_pos')

    if isinstance(target_pos, dict):
        target_pos = target_pos['xy']

    bc_srf_output = (BCSpatialRFOutput & bc_key).fetch1('bc_srf_output')
    bc_rect_output = (BCRectOutput & bc_key).fetch1('bc_rect_output')
    bc_noise_output = (BCNoiseOutput & bc_key).fetch1('bc_noise_output')
    rgc_synaptic_inputs = (RGCSynapticInputs & key).fetch1('rgc_synaptic_inputs')

    from alphacnn.visualize import plot_stimulus

    n_cols = 4
    n_rows = 5

    fis = np.linspace(0, rgc_synaptic_inputs.shape[0] - 1, n_cols, endpoint=True).astype(int)

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 2.5 * n_rows), sharex='row', sharey='row')

    axs[0, 0].set_ylabel('Video')
    plot_stimulus.plot_video_frames(video, n_rows=1, n_cols=n_cols, axs=axs[0, :], fis=fis)
    plot_stimulus.plot_target_positions(target_pos, axs=axs[0, :], fis=fis, c='darkred')

    axs[1, 0].set_ylabel('BC SRF')
    plot_stimulus.plot_video_frames(bc_srf_output, n_rows=1, n_cols=n_cols, fis=fis, axs=axs[1, :])

    axs[2, 0].set_ylabel('BC Rect')
    plot_stimulus.plot_video_frames(bc_rect_output, n_rows=1, n_cols=n_cols, fis=fis, axs=axs[2, :])

    axs[3, 0].set_ylabel('BC Noise')
    plot_stimulus.plot_video_frames(bc_noise_output, n_rows=1, n_cols=n_cols, fis=fis, axs=axs[3, :])

    axs[4, 0].set_ylabel('RGC Synaptic Inputs')
    plot_stimulus.plot_video_frames(rgc_synaptic_inputs, n_rows=1, n_cols=n_cols, fis=fis, axs=axs[4, :])

    plt.tight_layout()
    plt.show()


def fetch_dataset(stimulus_config_ids, rgc_id, bc_noise_id, with_pr_noise=False):
    dataset_tab = (Stimulus & [dict(stimulus_config_id=stimulus_config_id) for stimulus_config_id in
                               stimulus_config_ids]).proj('target_pos')

    rgc_inputs_tab = RGCSynapticInputs if not with_pr_noise else RGCSynapticPrNoiseInputs

    dataset_tab = dataset_tab * (rgc_inputs_tab & dict(rgc_id=rgc_id, bc_noise_id=bc_noise_id)).proj(
        **{f"response": "rgc_synaptic_inputs"})

    if len((dataset_tab & "wo_cricket=1")) > 0 and len((dataset_tab & "wo_cricket=0")) > 0:
        w_map = {k: k for k in ["target_pos", "response"]}
        wo_map = {f"{k}_wo": k for k in ["response"]}
        dataset_tab = (dataset_tab & "wo_cricket=0").proj(wo_cricket_1='wo_cricket', **w_map) * \
                      (dataset_tab & "wo_cricket=1").proj(wo_cricket_2='wo_cricket', **wo_map)

    return dataset_tab


def main():
    connect_to_database()
    plot_simulation(key=None)


if __name__ == '__main__':
    main()
