import os
from datetime import datetime

import keras
from matplotlib import pyplot as plt

from alphacnn import paths


class EarlyStoppingWarmup(keras.callbacks.EarlyStopping):
    # https://stackoverflow.com/questions/46287403/is-there-a-way-to-implement-early-stopping-in-keras-only-after-the-first-say-1/76728168#76728168
    def __init__(self, *args, start_from_epoch=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_from_epoch = start_from_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_from_epoch:
            super().on_epoch_end(epoch, logs)


def plot_decoder_loss(history):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    for i, h in enumerate(history):
        ax.semilogy(h['loss'], color=f"C{i}", label='Train' if i == 0 else '_')
        ax.semilogy(h['val_loss'], color=f"C{i}", label='Val' if i == 0 else '_', linestyle="--")
    ax.set(xlabel='Epoch', ylabel='Loss')
    plt.tight_layout()
    plt.legend()


def save_decoder(model_key, decoder, decoder_kind='pos_pres_decoder'):
    model_has_dict = model_key.copy()
    model_has_dict['timestamp'] = str(datetime.now())
    model_hash = str(hash(tuple(model_has_dict.items())))

    model_dir = os.path.join(paths.get_project_path(), 'model', decoder_kind)
    assert os.path.isdir(model_dir), f"Model directory {model_dir} does not exist."
    model_path = os.path.join(model_dir, model_hash)
    assert not any([f.startswith(model_hash) for f in os.listdir(model_dir)]), f"Model {model_path} already exists."

    print(f'Saving decoder to {model_path}')
    decoder.save_to_file(model_path=model_path)
    return model_path
