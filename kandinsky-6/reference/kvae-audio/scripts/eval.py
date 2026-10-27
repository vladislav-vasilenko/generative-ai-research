import csv
from dataclasses import dataclass
from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal, metrics
from audiotools.core import util
from tqdm import tqdm

from kvae_audio.metrics import loss


@dataclass
class State:
    stft_loss: loss.MultiScaleSTFTLoss
    mel_loss: loss.MelSpectrogramLoss
    waveform_loss: loss.L1Loss
    sisdr_loss: loss.SISDRLoss
    sdr_loss: loss.SISDRLoss
    snr_loss: loss.SISDRLoss


SAMPLE_RATES = [48000]


def get_metrics(signal_path, recons_path, state):
    output = {}
    signal = AudioSignal(signal_path)
    recons = AudioSignal(recons_path)

    if signal.audio_data.shape[-1] != recons.audio_data.shape[-1]:
        min_len = min(signal.audio_data.shape[-1], recons.audio_data.shape[-1])
        signal.audio_data = signal.audio_data[..., :min_len]
        recons.audio_data = recons.audio_data[..., :min_len]

    for sr in SAMPLE_RATES:
        x = signal.clone().resample(sr)
        y = recons.clone().resample(sr)
        key = str(sr)

        try:
            pesqwb = metrics.quality.pesq(x, y)
        except Exception:
            pesqwb = None

        output.update(
            {
                f"mel-{key}": state.mel_loss(x, y),
                f"stft-{key}": state.stft_loss(x, y),
                f"waveform-{key}": state.waveform_loss(x, y),
                f"sisdr-{key}": -state.sisdr_loss(x, y),
                f"sdr-{key}": -state.sdr_loss(x, y),
                f"snr-{key}": -state.snr_loss(x, y),
                f"pesq-wb-{key}": pesqwb,
            }
        )

    output["path"] = signal.path_to_file
    return output


@argbind.bind(without_prefix=True)
@torch.no_grad()
def main(
    input_path: str = "input_samples",
    output_path: str = "output_samples",
):

    state = State(
        waveform_loss=loss.L1Loss(),
        stft_loss=loss.MultiScaleSTFTLoss(),
        mel_loss=loss.MelSpectrogramLoss(),
        sisdr_loss=loss.SISDRLoss(scaling=True, zero_mean=True),
        sdr_loss=loss.SISDRLoss(scaling=False, zero_mean=False),
        snr_loss=loss.SISDRLoss(scaling=False, zero_mean=True),
    )

    audio_files = util.find_audio(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for audio_file in tqdm(audio_files, desc="Eval"):
        row = get_metrics(audio_file, output_dir / audio_file.name, state)
        for key, value in row.items():
            if torch.is_tensor(value):
                row[key] = value.item()
        rows.append(row)

    with open(output_dir / "metrics.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        main()
