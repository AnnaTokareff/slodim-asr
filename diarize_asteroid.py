import os
import numpy as np
import torch
import webrtcvad
import soundfile as sf
from tqdm import tqdm
from asteroid.models import BaseModel
from pystoi.stoi import stoi
from pesq import pesq
import pyloudnorm as pyln
import noisereduce as nr
from scipy.io import wavfile
from tqdm import tqdm

def process_audio_file(audio_file_path, model_path, output_dir, max_split_size_mb=1024, chunk_size=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaseModel.from_pretrained(model_path).to(device)
    audio_info = sf.info(audio_file_path)
    sample_rate = audio_info.samplerate

    # Calculate chunk size in samples
    chunk_size_samples = int(chunk_size * sample_rate)
    # Pad audio data to ensure it is evenly divisible by chunk size
    total_samples = audio_info.frames
    padding = chunk_size_samples - (total_samples % chunk_size_samples)
    if padding == chunk_size_samples:
        padding = 0
    audio_data, _ = sf.read(audio_file_path, frames=total_samples+padding)

    start_sample = 0
    end_sample = chunk_size_samples
    est_sources = []
    pbar = tqdm(total=total_samples, desc='Processing audio', unit='samples')
    while start_sample < total_samples:

        chunk_data = audio_data[start_sample:end_sample]
        tensor = torch.Tensor(chunk_data).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            est_source = model.forward(tensor)
        est_source = est_source.squeeze().cpu().detach().numpy()
        est_sources.append(est_source)

        start_sample = end_sample
        end_sample += chunk_size_samples
        pbar.update(chunk_size_samples)

    est_sources = np.concatenate(est_sources, axis=1)
    audio_file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    source1_path = os.path.join(output_dir, f"{audio_file_name}_source_1.wav")
    source2_path = os.path.join(output_dir, f"{audio_file_name}_source_2.wav")
    sf.write(source1_path, est_sources[0], sample_rate)
    sf.write(source2_path, est_sources[1], sample_rate)
    pbar.close()


def improve_audio_quality(output_dir, aggressiveness=2):
    # Voice activity detection
    vad = webrtcvad.Vad()
    vad.set_mode(aggressiveness)

    # Loop over audio files in the output directory
    for file_name in os.listdir(output_dir):
        if not file_name.endswith(".wav"):
            continue

        source_path = os.path.join(output_dir, file_name)

        # Load audio data and resample to 16kHz
        audio_data, sample_rate = sf.read(source_path)
        if sample_rate != 16000:
            audio_data = sf.resample(audio_data, sample_rate, 16000)
            sample_rate = 16000

        # Split audio into 30ms chunks
        chunk_duration_ms = 30
        chunk_size = int(chunk_duration_ms * sample_rate / 1000)
        chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]

        # Detect voice activity in each chunk and keep only the active chunks
        active_chunks = []
        for chunk in chunks:
          is_speech = vad.is_speech(chunk.tobytes(), sample_rate)
          if is_speech:
            active_chunks.append(chunk)
        active_audio = np.concatenate(active_chunks)

        # Write processed audio to file
        output_path = os.path.join(output_dir, f"{file_name[:-4]}_processed.wav")
        sf.write(output_path, active_audio, sample_rate)


def main():
  model_path = "mpariente/DPRNNTasNet-ks2_WHAM_sepclean"
  audio_file_path = "/content/BLS-056-Entr1-Audio_cut.wav"
  output_dir = "/content/res" 
  process_audio_file(audio_file_path, model_path, output_dir)
  improve_audio_quality(output_dir)

if __name__ == '__main__':
    main()
