import soundfile as sf
import numpy as np
import resampy
import os
from pydub import AudioSegment
from pyannote.audio import Audio
from pyannote.audio import Pipeline
import webrtcvad
from pystoi.stoi import stoi
from pesq import pesq
import pyloudnorm as pyln
import noisereduce as nr
from scipy.io import wavfile
from pydub.silence import split_on_silence
from pyannote.audio import Inference
from pyannote.core import Annotation, Segment
from pyannote.audio.utils.signal import Binarize


def preprocess_and_improve_audio(input_dir, output_dir):
    """
    Converts M4A and MP3 audio files in a directory to WAV format with parameters optimized for Whisper and Wav2Vec.
    Performs voice activity detection (VAD) to remove silence and noise from the audio.
    Improves the audio quality by removing background noise.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vad = webrtcvad.Vad()
    vad.set_mode(2)  # Set VAD aggressiveness level (2 for most aggressive)

    for filename in os.listdir(input_dir):
        if filename.endswith(".m4a") or filename.endswith(".MP3"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".wav")

            if filename.endswith(".m4a"):
                sound = AudioSegment.from_file(input_path, format="m4a")
            elif filename.endswith(".MP3"):
                sound = AudioSegment.from_file(input_path, format="MP3")

            sound = sound.set_channels(1).set_frame_rate(16000)

            # Remove noise from the audio
            sound = sound.low_pass_filter(1200)
            sound = sound.high_pass_filter(200)

            # Split the audio into chunks based on silence
            chunks = split_on_silence(sound, min_silence_len=100, silence_thresh=-50)
            output = AudioSegment.empty()
            for chunk in chunks:
                output += chunk

            # Export the preprocessed audio as a WAV file
            output.export(output_path, format="wav")

            # Resample the audio to 16kHz
            audio_data, sample_rate = sf.read(output_path)
            if len(audio_data) == 0:
                continue  # skip empty audio files
            if sample_rate != 16000:
                audio_data = resampy.resample(audio_data, sample_rate, 16000)
                sample_rate = 16000

            # Split audio into fixed-length chunks
            chunk_duration_ms = 30
            chunk_size = int(chunk_duration_ms * sample_rate / 1000)
            num_chunks = int(np.ceil(len(audio_data) / chunk_size))
            padded_audio_data = np.zeros(num_chunks * chunk_size)
            padded_audio_data[:len(audio_data)] = audio_data
            chunks = np.split(padded_audio_data, num_chunks)

            # Detect voice activity in each chunk and keep only the active chunks
            active_chunks = []
            for chunk in chunks:
                is_speech = vad.is_speech(chunk.tobytes(), sample_rate, length=len(chunk))
                if is_speech:
                    active_chunks.append(chunk)
            active_audio = np.concatenate(active_chunks)

            # Save the improved audio as a WAV file
            improved_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.wav")
            sf.write(improved_path, active_audio, sample_rate)


def main():
    input_dir = '/home/atokareva/slodim/corpus/audio'
    output_dir = '/home/atokareva/slodim/corpus/processed_audio'

    preprocess_and_improve_audio(input_dir, output_dir)

    print("Audio preprocessing and improvement complete!")

if __name__ == '__main__':
    main()
