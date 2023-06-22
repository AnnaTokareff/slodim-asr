from pydub import AudioSegment
from pyannote.audio import Audio
from pyannote.core import Annotation
from pyannote.audio import Pipeline
import webrtcvad
from pystoi.stoi import stoi
from pesq import pesq
import pyloudnorm as pyln
import noisereduce as nr
from scipy.io import wavfile
import soundfile as sf
import numpy as np
import resampy
import os

def diarize_audio(input_folder, output_folder, use_gpu=True):
    # Create Pyannote pipeline for overlapped speech detection
    pipeline = Pipeline.from_pretrained(
        "pyannote/overlapped-speech-detection",
        use_auth_token="ACCESS TOKEN")

    for file_name in os.listdir(input_folder):
        if not file_name.endswith(".wav"):
            # Convert to WAV format
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".wav")
            AudioSegment.from_file(input_path).export(output_path, format="wav")
            file_name = os.path.splitext(file_name)[0] + ".wav"

        # Load audio file       
        audio_path = os.path.join(input_folder, file_name)
        audio = AudioSegment.from_wav(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1)

        # Save audio in WAV format
        temp_audio_file_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_temp.wav")
        audio.export(temp_audio_file_path, format="wav")

        print("Diarizing...")

        # Run pipeline on audio
        diarization = pipeline({'audio': temp_audio_file_path})

        print("Split into 2 speakers....")

        # Split audio into separate files for each speaker
        speaker1_audio = AudioSegment.empty()
        speaker2_audio = AudioSegment.empty()
        speaker1_segments = []
        speaker2_segments = []

        for speech, _, label in diarization.itertracks(yield_label=True):
          print(speech, label)
          start_time = speech.start
          end_time = speech.end
          segment_audio = audio[start_time:end_time]

          if label.startswith('Speaker 1'):
              speaker1_audio += segment_audio
              speaker1_segments.append((start_time, end_time))

          elif label.startswith('Speaker 2'):
              speaker2_audio += segment_audio
              speaker2_segments.append((start_time, end_time))


        print("Saving the audios....")

        # Save speaker audio files
        speaker1_audio.export(os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_speaker1.wav"), format="wav")
        speaker2_audio.export(os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_speaker2.wav"), format="wav")


        # Save diarization output to RTTM file
        output_path = os.path.join(output_folder, "diarization.rttm")
        with open(output_path, "w") as rttm_file:
          diarization.write_rttm(rttm_file)
        # Save speaker segments
        with open(os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_speaker1_segments.txt"), "w") as f:
            for segment in speaker1_segments:
                f.write(f"{segment[0]}-{segment[1]}\n")
        with open(os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_speaker2_segments.txt"), "w") as f:
            for segment in speaker2_segments:
                f.write(f"{segment[0]}-{segment[1]}\n")


def improve_audio_quality(output_dir, aggressiveness=2):

    vad = webrtcvad.Vad()
    vad.set_mode(aggressiveness)

    for file_name in os.listdir(output_dir):
        if not file_name.endswith(".wav"):
            continue

        source_path = os.path.join(output_dir, file_name)
        # Resample audio to 16kHz
        audio_data, sample_rate = sf.read(source_path)
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
        target_path = os.path.join(output_dir, f"improved_{file_name}")
        sf.write(target_path, active_audio, sample_rate)


def main():
  input_folder = "/content/sample_data/audios"
  output_folder = "/content/sample_data/resss" 
  diarize_audio(input_folder, output_folder)
  improve_audio_quality(output_folder)


if __name__ == '__main__':
    main()
