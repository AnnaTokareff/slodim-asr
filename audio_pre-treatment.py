import os
import shutil
import torch
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pyannote.audio import Inference
from pyannote.core import Annotation


def preprocess_audio(input_dir, output_dir):
    """
    Converts all m4a audio files in a directory to wav audio files with parameters optimized for Whisper and Wav2Vec.
    Performs voice activity detection (VAD) to remove silence and noise from the audio.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".m4a"):
            print(filename)
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".wav")
            sound = AudioSegment.from_file(input_path, format="m4a")
            sound = sound.set_channels(1).set_frame_rate(16000)
            chunks = split_on_silence(sound, min_silence_len=100, silence_thresh=-50)
            output = AudioSegment.empty()
            for chunk in chunks:
                output += chunk
            output.export(output_path, format="wav")


def perform_speaker_diarization(input_dir, output_dir, model_name):
    """
    Performs speaker diarization on all wav audio files in a directory using a pre-trained speaker diarization model.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # download and load the pre-trained speaker diarization model
    model = torch.hub.load('pyannote/pyannote-audio', model_name)

    diarized_dir = os.path.join(output_dir, "diarized")

    if not os.path.exists(diarized_dir):
        os.makedirs(diarized_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(diarized_dir, os.path.splitext(filename)[0] + "_diarized.wav")
            audio = AudioSegment.from_file(input_path, format="wav")
            duration = audio.duration_seconds

            # create an Inference object with the speaker diarization model
            inference = Inference(model)

            # apply the diarization pipeline to the audio file
            diarization = inference(input_path)

            # apply the diarization to the audio signal
            result = diarization.apply(audio)

            # save the diarized audio file
            result.export(output_path, format="wav")



def main():
   input_dir = "PATH TO ALL THE AUDIOS IN m4a FORMAT"
   output_dir = "PATH FOR SAVING PROCESSED AUDIOS IN wav FORMAT"
   model_name = "dia_ami"

    # preprocess the audio files
   preprocess_audio(input_dir, output_dir)

   # perform speaker diarization on the preprocessed audio files
   perform_speaker_diarization(temp_dir, output_dir, model_name) # COMMENT THIS LINE IF YOU DON'T WANT SPEAKER DIARIZATION




if __name__ == '__main__':
    main()
