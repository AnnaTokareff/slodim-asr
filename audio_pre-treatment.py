import os
import shutil
import torch
import pyannote.audio
from pydub import AudioSegment
from pyannote.audio import Pipeline
from pydub.silence import split_on_silence
from pyannote.audio import Inference
from pyannote.core import Annotation, Segment
from pyannote.audio.utils.signal import Binarize





def preprocess_audio(input_dir, output_dir):
    """
    Converts all m4a audio files in a directory to wav audio files with 
    parameters optimized for Whisper and Wav2Vec.
    Performs voice activity detection (VAD) to remove silence and noise from the audio
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".m4a"):           
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".wav")
            sound = AudioSegment.from_file(input_path, format="m4a")
            sound = sound.set_channels(1).set_frame_rate(16000)

            # remove noise from the audio
            sound = sound.low_pass_filter(1200)
            sound = sound.high_pass_filter(200)

            # split the audio into chunks based on silence
            chunks = split_on_silence(sound, min_silence_len=100, silence_thresh=-50)
            output = AudioSegment.empty()
            for chunk in chunks:
                output += chunk
            output.export(output_path, format="wav")


def perform_speaker_diarization(input_dir, output_dir):
    """
    Performs speaker diarization on all wav and m4a audio files in a directory using a pre-trained speaker diarization model.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create inference object with the pre-trained speaker diarization model
    inference =  Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="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")

    for filename in os.listdir(input_dir):
        if filename.endswith(".wav") or filename.endswith(".m4a"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".rttm")
            
            # apply the diarization pipeline to the audio file
            diarization = inference(input_path)

            # convert the diarization to RTTM format and save to file
            with open(output_path, "w") as rttm:
              diarization.write_rttm(rttm)
            


def main():
   input_dir = "/content/audios" 
   output_dir = "/content/processed_audios"
   

    # preprocess the audio files
   wav_dir = os.path.join(output_dir, "wav_format")
   #preprocess_audio(input_dir, wav_dir)

   # perform speaker diarization on the preprocessed audio files
   diarization_dir = os.path.join(output_dir, "diarization")
   perform_speaker_diarization(wav_dir, diarization_dir)


if __name__ == '__main__':
    main()
