import json
import logging
import os
import re
import jiwer
import pandas as pd
from statistics import median

import torch
from pyannote.audio import Audio, Pipeline
from pyannote.core import Segment
from tqdm import tqdm
from pathlib import Path
from normalizer import TextNormalizer

import whisper
from utils import load_paths, load_target, save_as_file
import whisper.utils as wutils
from pydub import AudioSegment



def load_model(device, size="small"):
    """
    Load multilingual model according to the size
    size: str
    return: model object
    """
    print("Loading model...")

    return whisper.load_model(size)


def join_text_fields(json_file, output_file):
    """ take the json file and merges all the
    diarized segments of speech into one txt file"""
    
    with open(json_file, "r") as file:
        data = json.load(file)

    text = " ".join([segment["text"] for segment in data])
    print(text)

    with open(output_file, "w") as file:
        print("TXT file saved!")
        file.write(text)


def text_normalization(targets, transcriptions, prompt=True):
    """
    Normalize the text to exclude formatting factors
    targets, transcriptions: ( list(str) )
    return: data (pandas DataFrame)
    """
    print("Normalizing text...")
    normalizer = TextNormalizer(prompt=prompt)
    data = pd.DataFrame(dict(targets=targets, transcriptions=transcriptions))

    data["targets_clean"] = [normalizer(text) for text in data["targets"]]
    data["transcriptions_clean"] = [
        normalizer(trans) for trans in data["transcriptions"]
    ]
    return data


def wer_cer(data):
    """
    Calculate WER & CER
    data: pandas DataFrame
    return: wer, cer (float)
    """
    wer = jiwer.wer(list(data["targets_clean"]), list(data["transcriptions_clean"]))
    cer = jiwer.cer(list(data["targets_clean"]), list(data["transcriptions_clean"]))
    print(f"Final WER: {wer * 100:.2f}%, CER: {cer * 100:.2f}%")
    return wer, cer


def pyannote_diarize(audio_file, hf_token, num_speakers=2):
    """
    Run the diarization and split the audio into segments
    Returns:
    segments: list - list of diarized segments with start, end, and speaker information
    """

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1", use_auth_token=hf_token)

    print("Processing audio...")
    audio = Audio(sample_rate=16000, mono=True)
    diarization = pipeline(audio_file, num_speakers=num_speakers)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment = {"start": turn.start, "end": turn.end, "speaker": speaker}
        segments.append(segment)

    return segments


def transcribe_segments(audio_file, output_dir, segments, model):
    """
    Transcribe audio segments using the Whisper model
    Returns:
    result: dict - transcription results with text and segments information
    """

    model = load_model(model)
    audio = Audio(sample_rate=16000, mono=True)

    print("Transcribing audios with Whisper...")

    transcriptions = []

    for segment in tqdm(segments):
        waveform, sr = audio.crop(
            audio_file, Segment(segment["start"], segment["end"]))
        
        transcr = model.transcribe(waveform.squeeze().numpy(), verbose=None, language='fr')
        transcriptions.append({
        "text": transcr["text"],
        "start": segment["start"],
        "end": segment["end"],
        "speaker": segment["speaker"] })
        
    print(f"transcribed {len(transcriptions)}/{len(segment)}")
    
    diarization_file = Path(output_dir) / f"{Path(audio_file).stem}-diarized.json"
    with open(diarization_file, "w") as f:
            json.dump(transcriptions, f, indent=4)

    return transcriptions

            
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    audio_file = "path to audio file"
    output_dir = "path to output folder"
    json_file = "path to json file"
    num_speakers = 2  
    hf_token = "TOKEN" 
    model = "large"
    trs = "path to whisper transcriptions"
    golds = "path to golden transcriptions"
    segments = pyannote_diarize(audio_file, hf_token, num_speakers=2)
    transcribe_segments(audio_file, output_dir, segments, model)
    
    targets = load_target(golds)
    transcriptions = load_target(trs)
    
    data = text_normalization(targets, transcriptions)
    wer_cer(data)
    

    
if __name__ == "__main__":
    main()

