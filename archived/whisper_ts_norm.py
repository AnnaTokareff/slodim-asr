import torch
import whisper_timestamped
import jiwer
import pandas as pd
from normalizer import TextNormalizer
from datetime import datetime
from utils import load_paths, load_target, save_as_file


def load_model(device, size="small"):
    """
    load multilingual model according to the size
    size: str
    return: model object
    """
    print("Loading model...")
    return whisper_timestamped.load_model(size, device=device)


def transcribe(model, audio_paths, prompt):
    """
    Transcribe with whisper_timestamped
    audio_paths: list(str)
    return: transcriptions( list(str) )
    """
    print("Transcribing...")
    transcriptions = []

    for path in audio_paths:
        audio = whisper_timestamped.load_audio(path)
        results = whisper_timestamped.transcribe(
            model, audio, initial_prompt=prompt, language="fr"
        )
        transcriptions.append(results["text"])
        save_as_file(results, path)
        print(f"{audio_paths.index(path) + 1} / {len(audio_paths)} finished")

    return transcriptions


def text_normalization(targets, transcriptions):
    """
    Normalize the text to exclude formatting factors
    targets, transcriptions: ( list(str) )
    return: data (pandas Dataframe)
    """
    print("Normalizing text...")
    normalizer = TextNormalizer()
    data = pd.DataFrame(dict(targets=targets, transcriptions=transcriptions))

    data["targets_clean"] = [normalizer(text) for text in data["targets"]]
    data["transcriptions_clean"] = [
        normalizer(trans) for trans in data["transcriptions"]
    ]
    return data


def wer_cer(data):
    """
    Calculate WER & CER
    data: pandas Dataframe
    return: wer, cer (float)
    """
    wer = jiwer.wer(list(data["targets_clean"]), list(data["transcriptions_clean"]))
    cer = jiwer.cer(list(data["targets_clean"]), list(data["transcriptions_clean"]))
    print(f"Final WER: {wer * 100:.2f} %, CER: {cer * 100:.2f} %")
    return wer, cer


def main():
    targets = load_target(text_paths)
    model = load_model(device=devices)
    transcriptions = transcribe(model, audio_paths, prompt)
    data = text_normalization(targets, transcriptions)
    wer_cer(data)


if __name__ == "__main__":
    # audio_paths, text_paths = load_paths("temp_audio", "temp_txt")
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    audio_paths = ["./audio/BOC-066.m4a"]
    text_paths = ["./txt/P39682 - boc-066.zip (1).txt"]

    prompt = "mmm, c’est vrai, mmm... Ah ben euh la montagne elle est euh elle est dure."

    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"Duration: {end_time - start_time} for {len(audio_paths)} files")
