from data import load_paths, load_target
import jiwer
import whisper_timestamped
import pandas as pd
from normalizer import TextNormalizer


def load_model(size="base"):
    """
    load multilingual model according to the size
    size: str
    return: model object
    """
    print("Loading model...")
    if size == "base":
        return whisper_timestamped.load_model("base")
    return whisper_timestamped.load_model(size)


def transcribe(audio_paths):
    """
    Transcribe with whisper_timestamped
    audio_paths: list(str)
    return: transcriptions( list(str) )
    """
    print("Transcribing...")
    transcriptions = []

    for path in audio_paths:
        audio = whisper_timestamped.load_audio(path)
        results = whisper_timestamped.transcribe(model, audio, language="fr")
        transcriptions.append(results["text"])
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


if __name__ == "__main__":
    audio_paths, text_paths = load_paths("audio", "txt")
    targets = load_target(text_paths)
    model = load_model("large")
    transcriptions = transcribe(audio_paths)
    data = text_normalization(targets, transcriptions)
    wer_cer(data)
