import torch
import whisper
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
    return whisper.load_model(size, device=device)


def transcribe(model, audio_paths, prompt=True, word_timestamps=True):
    """
    Transcribe with whisperAI
    audio_paths: list(str)
    promt: str (True by default)
    promt helps to make model produce fillers
    word_timestamps: boolean (True by default)
    Extract word-level timestamps
    return: transcriptions( list(str) )
    """
    print("Transcribing...")
    transcriptions = []

    for path in audio_paths:
        if prompt:
            results = model.transcribe(path, initial_prompt=prompt, language="fr", word_timestamps=True)
        else: 
            results = model.transcribe(path, language="fr", word_timestamps=True)
            
        transcriptions.append(results["text"])
        save_as_file(results, path)
        print(f"{audio_paths.index(path) + 1} / {len(audio_paths)} finished")

    return transcriptions


def text_normalization(targets, transcriptions, prompt=True):
    """
    Normalize the text to exclude formatting factors
    targets, transcriptions: ( list(str) )
    return: data (pandas Dataframe)
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
<<<<<<< HEAD
    transcriptions = transcribe(model, audio_paths, prompt)
    data = text_normalization(targets, transcriptions, prompt=True)

    normalizer = TextNormalizer(prompt)
    with open("target_normed.txt", "w") as f:
        f.write(normalizer(targets[0]))
    with open("trans_normed.txt", "w") as f:
        f.write(normalizer(transcriptions[0]))

=======
    transcriptions = transcribe(model, audio_paths, prompt) # otherwise put: prompt=False
    data = text_normalization(targets, transcriptions)
>>>>>>> 2111da63c1ed4640071419fd2a3428c9d1c2ad3e
    wer_cer(data)


if __name__ == "__main__":
    # audio_paths, text_paths = load_paths("temp_audio", "temp_txt")
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    audio_paths = ["./audio/POT-016-Entr1-Audio.MP3", "./audio/RAF-007-Entr1-Audio.MP3"]
    text_paths = [
        "./txt/P40735 - pot-016-entr1-audio.zip.txt",
        "./txt/P40736 - raf-007-entr1-audio.zip.txt",
    ]

    prompt = "mmm, c’est vrai, mmm... Ah ben euh la montagne elle est euh elle est dure. Ouais."

    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"Duration: {end_time - start_time} for {len(audio_paths)} files")
