import torch
import whisper
import jiwer
import pandas as pd
import librosa
from normalizer import TextNormalizer
from datetime import datetime
from utils import load_paths, load_target, save_as_file
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

def load_model(device, size="small"):
    """
    load multilingual model according to the size
    size: str
    return: model object
    """
    print("Loading model...")
    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{size}")
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{size}")
    
    def transcribe_audio(path):
        audio_array, sampling_rate = librosa.load(path, sr=None, mono=True)
        # Resample to 16kHz if needed
        if sampling_rate != 16000:
            audio_array = librosa.resample(audio_array, sampling_rate, 16000)
            sampling_rate = 16000
        return audio_array, sampling_rate
    
    processor.transcribe_audio = transcribe_audio
    
    return model.to(device), processor

def transcribe(model, audio_paths, processor, prompt, word_timestamps):
    device = next(model.parameters()).device
    transcriptions = []
    for path in audio_paths:
        audio_array, sampling_rate = librosa.load(path, sr=None, mono=True)
        input_features = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_features
        input_features = input_features.to(device) # Move input to device
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True, normalize=False)
        res_dict = {"text": transcription[0]}
        save_as_file(res_dict, path)
        transcriptions.append(transcription[0])
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
    data["targets_clean"] = [normalizer(text.decode('utf-8')) for text in data["targets"]]
    data["transcriptions_clean"] = [
        normalizer(trans) for trans in data["transcriptions"]
    ]
    print(data)
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
    # audio_paths, text_paths = load_paths("temp_audio", "temp_txt")
    audio_paths = ["/content/BOC-066 (1).wav"]
    text_paths = ["/content/P40727 - bls-056-entr1-audio.zip.txt"]
    #audio_paths, text_paths = load_paths("", "")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prompt = "mmm, c’est vrai, mmm... Ah ben euh la montagne elle est euh elle est dure. Ouais."
    targets = load_target(text_paths)
    model, processor = load_model(device=device)
    transcriptions = transcribe(model, audio_paths, processor, prompt, word_timestamps=False)
    data = text_normalization(targets, transcriptions, prompt=True)

    normalizer = TextNormalizer(prompt)
    with open("target_normed.txt", "w", encoding = "utf-8") as f:
        f.write(normalizer(str(targets[0])))
    with open("trans_normed.txt", "w", encoding="utf-8") as f:
      f.write(normalizer(str(transcriptions[0])))

    wer_cer(data)


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"Duration: {end_time - start_time} for {len(audio_paths)} files")
