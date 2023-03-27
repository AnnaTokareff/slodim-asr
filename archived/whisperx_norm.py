import jiwer
import whisperx
import pandas as pd
from normalizer import TextNormalizer
from utils import load_paths, load_target, save_as_txt_file,  save_as_json_file, convert_into_right_format_whisperX
#!pip install git+https://github.com/m-bain/whisperx.git

def load_model(size="base", device="cpu"):
    """
    load multilingual model according to the size and device
    size: str
    device: str
    return: model object
    """
    print("Loading model...")
    model = load_model(size, device)
    return model
  
def transcribe(audio_paths, model, align=False, device="cpu"):
    """
    Transcribe with whisperx with 
    or without hard alignment 
    audio_paths: list(str)
    model: model object
    align: boolean (False by default)
    device: str (cpu be default)
    return: transcriptions( list(str) )
    """
    print("Transcribing...")

    transcriptions = []
    segments = []
    # load alignment model and metadata
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)

    for audio_path in audio_paths:
      results = model.transcribe(audio_path)
      transcriptions.append(results["text"])
      save_as_txt_file(results["text"], audio_path) # save transcriptions in txt

      if align:
        # align whisper output
          res_aligned = whisperx.align(result["segments"], model_a, metadata, audio_path, device)
          res_aligned = convert_into_right_format_whisperX(res_aligned)
          save_as_json_file(res_aligned, audio_path) # save aligned in json file
      else:
        save_as_json_file(results, audio_path) # save not aligned in json file
      print(f"{audio_paths.index(audio_path) + 1} / {len(audio_paths)} finished")

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
    
    # Load data
    audio_paths = load_paths()
    targets = load_target()

    # Transcribe
    transcriptions = transcribe(audio_paths, model, align=True)

    # Text Normalization
    data = text_normalization(targets, transcriptions)

    # Calculate WER & CER
    wer, cer = wer_cer(data)


if __name__ == "__main__":
    main()
