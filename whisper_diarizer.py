import torch
import whisper
import jiwer
import pandas as pd
from normalizer import TextNormalizer
from datetime import datetime
from utils import load_paths, load_target, save_as_file
from typing import List, Dict, Any
from whisperx import load_align_model, align
from whisperx.diarize import DiarizationPipeline, assign_word_speakers


def load_model(device, size="small"):
    """
    Load the multilingual model according to the size.

    Args:
        device: The device to use for inference (e.g., "cpu" or "cuda").
        size: The size of the model ("small" by default).

    Returns:
        The loaded model object.
    """
    print("Loading model...")
    return whisper.load_model(size, device=device)


def transcribe(model, audio_paths, prompt=True, word_timestamps=True):
    """
    Transcribe audio files using the whisperAI model.

    Args:
        model: The loaded whisperAI model.
        audio_paths: A list of paths to the audio files to transcribe.
        prompt: Whether to use a prompt for transcription (True by default).
        word_timestamps: Whether to extract word-level timestamps (True by default).

    Returns:
        A list of transcriptions for each audio file.
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
    Normalize the text to exclude formatting factors.

    Args:
        targets: A list of target texts.
        transcriptions: A list of transcriptions.
        prompt: Whether to include the prompt in normalization (True by default).

    Returns:
        A pandas DataFrame with the normalized texts.
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
    Calculate WER & CER.

    Args:
        data: A pandas DataFrame with the normalized texts.

    Returns:
        The calculated WER and CER.
    """
    wer = jiwer.wer(list(data["targets_clean"]), list(data["transcriptions_clean"]))
    cer = jiwer.cer(list(data["targets_clean"]), list(data["transcriptions_clean"]))
    print(f"Final WER: {wer * 100:.2f} %, CER: {cer * 100:.2f} %")
    return wer, cer


def diarize(audio_file: str, hf_token: str) -> Dict[str, Any]:
    """
    Perform speaker diarization on an audio file.

    Args:
        audio_file: Path to the audio file to diarize.
        hf_token: Authentication token for accessing the Hugging Face API.

    Returns:
        A dictionary representing the diarized audio file, including the speaker embeddings and the number of speakers.
    """
    diarization_pipeline = DiarizationPipeline(use_auth_token=hf_token)
    diarization_result = diarization_pipeline(audio_file)
    return diarization_result


def assign_speakers(
    diarization_result: Dict[str, Any], aligned_segments: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Assign speakers to each transcript segment based on the speaker diarization result.

    Args:
        diarization_result: Dictionary representing the diarized audio file, including the speaker embeddings and the number of speakers.
        aligned_segments: Dictionary representing the aligned transcript segments.

    Returns:
        A list of dictionaries representing each segment of the transcript, including the start and end times, the
        spoken text, and the speaker ID.
    """
    result_segments, word_seg = assign_word_speakers(
        diarization_result, aligned_segments["segments"]
    )
    results_segments_w_speakers: List[Dict[str, Any]] = []
    for result_segment in result_segments:
        results_segments_w_speakers.append(
            {
                "start": result_segment["start"],
                "end": result_segment["end"],
                "text": result_segment["text"],
                "speaker": result_segment["speaker"],
            }
        )
    return results_segments_w_speakers


def align_segments(
    segments: List[Dict[str, Any]],
    language_code: str,
    audio_file: str,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Align the transcript segments using a pretrained alignment model.

    Args:
        segments: List of transcript segments to align.
        language_code: Language code of the audio file.
        audio_file: Path to the audio file containing the audio data.
        device: The device to use for inference (e.g., "cpu" or "cuda").

    Returns:
        A dictionary representing the aligned transcript segments.
    """
    model_a, metadata = load_align_model(language_code=language_code, device=device)
    result_aligned = align(segments, model_a, metadata, audio_file, device)
    return result_aligned


def transcribe_and_diarize(
    audio_file: str,
    hf_token: str,
    model_name: str,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """
    Transcribe an audio file and perform speaker diarization to determine which words were spoken by each speaker.

    Args:
        audio_file: Path to the audio file to transcribe and diarize.
        hf_token: Authentication token for accessing the Hugging Face API.
        model_name: Name of the model to use for transcription.
        device: The device to use for inference (e.g., "cpu" or "cuda").

    Returns:
        A list of dictionaries representing each segment of the transcript, including the start and end times, the
        spoken text, and the speaker ID.
    """
    transcript = transcribe(audio_file, model_name, device)
    aligned_segments = align_segments(
        transcript["segments"], transcript["language_code"], audio_file, device
    )
    diarization_result = diarize(audio_file, hf_token)
    results_segments_w_speakers = assign_speakers(diarization_result, aligned_segments)

    # Print the results in a user-friendly way
    for i, segment in enumerate(results_segments_w_speakers):
        print(f"Segment {i + 1}:")
        print(f"Start time: {segment['start']:.2f}")
        print(f"End time: {segment['end']:.2f}")
        print(f"Speaker: {segment['speaker']}")
        print(f"Transcript: {segment['text']}")
        print("")

    return results_segments_w_speakers


def main():
    audio_paths = "path/audio"
    text_paths = "path/transcr"
    hf_token "YOUR HF token here"
    targets = load_target(text_paths)
    model = load_model(device=devices)
    transcriptions = transcribe(model, audio_paths, prompt)
    data = text_normalization(targets, transcriptions, prompt=True)

    normalizer = TextNormalizer(prompt)
    with open("target_normed.txt", "w") as f:
        f.write(normalizer(targets[0]))
    with open("trans_normed.txt", "w") as f:
        f.write(normalizer(transcriptions[0]))

    wer_cer(data)


if __name__ == "__main__":
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    prompt = "mmm, c’est vrai, mmm... Ah ben euh la montagne elle est euh elle est dure. Ouais."

    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"Duration: {end_time - start_time} for {len(audio_paths)} files")
