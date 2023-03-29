from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from whisper_transcribe import text_normalization, wer_cer
import torch
import json
import jiwer
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").to(device)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="transcribe")

# load dataset and read audio files
with open("./output/test.json", "r") as f:
    data = json.load(f)
labels = []
transcriptions = []

for idx, audio in enumerate(data[:100]):
    labels.append(audio["transcription"])
    input_features = processor(audio["waveform"], sampling_rate=16000, return_tensors="pt").input_features.to(device)

    # generate token ids
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

    # decode token ids to text without normalisation
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True, normalize=False)
    transcriptions.append(transcription[0])
    
    if (idx + 1) % 10 == 0:
        print(f"Transcribed {idx + 1} audio files")

data = pd.DataFrame(dict(targets=labels, transcriptions=transcriptions))
wer = jiwer.wer(list(data["targets"]), list(data["transcriptions"]))
cer = jiwer.cer(list(data["targets"]), list(data["transcriptions"]))
print(f"Final WER: {wer * 100:.2f} %, CER: {cer * 100:.2f} %")