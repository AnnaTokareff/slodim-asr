import whisper_timestamped
import jiwer
from normalizer import TextNormalizer

audio = whisper_timestamped.load_audio("./audio/BOC-066.m4a")

print("Loading model...")
model = whisper_timestamped.load_model("medium")

print("Transcribing...")
results = whisper_timestamped.transcribe(model, audio, language="fr")

print("Loading files...")
transcription = results["text"]
with open("./txt/P39682 - boc-066.zip (1).txt", "r", encoding="utf8") as f:
    target = f.read()

print("Normalizing text...")
norm = TextNormalizer()
transcription = norm(transcription)
target = norm(target)


def wer_cer(target, transcription):
    wer = jiwer.wer(target, transcription)
    cer = jiwer.cer(target, transcription)
    print(f"WER: {wer}\nCER: {cer}")


wer_cer(target, transcription)
