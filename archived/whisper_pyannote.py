import whisper
import re
import jiwer

from pydub import AudioSegment
from huggingface_hub import login
from pyannote.audio import Pipeline
from normalizer import TextNormalizer

# spacer
spacermilli = 2000
spacer = AudioSegment.silent(duration=spacermilli)

audio = AudioSegment.from_file("./audio/BOC-066_5min.m4a")  # lecun1.wav

audio = spacer.append(audio, crossfade=0)

audio.export("./audio/temp.wav", format="wav")

# log in huggingface_hub
login()

# pyannote pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=True)

print("Diarizing...")
DEMO_FILE = {"audio": "./audio/temp.wav", "num_speakers": 2}
dz = pipeline(DEMO_FILE)

with open("diarization.txt", "w") as text_file:
    text_file.write(str(dz))

print(*list(dz.itertracks(yield_label=True))[:10], sep="\n")

# prepare audio files according to diarization
def millisec(timeStr):
    spl = timeStr.split(":")
    s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2])) * 1000)
    return s


# group segments
print("Grouping segments...")
dzs = open("diarization.txt").read().splitlines()

groups = []
g = []
lastend = 0

for d in dzs:
    if g and (g[0].split()[-1] != d.split()[-1]):  # same speaker
        groups.append(g)
        g = []

    g.append(d)

    end = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=d)[1]
    end = millisec(end)
    if lastend > end:  # segment engulfed by a previous segment
        groups.append(g)
        g = []
    else:
        lastend = end
if g:
    groups.append(g)
print(*groups, sep="\n")

# save each part
print("Saving segments...")
audio = AudioSegment.from_wav("./audio/temp.wav")
gidx = -1
for g in groups:
    start = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[0])[0]
    end = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[-1])[1]
    start = millisec(start)  # - spacermilli
    end = millisec(end)  # - spacermilli
    print(start, end)
    gidx += 1
    audio[start:end].export(str(gidx) + ".wav", format="wav")

# run whisper
print("Transcribing...")
model = whisper.load_model("large")

transcription = ""
for i in range(gidx + 1):
    result = model.transcribe(str(i) + ".wav", language="fr", fp16=False)
    transcription += result["text"]

with open("./transcription/temp.txt", "w") as f:
    f.write(transcription)


print("Loading target...")
with open("./txt/P39682 - boc-066.zip (1)_5min.txt", "r", encoding="utf8") as f:
    target = f.read()

normalizer = TextNormalizer()

transcription = normalizer(transcription)
target = normalizer(target)

print("Calculating WER...")
wer = jiwer.wer(transcription, target)
print(f"WER: {wer}")
