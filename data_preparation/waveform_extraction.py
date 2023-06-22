import librosa
import json
import soundfile as sf
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Load the metadata CSV file
metadata_file = './output/data.csv'
metadata_df = pd.read_csv(metadata_file)

# Split the metadata into train and test sets
train_df, test_df = train_test_split(metadata_df, test_size=0.1, random_state=42)

# Process metadata and store the required information in a list
def process_data(df):
    data = []
    target_sampling_rate = 16000

    for _, row in df.iterrows():
        path = row['path']
        transcription = row['text']

        # Load audio file and get the waveform
        audio_data, sampling_rate = librosa.load(path, sr=None)

        # Check if the input signal length is 0
        if len(audio_data) == 0:
            print(f"Deleting {path} due to zero-length input signal")
            os.remove(path)
            continue

        # Resample audio to the target sampling rate
        if sampling_rate != target_sampling_rate:
            audio_data = librosa.resample(audio_data, orig_sr=sampling_rate, target_sr=target_sampling_rate)

        # Save resampled audio to the original file
        sf.write(path, audio_data, target_sampling_rate, format='wav')

        # Convert NumPy array to list
        audio_data_list = audio_data.tolist()

        # Store the information in a dictionary
        data.append({
            "path": path,
            "transcription": transcription,
            "waveform": audio_data_list
        })

    return data

train_data = process_data(train_df)
test_data = process_data(test_df)

# Save the train and test data to JSON files
with open("./output/train.json", "w") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open("./output/test.json", "w") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)
