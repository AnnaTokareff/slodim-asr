import os
import re
from pydub import AudioSegment
import xml.etree.ElementTree as ET

def rename_eaf(eaf_directory: str)
	'''
	rename eaf files as audio files
	'''
	# Regular expression to match and extract the desired part of the filename
	pattern = re.compile(r'(?i)([a-z]+)-(\d+)')

	# Iterate over the files in the directory
	for filename in os.listdir(eaf_directory):
		if filename.endswith('.eaf'):
			match = pattern.search(filename)
			if match:
				new_filename = f"{match.group(1).upper()}-{match.group(2)}.eaf"
				old_filepath = os.path.join(eaf_directory, filename)
				new_filepath = os.path.join(eaf_directory, new_filename)
				os.rename(old_filepath, new_filepath)
				print(f"Renamed {filename} to {new_filename}")

def convert_and_rename_audio(filename: str):
	'''
	convert an audio file into .wav format and rename it
	'''
    # Rename the file
    match = pattern.search(filename)
    if not match:
        return

    new_filename = f"{match.group(1).upper()}-{match.group(2)}.wav"
    old_filepath = os.path.join(audio_directory, filename)
    new_filepath = os.path.join(wav_directory, new_filename)

    # Convert the audio file to .wav format
    audio = AudioSegment.from_file(old_filepath)
    audio.export(new_filepath, format='wav')
    print(f"Converted and renamed {filename} to {new_filename}")

def process_audio_dir(audio_directory: str):
	'''
	convert and rename all audio files inside the directory
	'''
	# Regular expression to match and extract the desired part of the filename
	pattern = re.compile(r'(?i)([a-z]+)-(\d+)')

	# Iterate over the files in the directory
	for filename in os.listdir(audio_directory):
		if filename.endswith('.m4a') or filename.endswith('.MP3'):
			convert_and_rename_audio(filename)

def parse_eaf_file(eaf_file_path: str, audio_file_path: str, audio_filename: str):
	'''
	parse an eaf file and slice audio according to the timestamps
	'''
    # Load and parse the .eaf file
    tree = ET.parse(eaf_file_path)
    root = tree.getroot()

    # Load the audio file and create corresponding directory
    audio = AudioSegment.from_wav(audio_file_path)
    os.mkdir(output_directory + '/' + audio_filename)

    # Extract time slots
    time_slots = {}
    for time_slot in root.findall(".//TIME_SLOT"):
        time_slots[time_slot.attrib["TIME_SLOT_ID"]] = int(time_slot.attrib["TIME_VALUE"])

    # Function to slice and save audio segments
    def process_tier(tier):
        participant = tier.attrib["PARTICIPANT"]
        for annotation in tier.findall(".//ALIGNABLE_ANNOTATION"):
            start = time_slots[annotation.attrib["TIME_SLOT_REF1"]]
            end = time_slots[annotation.attrib["TIME_SLOT_REF2"]]
            segment = audio[start:end]
            output_path = f"{output_directory}/{audio_filename}/{participant}_{start}_{end}.wav"
            segment.export(output_path, format="wav")

    # Process each tier in the .eaf file
    for tier in root.findall(".//TIER"):
        process_tier(tier)

def process_eaf_dir(eaf_directory: str, audio_directory: str)
	'''
	process all eaf files in the directory
	'''
	for eaf_filename in os.listdir(eaf_directory):
		if eaf_filename.endswith('.eaf'):
			eaf_file_path = os.path.join(eaf_directory, eaf_filename)
			audio_filename = os.path.splitext(eaf_filename)[0] + '.wav'
			audio_file_path = os.path.join(audio_directory, audio_filename)

			if os.path.exists(audio_file_path):
				print(f"Processing {eaf_filename} with {audio_filename}")
				process_eaf_file(eaf_file_path, audio_file_path, audio_filename)
			else:
				print(f"Audio file not found for {eaf_filename}: {audio_filename}")

if __name__ == "__main__":
	eaf_directory = './eaf'
	audio_directory = './audio/m4a'
	wav_directory = './audio/wav'
	output_directory = './audio/seg'

	rename_eaf(eaf_directory)
	process_audio_dir(audio_directory)
	process_eaf_dir(eaf_directory, audio_directory)
