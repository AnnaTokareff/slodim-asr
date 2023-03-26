import os
import re
import shutil
from pydub import AudioSegment
from xml.etree import ElementTree


class DataPreprocessor:
    def __init__(self, eaf_directory, audio_directory):
        self.eaf_directory = eaf_directory
        self.audio_directory = audio_directory

    def rename_eaf(self):
        for filename in os.listdir(self.eaf_directory):
            if filename.endswith('.eaf'):
                new_filename = re.sub(r'(P\d+ - )|( \(1\).eaf)', '', filename).replace(' ', '').replace('.zip', '').upper()
                shutil.move(os.path.join(self.eaf_directory, filename), os.path.join(self.eaf_directory, new_filename))

    def convert_and_rename_audio(self):
        for filename in os.listdir(self.audio_directory):
            if filename.lower().endswith(('.m4a', '.mp3')):
                audio_format = filename.split('.')[-1].lower()
                audio = AudioSegment.from_file(os.path.join(self.audio_directory, filename), format=audio_format)

                new_filename = re.sub(r'(-Entr\d-Audio)|(-Entr\d-Audio)', '', filename)
                new_filename = new_filename.replace(' ', '').replace('.MP3', '.wav').replace('.m4a', '.wav').replace('.mp3', '.wav')

                if '-Entr' in filename:
                    entry_number = re.search(r'Entr(\d+)', filename).group(1)
                    new_filename = re.sub(r'\.wav', f'-{entry_number}.wav', new_filename)

                audio.export(os.path.join(self.audio_directory, new_filename), format='wav')
                os.remove(os.path.join(self.audio_directory, filename))

    def parse_eaf_file(self, eaf_file):
        eaf_tree = ElementTree.parse(os.path.join(self.eaf_directory, eaf_file))
        root = eaf_tree.getroot()
        time_slots = {slot.attrib['TIME_SLOT_ID']: int(slot.attrib['TIME_VALUE']) for slot in root.iter('TIME_SLOT')}

        segments = []
        for tier in root.iter('TIER'):
            for annotation in tier.iter('ANNOTATION'):
                alignable_annotation = annotation.find('ALIGNABLE_ANNOTATION')
                if alignable_annotation is not None:
                    start_slot = alignable_annotation.attrib['TIME_SLOT_REF1']
                    end_slot = alignable_annotation.attrib['TIME_SLOT_REF2']
                    start_time = time_slots[start_slot]
                    end_time = time_slots[end_slot]
                    transcription = alignable_annotation.find('ANNOTATION_VALUE').text
                    segments.append((start_time, end_time, transcription))

        return segments

	def slice_and_save_audio(self, audio_file, segments, output_directory):
        audio = AudioSegment.from_wav(os.path.join(self.audio_directory, audio_file))
        for start_time, end_time, _ in segments:
            sliced_audio = audio[start_time:end_time]
            output_filename = f"{os.path.splitext(audio_file)[0]}_{start_time}_{end_time}.wav"
            sliced_audio.export(os.path.join(output_directory, output_filename), format="wav")


if __name__ == "__main__":
    preprocessor = DataPreprocessor("./eaf", "./audio/m4a")

    # Rename eaf files
    preprocessor.rename_eaf()

    # Convert and rename audio files
    preprocessor.convert_and_rename_audio()

    # Process all eaf and audio files
    for eaf_file in os.listdir("./eaf"):
        audio_file_name = os.path.splitext(eaf_file)[0] + ".wav"
        
        # Parse eaf file
        segments = preprocessor.parse_eaf_file(eaf_file)

        # Slice and save audio segments
        preprocessor.slice_and_save_audio(audio_file_name, segments, "./audio/seg")
