import os
import re
from pydub import AudioSegment


class DataPreprocessor:
    def __init__(self, eaf_dir, m4a_dir, wav_dir):
        self.eaf_dir = eaf_dir
        self.m4a_dir = m4a_dir
        self.wav_dir = wav_dir

    def _get_eaf_new_name(self, filename):
        '''
        return formatted filename of eaf file by retrieving root id and session number
        '''
        pattern = r"P\d+\s*-\s*([a-zA-Z]+-\d+)"
        match = re.match(pattern, filename, re.IGNORECASE)

        if match:
            root_id = match.group(1).upper()
            # Find the session number
            if "entr2" in filename.lower():
                session = "2"
            elif "eye tracker" in filename.lower():
                session = "2"
            else:
                session = "1"

            return f"{root_id}-{session}.eaf"
        else:
            print(f"{filename} no match")
            return None

    def _get_audio_new_name_and_ext(self, filename):
        '''
        return new filename based on root id and session number, and extension
        '''
        pattern = r"([a-zA-Z]+-\d+)(?:-Entr(\d)|\s*Eye\s*Tracker)?(?:-Audio|-audio)?\.([a-zA-Z0-9]+)"
        match = re.match(pattern, filename, re.IGNORECASE)
        
        if match:
            root_id = match.group(1).upper()
            ext = match.group(3).lower()
            # Find the session number
            if match.group(2):
                session = match.group(2)
            elif "eye tracker" in filename.lower():
                session = "2"
            else:
                session = "1"
            return f"{root_id}-{session}.wav", ext
        else: 
            print(f"{filename} no match")
            return None, None
    
    def rename_eaf(self, filename):
        '''
        rename eaf file to format 'ABC-001-1.eaf'
        '''
        new_name = self._get_eaf_new_name(filename)
        if new_name:
            os.rename(os.path.join(self.eaf_dir, filename), os.path.join(self.eaf_dir, new_name))

    def convert_and_rename_audio(self, filename):
        '''
        convert audio to wav and rename as 'ABC-001-1.wav'
        '''
        new_name, ext = self._get_audio_new_name_and_ext(filename)
        if new_name and ext:
            input_path = os.path.join(self.m4a_dir, filename)
            output_path = os.path.join(self.wav_dir, new_name)
            audio = AudioSegment.from_file(input_path, ext)
            audio.export(output_path, format="wav")

    def rename_and_convert_files(self):
        for eaf_file in os.listdir(self.eaf_dir):
            self.rename_eaf(eaf_file)
        print(".eaf file renaming finished")

        for audio_file in os.listdir(self.m4a_dir):
            self.convert_and_rename_audio(audio_file)
        print("Audio file renaming and conversion finished")


if __name__ == "__main__":
    eaf_dir = "./eaf"
    m4a_dir = "./audio/m4a"
    wav_dir = "./audio/wav"

    data_preprocessor = DataPreprocessor(eaf_dir, m4a_dir, wav_dir)
    data_preprocessor.rename_and_convert_files()