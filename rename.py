import os
import re

class Renamer:
    def __init__(self, eaf_dir, wav_dir):
        self.eaf_dir = eaf_dir
        self.wav_dir = wav_dir

    def _get_eaf_new_name(self, filename):
        '''
        Return formatted filename of eaf file by retrieving root id and session number.
        '''
        pattern = r"([a-zA-Z]+-\d+)(?:-Entr(\d)|\s*Eye\s*Tracker)?(?:-Audio)?\.eaf"
        match = re.match(pattern, filename, re.IGNORECASE)

        if match:
            root_id = match.group(1).upper()
            session = match.group(2) if match.group(2) else "1"
            return f"{root_id}-{session}.eaf"
        else:
            print(f"{filename} no match")
            return None

    def _get_audio_new_name(self, filename):
        '''
        Return new filename based on root id and session number for audio file.
        '''
        pattern = r"([a-zA-Z]+-\d+)(?:-Entr(\d)|\s*Eye\s*Tracker)?(?:-Audio)?\.([a-zA-Z0-9]+)"
        match = re.match(pattern, filename, re.IGNORECASE)

        if match:
            root_id = match.group(1).upper()
            session = match.group(2) if match.group(2) else "1"
            return f"{root_id}-{session}.wav"
        else:
            print(f"{filename} no match")
            return None

    def rename_eaf(self, filename):
        '''
        Rename eaf file to format 'ABC-001-1.eaf'.
        '''
        new_name = self._get_eaf_new_name(filename)
        if new_name:
            os.rename(os.path.join(self.eaf_dir, filename), os.path.join(self.eaf_dir, new_name))

    def rename_audio(self, filename):
        '''
        Rename audio file to format 'ABC-001-1.wav'.
        '''
        new_name = self._get_audio_new_name(filename)
        if new_name:
            os.rename(os.path.join(self.wav_dir, filename), os.path.join(self.wav_dir, new_name))

    def rename_files(self):
        for eaf_file in os.listdir(self.eaf_dir):
            self.rename_eaf(eaf_file)
        print(".eaf file renaming finished")

        for audio_file in os.listdir(self.wav_dir):
            self.rename_audio(audio_file)
        print("Audio file renaming finished")


if __name__ == "__main__":
    eaf_dir = "./transcriptions"
    wav_dir = "./processed_audio"

    renamer = Renamer(eaf_dir, wav_dir)
    renamer.rename_files()
