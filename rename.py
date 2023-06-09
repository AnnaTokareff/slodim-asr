import os
import re

class Renamer:
    def __init__(self, eaf_dir, wav_dir):
        self.eaf_dir = eaf_dir
        self.wav_dir = wav_dir


    def _get_eaf_new_name(self, filename):
        pattern = r"P\d+\s*-\s*([a-z]+)-(\d+).*\.eaf"
        match = re.match(pattern, filename, re.IGNORECASE)
        if match:
            prefix = match.group(1).upper()
            number = match.group(2)
            transformed_filename = f"{prefix}-{number}.eaf"
            return transformed_filename
        else:
            return None


    def _get_audio_new_name(self, filename):
        '''
        Return new filename based on root ID and session number, and extension.
        '''
        pattern = r"P(\d+)\s*-\s*([a-zA-Z]+-\d+)(?:\s*-\s*(?:entr(\d)|eye\stracker))?(?:-audio)?\.([a-zA-Z0-9]+)"
        match = re.match(pattern, filename, re.IGNORECASE)

        if match:
            root_id = match.group(2).upper()
            ext = match.group(4).lower()
            # Find the session number
            if match.group(3):
                session = match.group(3)
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
        Rename eaf file to format 'ABC-001-1.eaf'.
        '''
        new_name = self._get_eaf_new_name(filename)
        if new_name:
            os.rename(os.path.join(self.eaf_dir, filename), os.path.join(self.eaf_dir, new_name))

    def rename_audio(self, filename):
        '''
        Rename audio file to format 'ABC-001-1.wav'.
        '''
        new_name, ext = self._get_audio_new_name(filename)
        if new_name and ext:
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
