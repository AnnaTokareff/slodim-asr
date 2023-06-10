import os
import re


class Renamer:
    def __init__(self, eaf_dir, wav_dir):
        self.eaf_dir = eaf_dir
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
            if "entr2" in filename.lower() or "eye tracker" in filename.lower():
                session = "2"
            else:
                session = "1"

            return f"{root_id}-{session}.eaf"
        else:
            print(f"{filename} no match")
            pass

    def _get_wav_new_name(self, filename):
        '''
        return formatted filename of wav file by retrieving root id and session number
        '''
        pattern = r"P\d+\s*-\s*([a-zA-Z]+-\d+)(?:-Entr(\d)|\s*Eye\s*Tracker)?(?:-Audio)?\.wav"
        match = re.match(pattern, filename, re.IGNORECASE)

        if match:
            root_id = match.group(1).upper()
            # Find the session number
            if match.group(2):
                session = match.group(2)
            elif "eye tracker" in filename.lower():
                session = "2"
            else:
                session = "1"
            return f"{root_id}-{session}.wav"
        else:
            # Modify the filename to match the desired format
            pattern1 = r"([a-zA-Z]+-\d+)(?:-Entr(\d)|-Entr(\d)-Audio)?\.wav"
            match1 = re.match(pattern1, filename, re.IGNORECASE)

            if match1:
                root_id = match1.group(1).upper()
                session = match1.group(2) or match1.group(3) or ""
                return f"{root_id}-{session}.wav"
            else:
                # Modify the filename for BOC-066 Eye Tracker.wav
                pattern2 = r"([a-zA-Z]+-\d+)\s*Eye\s*Tracker\.wav"
                match2 = re.match(pattern2, filename, re.IGNORECASE)

                if match2:
                    root_id = match2.group(1).upper()
                    return f"{root_id}-ET.wav"
                else:
                    print(f"{filename} no match")
                    pass

    def rename_eaf(self, filename):
        '''
        rename eaf file to format 'ABC-001-1.eaf'
        '''
        new_name = self._get_eaf_new_name(filename)
        if new_name:
            os.rename(os.path.join(self.eaf_dir, filename), os.path.join(self.eaf_dir, new_name))

    def rename_wav(self, filename):
        '''
        rename wav file to format 'ABC-001-1.wav'
        '''
        new_name = self._get_wav_new_name(filename)
        if new_name:
            os.rename(os.path.join(self.wav_dir, filename), os.path.join(self.wav_dir, new_name))

    def rename_files(self):
        for eaf_file in os.listdir(self.eaf_dir):
            self.rename_eaf(eaf_file)

if __name__ == "__main__":
    eaf_dir = "./transcriptions"
    wav_dir = "./processed_audio"

    renamer = Renamer(eaf_dir, wav_dir)
    renamer.rename_files()
