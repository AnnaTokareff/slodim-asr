import os
import json
from os.path import isfile, join


def load_paths(audio_folder, text_folder):
    """
    load paths of all audio and text files inside the folders
    audio_folder: str, ONLY folder name
    text_folder: str, ONLY folder name
    return: audio_paths, text_paths ( list(str) )
    """
    print("Loading paths...")

    audio_paths = [
        join(f"./{audio_folder}", f)
        for f in os.listdir(f"./{audio_folder}")
        if isfile(join(f"./{audio_folder}", f))
    ]

    text_paths = [
        join(f"./{text_folder}", f)
        for f in os.listdir(f"./{text_folder}")
        if isfile(join(f"./{text_folder}", f))
    ]

    return audio_paths, text_paths


def load_target(text_paths):
    """
    load target text files
    text_paths: list(str)
    return: targets ( list(str) )
    """
    print("Loading targets...")
    targets = []

    for path in text_paths:
        with open(path, "r", encoding="utf8") as f:
            targets.append(f.read())

    return targets


def save_as_json_file(res_dict, audio_name):
    """
    save the results of dict format to json
    res_dict: dict, results of processing one audio file
    audio_name: str, can be path to one file or just a name
    """
    with open(f'{audio_name}_transcribed.json', 'w') as json_file:
      json.dump(res_dict, json_file)
    print("json file saved!")

def save_as_txt_file(text, audio_name):
    """save the transcriptions to txt file
    text: str, transcriptions
    audio_name: str, can be path to one file or just a name
    """
    with open(f'{audio_name}_transcribed.txt', 'w') as txt_file:
      txt_file.write(text)
    print("txt file saved!")
