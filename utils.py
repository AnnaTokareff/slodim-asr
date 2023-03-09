import os
import json
from os.path import isfile, join


def load_paths(audio_folder, text_folder):
    """
    load paths of all audio and text files inside the folders
    audio_folder: str, ONLY folder name
    text_folder: str, ONLY folder name
    return: sorted audio_paths, text_paths ( list(str) )
    """
    print("Loading paths...")

    audio_paths = [
        join(f"./{audio_folder}", f)
        for f in os.listdir(f"./{audio_folder}")
        if isfile(join(f"./{audio_folder}", f))
    ].sort()

    text_paths = [
        join(f"./{text_folder}", f)
        for f in os.listdir(f"./{text_folder}")
        if isfile(join(f"./{text_folder}", f))
    ].sort()

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


def save_as_file(res_dict, audio_name, to_json=True, to_txt=True):
    """
    save the results of dict format to json and text format to txt
    res_dict: dict, results of processing one audio file
    audio_name: str, can be path to one file or just a name
    """
    if to_json == True:
        with open(f"{audio_name}_transcribed.json", "w") as json_file:
            json.dump(res_dict, json_file)
        print("JSON file saved!")

    if to_txt == True:
        with open(f"{audio_name}_transcribed.txt", "w") as txt_file:
            txt_file.write(res_dict["text"])
        print("TXT file saved!")


def convert_into_right_format_whisperX(result_aligned):
    """
    function for converting res_aligned
    into a proper format for avoiding errors
    """

    for i, el in enumerate(result_aligned["segments"]):
        el["word-segments"] = el["word-segments"].to_dict()
        el["char-segments"] = el["char-segments"].to_dict()
    return result_aligned
