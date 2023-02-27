import os
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

    try:
        audio_paths.remove(join(f"./{audio_folder}", "BOC-066_5min.m4a"))
        audio_paths.remove(join(f"./{audio_folder}", "BOC-066_spaced.wav"))
        audio_paths.remove(join(f"./{audio_folder}", "temp.wav"))
    except:
        pass

    text_paths = [
        join(f"./{text_folder}", f)
        for f in os.listdir(f"./{text_folder}")
        if isfile(join(f"./{text_folder}", f))
    ]

    try:
        text_paths.remove(join(f"./{text_folder}", "P39682 - boc-066.zip (1)_5min.txt"))
    except:
        pass

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
