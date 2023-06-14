import os
import pympi


def batch_convert_eaf_to_txt(eaf_folder_path, txt_folder_path):
    """
    Batch convert EAF files to TXT format.
    eaf_folder_path: str, path to the folder containing EAF files
    txt_folder_path: str, path to the folder to save the converted TXT files
    """
    os.makedirs(txt_folder_path, exist_ok=True)

    # Get a list of all EAF files in the folder
    eaf_files = [
        file for file in os.listdir(eaf_folder_path) if file.endswith(".eaf")
    ]

    for eaf_file in eaf_files:
        eaf_file_path = os.path.join(eaf_folder_path, eaf_file)
        txt_file_name = os.path.splitext(eaf_file)[0] + ".txt"
        txt_file_path = os.path.join(txt_folder_path, txt_file_name)

        convert_eaf_to_txt(eaf_file_path, txt_file_path)


def convert_eaf_to_txt(eaf_file_path, txt_file_path):
    """
    Convert EAF file to TXT format.
    eaf_file_path: str, path to the EAF file
    txt_file_path: str, path to save the converted TXT file
    """
    eaf = pympi.Elan.Eaf(eaf_file_path)
    tiers = eaf.get_tier_names()

    with open(txt_file_path, "w", encoding="utf-8") as txt_file:
        for tier in tiers:
            annotations = eaf.get_annotation_data_for_tier(tier)
            for annotation in annotations:
                start_time = annotation[0]
                end_time = annotation[1]
                annotation_text = annotation[2]
                txt_file.write(f"{start_time}\t{end_time}\t{annotation_text}\n")

    print(f"Conversion completed: {eaf_file_path} -> {txt_file_path}")


eaf_folder_path = "./transcriptions_eaf"
txt_folder_path = "./transcr_txt"

batch_convert_eaf_to_txt(eaf_folder_path, txt_folder_path)
