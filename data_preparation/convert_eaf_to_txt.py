import os
from pympi import Eaf

def convert_eaf_to_txt(eaf_folder, output_folder):
    for file_name in os.listdir(eaf_folder):
        if file_name.endswith('.eaf'):
            eaf_file = os.path.join(eaf_folder, file_name)
            txt_file = os.path.splitext(file_name)[0] + '.txt'
            txt_file_path = os.path.join(output_folder, txt_file)
            convert_single_eaf(eaf_file, txt_file_path)

def convert_single_eaf(eaf_file, txt_file_path):
    eaf = Eaf(eaf_file)
    tier_names = list(eaf.get_tier_names())

    texts = []
    for tier_name in tier_names:
        annotations = eaf.get_annotation_data_for_tier(tier_name)
        text = ' '.join([annotation[-1] for annotation in annotations])
        texts.append(text)
    text = ' '.join(texts)
    with open(txt_file_path, 'w', encoding='utf-8') as file:
        file.write(text)
        print(f"Conversion completed: {eaf_file} -> {txt_file}")



eaf_folder_path = "./transcriptions_eaf"
txt_folder_path = "./transcr_txt"

convert_eaf_to_txt(eaf_folder_path, txt_folder_path)
