''' This Script cleans all filenames from problematic characters in the dataset'''

import os
import re
import shutil

def clean_filename(filename):
    replacements = {
        'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss',
        'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue',
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a',
        'ç': 'c', 'ñ': 'n'
    }
    cleaned = filename
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)

    cleaned = re.sub(r'[^a-zA-Z0-9\-_.]', '_', cleaned)

    cleaned = re.sub(r'_+', '_',cleaned)

    return cleaned

def fix_all_filenames(data_dir):
    '''fix all filenames in data'''
    fixed_count = 0
    
    for image_class in os.listdir(data_dir):
        image = os.path.join(data_dir, image_class)

        if not os.path.isdir(image):
            continue 

        #print (f"Bearbeitete Ordner: {image_class}")

        for filename in os.listdir(image):
            old_path = os.path.join(image, filename)
            clean_name = clean_filename(filename)

            if clean_name != filename:
                new_path = os.path.join(image, clean_name)

                counter = 1
                base_name, ext = os.path.splitext(clean_name)
                while os.path.exists(new_path):
                    clean_name = f"{base_name}_{counter}{ext}"
                    new_path = os.path.joi(image, clean_name)
                    counter += 1

                try:
                    os.rename(old_path, new_path)
                    #print (f"Umbenannt: {filename} -> {clean_name}")
                    fixed_count += 1
                except Exception as e:
                    print(f"Fehler beim Umbenennen {filename}:{e}")
    #print(f"\nInsgesamt {fixed_count} Dateien umbenannt.")
    #return fixed_count

if __name__ == "__main__":
    data_dir = 'data'
    fix_all_filenames(data_dir)

            

