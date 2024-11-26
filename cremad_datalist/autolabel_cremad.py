import os
import pandas as pd

data_path = './data/CremaD'
wav_files_path_list = []
emo_label_list = []

file_path = os.listdir(data_path)

for file in file_path:
    emo_label = file.split("_")[2]
    if emo_label == 'NEU':
        new_label = '0'
        emo_label_list.append(new_label)
        wav_files_path_list.append(file)
    if emo_label == 'HAP':
        new_label = '1'
        emo_label_list.append(new_label)
        wav_files_path_list.append(file)
    if emo_label == 'SAD':
        new_label = '2'
        emo_label_list.append(new_label)
        wav_files_path_list.append(file)
    if emo_label == 'ANG':
        new_label = '3'
        emo_label_list.append(new_label)
        wav_files_path_list.append(file)

df = pd.DataFrame(list(zip(emo_label_list, wav_files_path_list)))
df.to_csv("cremaD_full_dataset", sep='\t', index=False)






