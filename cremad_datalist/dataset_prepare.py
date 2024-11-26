import os
import pandas as pd
import random
import csv

def extract_data(data_file):
    lines = open(data_file, "r")
    all_data = lines.readlines()
    return all_data

def randomize_speaker(all_data):
    speaker_id_list = []

    for line in all_data: 
        filename = line.split()[1]
        speaker_id = filename.split("_")[0]
        speaker_id_list.append(speaker_id) if speaker_id not in speaker_id_list else None

    random.shuffle(speaker_id_list)
    train_speaker = speaker_id_list[:64]
    val_speaker = speaker_id_list[64:72]
    test_speaker = speaker_id_list[72:91]

    return train_speaker, val_speaker, test_speaker

def split_list(speaker_list, all_data, output_file):
    wav_file_list = []
    label_list = []

    for speaker in speaker_list:
        for data in all_data:
            if speaker in data:
                # label_list.append(data[0])
                # wav_file = data.split('\t')[-1].split('\n')[0]
                # wav_file_list.append(wav_file)
                wav_file_list.append(data)
    
    # df = pd.DataFrame(list(zip(label_list, wav_file_list)))
    # df.to_csv(output_file, sep='\t', index=False)
    txt_file = open(output_file, 'w')
    txt_file.writelines(wav_file_list)
    

data_file = "./cremaD_full_dataset.txt"
all_data = extract_data(data_file)
train_speaker = []
val_speaker = []
test_speaker = []
train_speaker, val_speaker, test_speaker = randomize_speaker(all_data)

# speaker_lists = pd.DataFrame(list(zip(train_speaker, val_speaker, test_speaker)))
pd_train_speaker = pd.DataFrame({'train': train_speaker})
pd_val_speaker = pd.DataFrame({'val': val_speaker})
pd_test_speaker = pd.DataFrame({'test': test_speaker})
speaker_lists = pd.concat([pd_train_speaker, pd_test_speaker, pd_val_speaker], ignore_index=False, axis=1)
# print(speaker_lists)
speaker_lists.index +=1
speaker_lists.to_csv('k5_speaker.txt', sep='\t')

split_list(train_speaker, all_data, 'k5_train_list.txt')
split_list(val_speaker, all_data, 'k5_val_list.txt')
split_list(test_speaker, all_data, 'k5_test_list.txt')


