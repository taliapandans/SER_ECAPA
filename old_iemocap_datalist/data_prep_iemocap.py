import pandas as pd
import csv
import os

data_file = './data.txt'
train_path = './data/IEMOCAP_full_release'
# num_frames = 200

# train_list = 'data_train.txt'
# Load data & labels
# data_list  = []
# data_label = []
# lines = open(data_file).read().splitlines()
# dictkeys = list(set([x.split()[0] for x in lines]))
# dictkeys.sort()
# dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
# for index, line in enumerate(lines):
# 	emo_label = dictkeys[line.split()[0]]
# 	file_name     = os.path.join(train_path, line.split()[1])
# 	data_label.append(emo_label)
# 	data_list.append(file_name)

# df = pd.read_csv('iemocap_full_dataset.csv')
# emo = df['emotion']
# audio_path = df['path']

# inside_csv = []
# emo_list = []
# path_list = []

# with open('iemocap_full_dataset.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     line_count = 0
#     for row in csv_reader:
#         inside_csv.append(row)

# for row in inside_csv:
#     if not row[3] == "xxx":
#         emo_list.append(row[3])
#         path_list.append(row[6])

# # for idx, content in enumerate(emo_list):
# #     if content == 'xxx':
# #         emo_list.pop(idx)
# #         path_list.pop(idx)

# new_df = pd.DataFrame(list(zip(emo_list, path_list)))
# new_df.to_csv('emo_label.txt', sep='\t', index=False)

path_a = 'train_emo_label.txt'
path_b = 'test_emo_label.txt'

def four_labels(file_path, output_file):
    clean_files = []

    lines = open(file_path, "r")
    for line in lines:
        label = line.split()[0]

        if label != 'fru':
            if label != 'sur':
                if label == 'exc':
                    clean_files.append(line.replace('exc', 'hap'))
                else:
                    clean_files.append(line)

    return clean_files

def num_labels(file_path, output_file):
    new_label = []

    lines = open(file_path, "r")
    for line in lines:
        label = line.split()[0]
        if label == 'neu':
            new_label.append(line.replace('neu', '0'))
        elif label == 'hap':
            new_label.append(line.replace('hap', '1'))
        elif label == 'sad':
            new_label.append(line.replace('sad', '2'))
        elif label == 'ang':
            new_label.append(line.replace('ang', '3'))
    
    return new_label

def get_speaker_id(file_path, output_file=''):
    speaker_id_list = []

    lines = open(file_path, "r")
    for line in lines:
        wav = line.split()[1]
        session = wav.split('/')[3]
        speaker_id = session.split('_')[0]
        speaker_id_list.append(speaker_id)
    return speaker_id_list

def write_to_txt(data, output_file):
    with open(output_file, 'w') as output:
            for row in data:
                output.write(str(row))

# num_labels("train_filtered.txt", "data_train.txt")
get_speaker_id(path_b)