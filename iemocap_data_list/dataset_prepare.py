import pandas as pd
import csv
import random

# lines 1: session, lines 2: method, lines 3: gender
# lines 4: emotion, lines 5: n_annotators
# lines 6: agreement, lines 7: wav path

full_list = "./iemocap_full_dataset.csv"

# get list of all emotion and wav file except xxx
def get_all_emo_datalist(full_list):
    wav_file_path_list = []
    emo_label_list = []
    with open(full_list, mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
                if lines[3] != "xxx":
                    emo_label_list.append(lines[3])
                    wav_file_path_list.append(lines[6])
    return emo_label_list, wav_file_path_list

# get list of four emotions and wav file
def get_four_emo_datalist(full_list):
    wav_file_path_list = []
    emo_label_list = []
    with open(full_list, mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            if lines[3] not in ['xxx', 'fru', 'sur', 'fea', 'dis', 'oth']:
                if lines[3] == 'exc':
                    emo_label_list.append(lines[3].replace('exc', 'hap'))
                    wav_file_path_list.append(lines[6])
                else:
                    emo_label_list.append(lines[3])
                    wav_file_path_list.append(lines[6])
    return emo_label_list, wav_file_path_list

# convert label str to int
def num_labels(emo_label_list):
    new_label_list = []

    for label in emo_label_list:
        if label == 'neu':
            new_label_list.append(label.replace('neu', '0'))
        elif label == 'hap':
            new_label_list.append(label.replace('hap', '1'))
        elif label == 'sad':
            new_label_list.append(label.replace('sad', '2'))
        elif label == 'ang':
            new_label_list.append(label.replace('ang', '3'))
        elif label == 'dis':
            new_label_list.append(label.replace('dis', '4'))
        elif label == 'fea':
            new_label_list.append(label.replace('fea', '5'))
    return new_label_list

# get k5 datalist
def get_k5_datalist(emo_label_list, wav_file_path_list):
    if len(wav_file_path_list[0]) < 8:
        emo_label_list = emo_label_list[1:]
        wav_file_path_list = wav_file_path_list[1:]

    for i in range(1,6):
        session_wav_list = []
        session_out_wav_list = []
        label_session_wav_list = []
        label_session_out_wav_list = []

        session = "Session" + str(i)

        for file in wav_file_path_list:
            # index_file = wav_file_path_list.index(file)
            if session in file:
                session_wav_list.append(file)
                label_session_wav_list.append(emo_label_list[wav_file_path_list.index(file)])
            else:
                session_out_wav_list.append(file)
                label_session_out_wav_list.append(emo_label_list[wav_file_path_list.index(file)])

        session_test_file = "ses%d_test.txt"%(i)
        session_train_file = "ses%d_train.txt"%(i)
        session_val_file = "ses%d_val.txt"%(i)

        int_label_session_wav_list = num_labels(label_session_wav_list)
        int_label_session_out_wav_list = num_labels(label_session_out_wav_list)

        df_test = pd.DataFrame(list(zip(int_label_session_wav_list, session_wav_list)))

        df_rest = pd.DataFrame(list(zip(int_label_session_out_wav_list, session_out_wav_list)))
        df_train = df_rest.sample(frac=0.8,random_state=200)
        df_val = df_rest.drop(df_train.index)

        df_test.to_csv(session_test_file, sep='\t', index=False, header=None)
        df_train.to_csv(session_train_file, sep='\t', index=False, header=None)
        df_val.to_csv(session_val_file, sep='\t', index=False, header=None)

#get speaker id
def get_speaker_id(emo_label_list, wav_file_path_list):
    if len(wav_file_path_list[0]) < 8:
        emo_label_list = emo_label_list[1:]
        wav_file_path_list = wav_file_path_list[1:]
        
    speaker_id_list = []

    for file in wav_file_path_list:
        speaker_id = file[file.index('wav/Ses')+len('wav/Ses'):file.index('_')]
        if speaker_id not in speaker_id_list:
            speaker_id_list.append(speaker_id)
    
    return speaker_id_list


# get sp idp k10 datalist 
def get_k10_datalist(emo_label_list, wav_file_path_list, speaker_id_list):
    wav_file_train = []
    wav_file_val = []
    wav_file_test = []

    label_train = []
    label_val = []
    label_test = []

    speaker_list_test = speaker_id_list[:]
    speaker_list_val = []
    speaker_list_train = []

    speaker_list_val_temp = speaker_id_list[:]

    for index in range(len(speaker_list_val_temp)):

        randindex_val = random.randrange(len(speaker_list_val_temp))
        while(speaker_list_test[index] == speaker_list_val_temp[randindex_val]):
            randindex_val = random.randrange(len(speaker_list_val_temp))
        
        speaker_list_val.append(speaker_list_val_temp[randindex_val])
        
        speaker_list_train_temp = speaker_id_list[:]
        speaker_list_train_temp.remove(speaker_list_test[index])
        speaker_list_train_temp.remove(speaker_list_val_temp[randindex_val])
        speaker_list_train.append(speaker_list_train_temp)

        del speaker_list_val_temp[randindex_val]

    with open('speaker_list_train.txt', 'w') as f:
        for line in speaker_list_train:
            f.write(f"{','.join(line)}\n")

    with open('speaker_list_val.txt', 'w') as f:
        for line in speaker_list_val:
            f.write(f"{line}\n")

    with open('speaker_list_test.txt', 'w') as f:
        for line in speaker_list_test:
            f.write(f"{line}\n")


    for i in range(len(speaker_id_list)):
        wav_file_train = []
        wav_file_val = []
        wav_file_test = []

        label_train = []
        label_val = []
        label_test = []

        for file in wav_file_path_list:
            if len(wav_file_path_list[0]) < 8:
                emo_label_list = emo_label_list[1:]
                wav_file_path_list = wav_file_path_list[1:]
            
            if speaker_id_list[i] in file:
                wav_file_test.append(file)
                label_test.append(emo_label_list[wav_file_path_list.index(file)])
            
            if speaker_list_val[i] in file:
                wav_file_val.append(file)
                label_val.append(emo_label_list[wav_file_path_list.index(file)])

            for j in range(len(speaker_list_train[i])):
                if speaker_list_train[i][j] in file:
                    wav_file_train.append(file)
                    label_train.append(emo_label_list[wav_file_path_list.index(file)])

        train_file_name = "k%d_fold_train.txt"%(i+1)
        val_file_name = "k%d_fold_val.txt"%(i+1)
        test_file_name = "k%d_fold_test.txt"%(i+1)

        int_label_train = num_labels(label_train)
        int_label_val = num_labels(label_val)
        int_label_test = num_labels(label_test)

        df_train = pd.DataFrame(list(zip(int_label_train, wav_file_train)))
        df_val = pd.DataFrame(list(zip(int_label_val, wav_file_val)))
        df_test = pd.DataFrame(list(zip(int_label_test, wav_file_test)))

        df_train.to_csv(train_file_name, sep='\t', index=False, header=None)
        df_val.to_csv(val_file_name, sep='\t', index=False, header=None)
        df_test.to_csv(test_file_name, sep='\t', index=False, header=None)


emo_label_list, wav_file_path_list = get_four_emo_datalist(full_list)

speaker_id_list = get_speaker_id(emo_label_list, wav_file_path_list)

get_k10_datalist(emo_label_list, wav_file_path_list, speaker_id_list)


# # remove column name
# df = pd.DataFrame(list(zip(emo_label_list[1:], wav_file_path_list[1:])))
# df.to_csv('iemocap_four_emotion.csv', sep='\t', index=False, header=None)
