import os
import pandas as pd
import random

def get_audio_files_list(data_path):
    # folder_path_1 = "Session1/sentences/wav"
    # folder_path_2 = "Session2/sentences/wav"
    # folder_path_3 = "Session3/sentences/wav"
    folder_path_4 = "Session4/sentences/wav"
    folder_path_5 = "Session5/sentences/wav"

    # folder_path = [folder_path_1, folder_path_2, folder_path_3, 
    #             folder_path_4, folder_path_5]
    folder_path = [folder_path_4, folder_path_5]

    wav_files_path_list = []

    ## save all audio path
    for path in folder_path:
        dataset_path = data_path + '/' + path
        dataset_folders = os.listdir(dataset_path)
        for folder in dataset_folders:
            if not folder.startswith('._'):
                wav_folder_path = dataset_path + '/' + folder
                wav_files = os.listdir(wav_folder_path)
                for file in wav_files:
                    if not file.startswith('._'):
                        # wav_file_path = wav_folder_path + '/' + file
                        wav_file_path = path + '/' + folder + '/' + file
                        if wav_file_path.endswith('.wav'):
                            wav_files_path_list.append(wav_file_path)    
    return wav_files_path_list

def get_speaker_id(audio_files_list):
    ## speaker id
    ## gender -8 from last string
    speaker_ids_list = []
    for file in audio_files_list:
        gender = file[-8]
        gender_id = file.split("Session", 1)[1][0]
        speaker_id = 'id_' + gender + '_' + gender_id
        speaker_ids_list.append(speaker_id)
    return speaker_ids_list

def get_train_list(train_data_list, txt_file_name):
    speaker_id_train = get_speaker_id(train_data_list)
    df_train = pd.DataFrame(list(zip(speaker_id_train, train_data_list)))
    df_train.to_csv(txt_file_name, sep='\t', index=False)

def get_test_list_random(test_data, txt_file_name):
    label = []
    file_1 = []
    file_2 = []
    for file_name in test_data:
        for other_file_name in test_data:
            if file_name != other_file_name:
                file_1.append(file_name)
                file_2.append(other_file_name)
    speaker_id_file_1 = get_speaker_id(file_1)
    speaker_id_file_2 = get_speaker_id(file_2)

    for index in range(len(speaker_id_file_1)):
            if speaker_id_file_1[index] == speaker_id_file_2[index]:
                label.append('1')
            else:
                label.append('0')
    
    df_test = pd.DataFrame(list(zip(label, file_1, file_2)))
    df_test.to_csv(txt_file_name, sep='\t', index=False)

def get_test_list(test_data, expected_amount, txt_file_name):
    label = []
    file_1_list = list(map(lambda x: random.choice(test_data), range(expected_amount)))
    file_2_list = list(map(lambda x: random.choice(test_data), range(expected_amount)))

    for index in range(len(file_1_list) - 1, 0, -1):
        if file_1_list[index] == file_2_list[index]:
            file_1_list.remove(file_1_list[index])
            file_2_list.remove(file_2_list[index])

    speaker_id_file_1 = get_speaker_id(file_1_list)
    speaker_id_file_2 = get_speaker_id(file_2_list)

    for index in range(len(speaker_id_file_1)):
        if speaker_id_file_1[index] == speaker_id_file_2[index]:
            label.append('1')
        else:
            label.append('0')
    
    df_test = pd.DataFrame(list(zip(label, file_1_list, file_2_list)))
    df_test.to_csv(txt_file_name, sep='\t', index=False)

data_path = './data/IEMOCAP_dataset'
# audio_files_list = get_audio_files_list(data_path)
# train_data = get_audio_files_list(data_path)
test_data = get_audio_files_list(data_path)
get_test_list(test_data, 1500, 'iemocap_test_data.txt')

# for file_name_1 in file_1_list:
#     for file_name_2 in file_2_list:
#         if file_name_1 == file_name_2:
#             file_1_list.remove(file_name_1)
#             file_2_list.remove(file_name_2)


# print(audio_files_list)

# random.shuffle(audio_files_list)
# train_data = audio_files_list[:9822]
# test_data = audio_files_list[9822:]
# train_data = audio_files_list[:int((len(audio_files_list)+1)*.90)] #Remaining 90% to training set
# test_data = audio_files_list[int((len(audio_files_list)+1)*.90):] #Splits 10% data to test set
# print(len(audio_files_list))
# print(len(train_data))
# print(len(test_data))

# get_train_list(train_data, 'iemocap_train_list.txt')
# get_test_list(test_data, 'iemocap_test_list.txt')
