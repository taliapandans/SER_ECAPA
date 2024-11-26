from sklearn.model_selection import KFold
import csv
import random

data_file = './ses1_out_data_eval.txt'
data_list = []
# train_list_random = []
# val_list_random = []

lines = open(data_file, "r")
for line in lines:
    data_list.append(line)

# k_folds = 10
# kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
# for i, (train_index, val_index) in enumerate(kf.split(data_list)):

#     print(f"Fold {i}:")
#     # print(f"  Train: index={train_index}")
#     # print(f"  Test:  index={val_index}")

#     train_list = [data_list[i] for i in train_index]
#     val_list = [data_list[i] for i in val_index]
#     random.shuffle(train_list)
#     random.shuffle(val_list)

#     train_file = "data_train_" + str(i+1) + ".txt"
#     val_file = "data_test_" + str(i+1) + ".txt"

#     with open(train_file, 'w') as output:
#             for row in train_list:
#             # for row in train_list_random:
#                 output.write(str(row))
#     with open(val_file, 'w') as output:
#             for row in val_list:
#             # for row in val_list_random:
                # output.write(str(row))
train_list = []
for data in data_list:
    # train_list = [data_list[i] for i in train_index]
    train_list.append(data)
    random.shuffle(train_list)
    
train_file = "ses1_out_sf_data_val" + ".txt"
with open(train_file, 'w') as output:
        for row in train_list:
        # for row in train_list_random:
            output.write(str(row))
