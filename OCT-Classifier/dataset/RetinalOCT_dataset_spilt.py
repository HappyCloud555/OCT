import random
import os
import torch
import re
import shutil
from PIL import Image

seed = 12  # top -> 5 6
random.seed(seed)
randI = random.sample(range(1, 16), 15)
randI = [str(i) for i in randI]


def save_image(namef, nameS, c, image):
    # Check directory exists or not
    directory = os.path.dirname('./dataset/RetinalOCT_Semi_87/' + namef + '/' + nameS + '/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    outfile = directory + '/' + nameS + str(c) + '.jpg'
    image.thumbnail(image.size)
    image.save(outfile, "JPEG", quality=100)


data_dir = './dataset/RetinalOCT/train'
torch.manual_seed(seed)

if not os.path.exists(data_dir):
    print("RetinalOCT dataset not exists")
    exit()

patternName = re.compile(r'(?<=train/)[a-zA-Z]+')

# Define number of training samples for each class
train_ratio = 0.010
unlabel_ratio = 1 - train_ratio

# remove the folder before regenerating the result
if os.path.exists('./dataset/RetinalOCT_Semi_87/'):
    shutil.rmtree('./dataset/RetinalOCT_Semi_87/')

train_c = 0
train_t = 0
train_v = 0
unlabel_c = 0
total_c = 0

for i, j, k in os.walk(data_dir):
    file_counts = len(k)
    if file_counts > 1:
        i = i.replace('\\', '/')
        cla = patternName.findall(i)[0]  # find class: AMD,DME,CNV,DRUSEN or NORMAL
        train_count = int(train_ratio * file_counts)
        unlabel_count = int(file_counts - train_count)
        temp_list = list(k)

        # File Partition
        random.shuffle(temp_list)
        file_train_list = temp_list[0:train_count]
        file_unlabel_list = temp_list[train_count:]

        for fileN in k:
            f = i + '/' + fileN
            print(f)
            image = Image.open(f)
            # if cla not in ['CNV', 'DME', 'DRUSEN', 'NORMAL', 'AMD']:
            #     print('Error!')
            if fileN in file_train_list:
                print('to train')
                train_c += 1
                save_image('train', cla, total_c, image)
            elif fileN in file_unlabel_list:
                print('to unlabel')
                unlabel_c += 1
                save_image('unlabel', cla, total_c, image)
            else:
                print("Error")
            total_c += 1

type_d = ['test', 'val']


def copy_file(data_type):
    global train_t
    global train_v
    global total_c
    data_dir = './dataset/RetinalOCT/' + data_type
    patternName = re.compile(r'(?<=' + data_type + '/)[a-zA-Z]+')
    for i, j, k in os.walk(data_dir):
        file_counts = len(k)
        if file_counts > 1:
            i = i.replace('\\', '/')
            cla = patternName.findall(i)[0]  # find class: AMD,DME or Normal
            for fileN in k:
                f = i + '/' + fileN
                print(f)
                image = Image.open(f)
                print('to ' + data_type)
                if data_type == 'test':
                    train_t += 1
                elif data_type == 'val':
                    train_v += 1
                save_image(data_type, cla, total_c, image)
                total_c += 1


# handle test and val
for i in type_d:
    copy_file(i)

print("total_count:", total_c)
print("train_count:", train_c)
print("test_count:", train_t)
print("val_count:", train_v)
print("unlabel_count:", unlabel_c)
