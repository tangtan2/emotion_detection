import os
import json
import cv2

if __name__ == '__main__':

    # folder paths
    FER = '/Users/tanyatang/Documents/Code/python/emotion_detection/data/fer'
    CK = '/Users/tanyatang/Documents/Code/python/emotion_detection/data/ck'
    VISGRAF = '/Users/tanyatang/Documents/Code/python/emotion_detection/data/visgraf'
    TARGET = '/Users/tanyatang/Documents/Code/python/emotion_detection/data/images'

    # arrays to hold all images and labels
    images = []
    image_ids = []
    image_props = {}
    labels_to_emotion = {0: 'anger',
                         1: 'disgust',
                         2: 'fear',
                         3: 'happy',
                         4: 'sadness',
                         5: 'surprise',
                         6: 'neutral'}

    # prepare FER images
    fer_csv = FER + '/fer.csv'
    header = True
    for line in open(fer_csv):
        if header:
            header = False
            continue
        split = line.split(',')
        image = [int(pixel) for pixel in split[1].split()]
        images.append(image)
        image_id = 'fer_' + split[0] + '_' + str(len(images)) + '.png'
        image_ids.append(image_id)
        image_props[image_id]['label'] = split[0]
        image_props[image_id]['num'] = str(len(images) - 1)
        image_props[image_id]['emotion'] = labels_to_emotion[int(split[0])]
        image_props[image_id]['database'] = 'fer'

    # prepare CK images
    ckit = 1
    folders = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    for i, folder in enumerate(folders):
        folder_path = CK + '/' + folder
        for file in os.listdir(folder_path):
            image = cv2.imread(folder_path + '/' + file)
            images.append(image)
            image_id = 'ck_' + str(i) + '_' + str(len(images)) + '.png'
            image_ids.append(image_id)
            image_props[image_id]['label'] = str(i)
            image_props[image_id]['num'] = str(len(images) - 1)
            image_props[image_id]['emotion'] = labels_to_emotion[i]
            image_props[image_id]['database'] = 'ck'

    # prepare VISGRAF images
    emotions = ['04', '05', '06', '01', '02', '03', '00']
    subjects = ['s001', 's002', 's003', 's004', 's005', 's006', 's007', 's008', 's009', 's010',
                's011', 's012', 's012', 's013', 's014', 's015', 's016', 's017', 's018', 's019',
                's021', 's023', 's024', 's025', 's026', 's027', 's028', 's029', 's030', 's031',
                's032', 's033', 's034', 's035', 's036', 's037', 's038']
    for subject in subjects:
        for i, emotion in enumerate(emotions):
            file_path = VISGRAF + '/' + subject + '/tif/' + subject + '-' + emotion + '_img.tif'
            image = cv2.imread(file_path)
            images.append(image)
            image_id = 'visgraf' + str(i) + '_' + str(len(images)) + '.png'
            image_ids.append(image_id)
            image_props[image_id]['label'] = str(i)
            image_props[image_id]['num'] = str(len(images) - 1)
            image_props[image_id]['emotion'] = labels_to_emotion[i]
            image_props[image_id]['database'] = 'visgraf'

    # save images to one folder and save image properties dictionary as json
    os.mkdir(TARGET)
    for i, image in enumerate(images):
        cv2.imwrite(TARGET + '/' + image_ids[i], image)
    with open(TARGET + '/image_props.json', 'w') as f:
        json.dump(image_props, f, indent=4)
