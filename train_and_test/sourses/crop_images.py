import os
import numpy as np
from cv2 import imread
from shutil import move
from PIL import Image

def load_images_from_folder(folder):
    '''
    Loads images from a specific folder (provide the folder path),
    Returns a list of images as numpy arrays.
    '''
    images = []
    for filename in os.listdir(folder):
        img = imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    print('[INFO] load images done!')
    return images

def split_img_array(img_arr, ver_split, hor_split):
    '''
    Recieves a list of images, as numpy arrays, and take one by one 
    to vertically and horizontally split into multiple small images.
    Returns a list of small images as numpy arrays.
        - img_arr: list of images as numpy array.
        - ver_split: number of vertical divitions, uses the method numpy.vsplit
        - hor_split: number of horizontal divitions, uses the method numpy.hsplit
    '''
    first_spli_img = []
    secon_split_img = []
    for img in img_arr:
        first_spli_img += np.vsplit(img, ver_split)
        for ver_img in first_spli_img:
            secon_split_img += np.hsplit(ver_img, hor_split)
    print('[INFO] split done!')
    return secon_split_img

def save_img_array(img_arr, path_base):
    '''
    Given a list of images, as numpy arrays, and a directory path,
    Saves these images in tha path given as jpg.
    '''
    fill_zeros = len(str(len(img_arr)))
    for i,img in enumerate(img_arr):
        img_number = str(i).zfill(fill_zeros)
        Image.fromarray(img).save(path_base + f'{img_number}.jpg')
    print('[INFO] save done!')
    return 'Done!!'

def move_random_files(origin, destin, percent):
    '''
    Moves a percentage of file, randomly, from a directory (origin) to another one (destin)
        - origin: the path where the files are currently located.
        - destin: the new path to move the files.
        - percent: a float between 0 and 1, where 1 means moving all files.
    '''
    files = os.listdir(origin)
    list_len = len(files)
    k = int(list_len * percent)
    test_img = np.random.choice(files, size=k, replace=False)
    for i in test_img:
        file_path = origin+i
        move(file_path, destin)
    print('[INFO] random move done!')
    return 'Done!'


# path = 'dataset/images-for-recognition/no-traffic-signs'
# path_1 = 'dataset/Train/no-traffic-sign/'
# path_2 = 'dataset/Test_class_id/no-traffic-sign/'

# img_arr = load_images_from_folder(path)
# print('Images loaded:',len(img_arr))
# split_img = split_img_array(img_arr, 8, 10)
# print('Splir images produced',len(split_img))
# save_img_array(split_img, path_1)
# move_random_files(path_1, path_2, 0.2)