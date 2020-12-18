import shutil
import pandas as pd
import os

def reorganize_images(class_list, testfile):
    for i in class_list:
        current_dirs = testfile[testfile.ClassId == i].Path.values
        path_to_dir = 'dataset/test_class_id/' + str(i) + '/'
        
        if not os.path.exists(os.path.dirname(path_to_dir)): 
            try:
                os.makedirs(os.path.dirname(path_to_dir))
            except:
                print(f'an error has ocurred when tried to created {path_to_dir}')
        for image in current_dirs:
            shutil.copy(image, path_to_dir)

    
    return 'reorganization done!!'

print(os.getcwd())
test_csv = pd.read_csv('dataset/Train.csv')
testfile = test_csv[['ClassId', 'Path']].sort_values(by='ClassId')
testfile.Path = testfile.Path.apply(lambda x: 'dataset/' + x)

# print(testfile.head())
class_list = list(range(0,43))
reorganize_images(class_list, testfile)