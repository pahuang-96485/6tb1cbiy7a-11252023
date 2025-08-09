import os 
import glob 
import random 
import shutil 
import numpy as np
from PIL import Image

'''统计所有图片每个通道的均值和标准差'''

if __name__=='__main__':
    dataset_train_dir="../dataset/train"
    train_files = glob.glob(os.path.join(dataset_train_dir, '*', '*jpg'))

    print(f' Total {len(train_files)} files for training')

    result = []

    for file in train_files:
        img = Image.open.covert('RGB')
        img = np.array(img).astype(np.uint8)
        img = img/255.
        result.append(img)

    print(np.shape(result)) #[BS,H,W,C]
    mean = np.mean(result, azis = (0,1,2))
    std = np.std(result, azis = (0,1,2))

    print(mean)
    print(std)