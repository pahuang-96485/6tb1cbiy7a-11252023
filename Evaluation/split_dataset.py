
import os 
import glob 
import random 
from PIL import Image

'''Convert all images to RGB and resize them to a uniform size'''

if __name__=='__main__':
    test_split_ratio = 0.05
    desired_size = 128
    raw_path="../dataset/raw"
    output_train_dir="../dataset/train"
    output_test_dir="../dataset/test"

    dirs = glob.glob(os.path.join(raw_path,'*'))
    dirs = [d for d in dirs if os.path.isdir(d)]
    print (f'Total {len(dir)} classes: {dirs}')

    for path in dirs:
        path=path.split('/')[-1]

        os.makedirs(f'{output_train_dir}/{path}',exist_ok=True)
        os.makedirs(f'{output_test_dir}/{path}',exist_ok=True)

        files = glob.glob(os.path.join(raw_path, path, '*.jpg'))
        files += glob.glob(os.path.join(raw_path, path, '*.JPG'))
        files = glob.glob(os.path.join(raw_path, path, '*.png'))

        random.shuffle(files)

        boundary = int(len(files)*test_split_ratio)

        for i, file in enumerate(files):
            img = Image.open(file).covert('RBG')
            old_size = img.size #in (width, height)

            ratio = float(desired_size)/max(old_size)

            new_size = tuple([int(x*ratio) for x in old_size])

            im = img.resize(new_size, Image.ANTIALIAS)

            new_im = Image.new('RGB', (desired_size,desired_size) )

            new_im.paste(im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))

            assert new_im.mode =='RGB'

            if i <= boundary:
                new_im.save(os.path.join(f'{output_test_dir}/{path}', file.split('/')[-1].split('.')[0]+'jpg'))
            else:
                new_im.save(os.path.join(f'{output_train_dir}/{path}', file.split('/')[-1].split('.')[0]+'jpg'))


    test_files= glob.glob (os.path.join(output_test_dir, '*', '*jpg'))
    train_files= glob.glob (os.path.join(output_train_dir, '*', '*jpg'))

    print(f'Total {len(train_files)} files for training ')
    print(f'Total {len(test_files)} files for testing ')


