# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import os
import argparse
import glob
from tqdm import tqdm
import numpy as np
import torch
import torchvision
from PIL import Image
import timm
import clip

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args_parser():

    parser = argparse.ArgumentParser('White box attack task', add_help=False)

    parser.add_argument('--input_size', default = 128, type = int, help = 'image imput size')

    parser.add_argument('--dataset_dir', default = '../dataset/train', type = int, help = 'path to load images')

    parser.add_argument('--test_image_dir', default = '../dataset/test', type = int, help = 'images to test')

    parser.add_argument('--save_dir', default = './outputd_dir', type = int, help = 'path to save, empty for no saving')

    parser.add_argument('--model_name', default = 'resnet50', type = int, help = 'resnet50, resnet152, clip')

    parser.add_argument('--feature_dict_file', default = 'corpis_feature_dict.npy',  help = 'file name to save image presentation')

    parser.add_argument('--mode', default = 'extract', type = int, help = 'extract ot predict')

    return parser




def extract_feature_single(args, model, file):
    img_rgb = Image.open(file).convert('RGB')
    image = img_rgb.resize((args.input_size, args.input_size), image.ANTIALIAS)
    image =  torchvision.transformers.ToTensor()(image)
    trainset_mean = [] 
    trainset_std = []

    image = torchvision.transformers.Normalize(mean = trainset_mean, std =trainset_std)(image).unsqueeze(0)

    with torch.no_grade():
        features = model.forward_features(image)
        vec = model.global_pool(features)
        vec = vec.squeeze().numpy()

    img_rgb.close()

    return vec



def extract_feature_by_CLIP(model, preprocess, file):
    image = preprocess(Image.open(file)).unsqueeze(0).to(device)
    with torch.no_grade():
        vec = model.encode_image(image)
        vec = vec.squeeze().numpy()

    return vec


def extract_features(args, model, image_path='', preprocess=None):

    allVectors = {}
    for image_file in tqdm.tqdm(glob.glob(os.path.join(args.image_path, '*','*.jpg'))):
        if args.model_name =="clip":
            allVectors[image_file] = extract_feature_by_CLIP(model, preprocess, image_file)
        else:
            allVectors[image_file] = extract_feature_single(args, model, image_file)

    os.makedirs(f"{args.save_dir}/{args.model_name}", exist_ok=True)
    np.save(f"{args.save_dir}/{args.model_name}/{args.feature_dict_file}", allVectors) #########

    return allVectors


def main():

    model_name = timm.list_models(pretrained=True)

    args = get_args_parser()
    args = args.parse_args()

    model =None
    preprocess = None

    if args.model_name !='clip':
        model = timm.create_model(args.model_name, pretrained=True)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requres_grad)
        print('number of trainable params (M): %.2f' % (n_parameters / 1.e6))
        model.eval()
    else:
        model, preprocess = clip.load("ViT-B/32", device=device)

    
    if args.mode == "extract":

        print(f'use pretrained model {args.model_name} to extract features')
        allVectors = extract_features(args, model, image_path=args.dataset_dir, preprocess=preprocess)
    else:

        print(f'image from pretrained model {args.model_name} i sunder attack')
        test_image = glob.glob(os.path.join(args.test_image_dir, "*.png"))
        













