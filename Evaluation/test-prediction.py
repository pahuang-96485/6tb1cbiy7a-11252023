import os
import clip
import torch
from PIL import Image
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
###########Zero Prediction############


device = "cuda" if torch.cuda.is_available() else "cpu"

def get_args_parser():

    parser = argparse.ArgumentParser('White box attack task', add_help=False)

    parser.add_argument('--input_size', default = 128, type = int, help = 'image imput size')

    parser.add_argument('--image_dir', default = './dataset/train', type = str, help = 'path to load images')

    parser.add_argument('--text_dir', default = './dataset', type = str, help = 'path to text token')

    #parser.add_argument('--test_image_dir', default = './dataset/test', type = str, help = 'images to test')

    parser.add_argument('--save_dir', default = '../outputd_dir', type = str, help = 'path to save, empty for no saving')

    parser.add_argument('--model_name', default = 'ViT-B/32', type = str, help = 'clip in RN50, RN101, RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px')

    parser.add_argument('--feature_dict_file', default = 'corpis_feature_dict.npy',  help = 'file name to save image presentation')

    #parser.add_argument('--mode', default = 'extract', type = int, help = 'extract ot predict')

    return parser



def extract_feature_by_CLIP(model, preprocess, file):
    image = preprocess(Image.open(file)).unsqueeze(0).to(device)
    with torch.set_grad_enabled(False):
        image_features  = model.encode_image(image)
        image_features  = image_features.squeeze().numpy()

    return image_features 


def extract_features(args, model, image_path='', preprocess=None):

    image_features = {}
    for image_file in glob.glob(os.path.join(image_path,'*.JPEG')):
        image_features [image_file] = extract_feature_by_CLIP(model, preprocess, image_file)

    os.makedirs(f"{args.save_dir}/{args.model_name}", exist_ok=True)
    np.save(f"{args.save_dir}/{args.model_name}/{args.feature_dict_file}", image_features) #########

    return image_features 



def main():

    # Load the model
    
    args = get_args_parser()
    args = args.parse_args()

    model =None
    preprocess = None

    model_name = args.model_name

    print ("model_name ", model_name)
    print(f'use pretrained model {args.model_name} to extract features')
    model, preprocess = clip.load(model_name, device=device)
    #RN50, RN101, RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px
    
    
   
    
    # prepare image features
    image_encoding= extract_features(args, model, image_path=args.image_dir, preprocess=preprocess)
    
    # prepare text features
    with open(args.text_dir, "r") as text_file:
        texts = [line.strip() for line in text_file.readlines()]
    text_inputs = clip.tokenize(texts).to(device)

    text_features = model.encode_text(text_inputs)
    
    print("image_features size ", len(image_encoding))
    print ("image_encoding:", list(image_encoding)[0])
    image_features = image_encoding[list(image_encoding)[0]]

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, top_5_indices = similarity[0].topk(5)

    plt.bar([f"Text {index}" for index in top_5_indices], similarity[top_5_indices])
    plt.xlabel("Texts")
    plt.ylabel("Similarity Score")
    plt.title("Top 5 Similarity Scores for the Given Image")
    plt.show()


if __name__ == '__main__':
    main()