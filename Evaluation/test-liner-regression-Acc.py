import os
import clip
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from tqdm import tqdm
from torchvision import datasets
from sklearn.metrics import accuracy_score

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


# Function to extract features from images
def get_image_features(dataset):
    image_features = []
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100, shuffle=False)):
            print("image shape: ", images.shape)############torch.Size([20, 3, 224, 224])
            features = model.encode_image(images.to(device))
            print("features shape: ", features.shape)###### torch.Size([20, 512])
            image_features.append(features)
        print("image_features size: ", len(image_features))#####image_features size:  1
    return torch.cat(image_features).cpu().numpy()

# Function to extract features from text
def get_text_features(texts):
    text_features = []
    with torch.no_grad():
        texts = clip.tokenize(texts).to(device)#####3texts are now tensors
        print("texts shape:", texts.shape)######torch.Size([50, 77])
        features = model.encode_text(texts)
        print("features shape: ", features.shape)#####torch.Size([50, 512])
        text_features.append(features)
        print("text_features size: ", len(text_features))#####text_features size:  1
    return torch.cat(text_features).cpu().numpy()

# Function to perform zero-shot prediction and combine top-1 predictions
def zero_shot_prediction(image_features, text_features):
    # Normalize image features
    image_features /= np.linalg.norm(image_features, axis=-1, keepdims=True)

    # Normalize text features
    text_features /= np.linalg.norm(text_features, axis=-1, keepdims=True)

    # Calculate similarity
    similarity = torch.from_numpy((100.0 * image_features @ text_features.T)).softmax(axis=-1)

    # Get the index of the top-1 result for the single image
    top1_predictions = np.argmax(similarity, axis=1)

    #print("top1_predictions: ", top1_predictions)

    values, indices = similarity[19].topk(1)

    # Prepare text inputs
    with open('../dataset/report/text-tokens.txt', "r") as text_file:
        texts = [line.strip() for line in text_file.readlines()]

    # Print the top-5 results
    for value, index in zip(values, indices):
        print(f"{texts[index]:>16}: {100 * value.item():.2f}%")   

    return top1_predictions


# Load the dataset
root = 'test_data'
test_root = '../dataset/report/adv'
train_dataset = datasets.ImageFolder(root,transform=preprocess)
test_dataset = datasets.ImageFolder(test_root,transform=preprocess)


# Extract image features for dataset 
train_image_features = get_image_features(train_dataset)
print("train_image_features shape", train_image_features.shape)
test_image_features = get_image_features(test_dataset)

# Extract text features for training set
#with open('extracted_labels_1K.txt', "r") as f:
with open('../dataset/report/text-tokens.txt', "r") as f:
    labels = [line.strip() for line in f.readlines()]
text_features = get_text_features(labels)
print("text_features size = get_text_features(labels): shape", text_features.shape)


# Perform zero-shot prediction on the training set
train_predictions = zero_shot_prediction(train_image_features, text_features)
print("train_predictions size: ", len(train_predictions)) 
print("train_predictions: ", train_predictions) 
# Perform zero-shot prediction on the test set
test_predictions = zero_shot_prediction(test_image_features, text_features)
print("test_predictions size: ", len(test_predictions)) 
print("test_predictions: ", test_predictions)

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_image_features, train_predictions)

# Evaluate the accuracy
accuracy = accuracy_score(train_predictions, test_predictions).astype(float) * 100.
print(f"Accuracy: {accuracy:.2f}%")

