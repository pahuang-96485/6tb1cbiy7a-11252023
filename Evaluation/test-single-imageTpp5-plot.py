import os
import clip
import torch
from PIL import Image
import matplotlib.pyplot as plt


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


# Prepare text inputs
with open('.../complex_labels_2.txt', "r") as text_file:
    texts = [line.strip() for line in text_file.readlines()]
    
text_inputs = clip.tokenize(texts).to(device)

image_path = '.../fgsm_eps_16/adv/00029.JPEG'
original_image = Image.open(image_path)
image = preprocess(original_image).unsqueeze(0).to(device)
with torch.set_grad_enabled(False):
    image_features  = model.encode_image(image)
    text_features = model.encode_text(text_inputs)


# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

#print("similarity", similarity)
values, top_5_indices = similarity[0].topk(5)

# Print the top-5 results
for value, index in zip(values, top_5_indices):
    print(f"{texts[index]:>16}: {100 * value.item():.2f}%")


# Visualize the results with a horizontal bar plot
plt.figure(figsize=(10, 3))

# Plot the horizontal bar chart for top-5 results with different light soft colors
light_soft_colors = [plt.cm.Pastel2(i / len(values)) for i in range(len(values))]
bars = plt.barh(range(len(top_5_indices)), values.cpu().numpy(), color=light_soft_colors, height=0.5)

# Add text annotations inside each bar, aligned to the left
for bar, value, index in zip(bars, values, top_5_indices):
    plt.text(bar.get_x() + 0.02, bar.get_y() + bar.get_height() / 2, f'{texts[index]}: {100 * value.item():.2f}%', va='center', ha='left', color='black', fontsize=8)

plt.xlabel('Similarity (%)')
plt.title('Top-5 CLIP Predictions')

plt.show()