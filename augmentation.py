import cv2
import numpy as np
import os
import random
from keras.preprocessing.image import load_img, img_to_array, save_img

def dynamic_simulate_shadows(image):
    """Add dynamic shadows to simulate obscured lane lines."""
    height, width, _ = image.shape
    num_shadows = random.randint(1, 3)
    for _ in range(num_shadows):
        top_x = random.randint(0, width // 2)
        top_y = random.randint(0, height // 2)
        bottom_x = random.randint(width // 2, width)
        bottom_y = random.randint(height // 2, height)
        overlay = image.copy()
        shadow_color = (0, 0, 0)
        polygon = np.array([[top_x, top_y], [bottom_x, top_y], [bottom_x, bottom_y], [top_x, bottom_y]])
        cv2.fillPoly(overlay, [polygon], shadow_color)
        alpha = random.uniform(0.4, 0.7)  # Adjusted transparency range
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image

def dynamic_simulate_occlusion(image):
    """Add dynamic occlusions to simulate blockages from objects."""
    height, width, _ = image.shape
    num_occlusions = random.randint(1, 3)
    for _ in range(num_occlusions):
        top_x = random.randint(250, width - 250)
        top_y = random.randint(200, height - 30)
        block_width = random.randint(100, 150)  # Increased size
        block_height = random.randint(100, 150)  # Increased size
        color = random.choice([(117, 0, 0), (0,76,153)])  # Red  or BLUE
        cv2.rectangle(image, (top_x, top_y), (top_x + block_width, top_y + block_height), color, -1)
    return image

def augment_dataset(txt_file):
    """Apply dynamic data augmentation to 1/10 of the images listed in a text file."""
    with open(txt_file, 'r') as f:
        filenames = [line.strip() for line in f.readlines()]

    random.shuffle(filenames)

    # Select 1/10 of the images for augmentation
    num_to_augment = len(filenames) // 8
    augmented_files = set(filenames[:num_to_augment])

    total_files = len(filenames)
    for index, filepath in enumerate(filenames):
        if filepath in augmented_files:
            augmented_path = '../../CULane/' + filepath
            print(f"Processing augmented image ({index + 1}/{total_files/8}): {augmented_path}")  # Print progress
            image = img_to_array(load_img(augmented_path))

            # Randomly select one augmentation strategy
            augmentation_strategy = random.choice([
                dynamic_simulate_shadows,
                dynamic_simulate_occlusion
            ])
            image = augmentation_strategy(image)

            save_img(augmented_path, image)

# Example usage:
txt_file = '../../CULane/list/train.txt'  # Input text file listing image paths
augment_dataset(txt_file)
print("完成")
