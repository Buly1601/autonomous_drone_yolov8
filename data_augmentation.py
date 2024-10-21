import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

def augment_images(input_dir, output_dir, augment_count=5):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set up data augmentation transformations
    datagen = ImageDataGenerator(
        rotation_range=40,      # Rotate images by 0-40 degrees     
        width_shift_range=0.2,  # Horizontal shift
        height_shift_range=0.2, # Vertical shift
        shear_range=0.2,        # Shearing
        zoom_range=0.2,         # Zoom in/out
        horizontal_flip=True,   # Random horizontal flip
        brightness_range=[0.8, 1.2], # Adjust brightness
        fill_mode='nearest'     # Fill missing pixels
    )

    # Loop through each image in the input directory
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        
        # Load the image and convert it to an array
        img = load_img(img_path)
        img_array = img_to_array(img)  # Convert to NumPy array
        img_array = img_array.reshape((1,) + img_array.shape)  # Add batch dimension

        # Generate augmented images
        print(f"Augmenting {img_name}...")
        i = 0
        for batch in datagen.flow(img_array, batch_size=1, 
                                  save_to_dir=output_dir, 
                                  save_prefix='aug', 
                                  save_format='jpeg'):
            i += 1
            if i >= augment_count:
                break  # Stop after generating specified number of augmentations

    print("Augmentation complete.")

# Usage example:
input_directory = '/home/buly/Desktop/drone_competition/data/raw_images'
output_directory = '/home/buly/Desktop/drone_competition/data/all_images'
augment_images(input_directory, output_directory)
