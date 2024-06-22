import cv2
import os

def save_image(img_data, img_name):
    # Save the image to the 'images' folder inside the 'backend' folder
    save_path = os.path.join('backend', 'images', img_name)
    cv2.imwrite(save_path, img_data)

    return save_path  # Return the path to the saved image
