from pathlib import Path

from PIL import Image

images_path = Path("images")

image_paths = images_path.glob("*.jpg")

# images = [Image.open(image_path) for image_path in image_paths]

for image_path in image_paths:
    # print width and height of each image

    with Image.open(image_path) as img:
        width, height = img.size
        print(f"Image: {image_path.name}, Width: {width}, Height: {height}")
