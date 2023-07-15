from PIL import Image


def reduce_image_filesize(input_path, output_path, quality=85):
    # Open the image using Pillow with reduced quality
    image = Image.open(input_path)
    image.save(output_path, optimize=True, quality=quality)

    # Print the original and reduced file sizes
    original_size = image.size
    reduced_image = Image.open(output_path)
    reduced_size = reduced_image.size
    print(f"Original Size: {original_size} | Reduced Size: {reduced_size}")

    # Close the images
    image.close()
    reduced_image.close()


reduce_image_filesize("/Users/paul/Downloads/IMG_2EFE2ADFE1DE-1.jpeg", "temp.jpeg", 50)
