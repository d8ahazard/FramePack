from PIL import Image, ImageDraw
import os

# Create a directory for the favicon if it doesn't exist
os.makedirs("static/images", exist_ok=True)

# Create a new image with a blue background
img = Image.new('RGB', (32, 32), color=(67, 97, 238))
draw = ImageDraw.Draw(img)

# Draw a white "F" letter
draw.rectangle([8, 4, 24, 8], fill=(255, 255, 255))  # Top horizontal line
draw.rectangle([8, 4, 12, 16], fill=(255, 255, 255))  # Left vertical line
draw.rectangle([8, 14, 20, 18], fill=(255, 255, 255))  # Middle horizontal line
draw.rectangle([8, 18, 12, 28], fill=(255, 255, 255))  # Bottom vertical line

# Save the image as .ico
img.save('static/images/favicon.ico')

print("Favicon created successfully at static/images/favicon.ico") 