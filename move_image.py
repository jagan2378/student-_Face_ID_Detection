import os
import shutil

# Create the static/images directory if it doesn't exist
os.makedirs('static/images', exist_ok=True)

# Source and destination paths
source_path = 'D:/person-detection-project/Digital-Employee-ID-2.jpg'
destination_path = 'static/images/Digital-Employee-ID-2.jpg'

# Copy the image file
if os.path.exists(source_path):
    shutil.copy2(source_path, destination_path)
    print(f"Image copied successfully to {destination_path}")
else:
    print(f"Source image not found at {source_path}") 