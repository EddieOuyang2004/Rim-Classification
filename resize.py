from PIL import Image
import os

# === Settings ===
input_folder = r"C:\Users\Eddie\OneDrive - University College London\ELEC0145\image\TrainData\test"  # Change this
output_folder = r"C:\Users\Eddie\OneDrive - University College London\ELEC0145\image\comparison"  # Change this
target_size = (227, 227)  # Width x Height

# === Ensure output folder exists ===
os.makedirs(output_folder, exist_ok=True)

# === Process each image ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        try:
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            img = img.convert("RGB")
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)

            output_path = os.path.join(output_folder, filename)
            resized_img.save(output_path)

            print(f"Resized and saved: {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
