import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO

# 加载示例图片（你、可以替换成本地文件路径）
image_path = r"C:\Users\Eddie\Desktop\image\TrainData\test\0f5380d9-a5b2-4fe4-ba60-629fd08a4f8e.jpg"
original_image = Image.open(image_path).convert("RGB")
# 定义数据增强方式
transformations = {
    "Original": transforms.Compose([]),
    "Horizontal Flip": transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0)
    ]),
    "Rotation": transforms.Compose([
        transforms.RandomRotation(degrees=15)
    ]),
    "Random Crop & Resize": transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0))
    ]),
    "Color Jitter": transforms.Compose([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    ]),
    "Normalization (shown as image)": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.ToPILImage()
    ])
}

# 可视化结果
plt.figure(figsize=(15, 10))
for i, (name, transform) in enumerate(transformations.items()):
    transformed_image = transform(original_image)
    plt.subplot(2, 3, i + 1)
    plt.imshow(transformed_image)
    plt.title(name)
    plt.axis("off")

plt.tight_layout()
plt.show()
