import os
import numpy as np
from PIL import Image
import argparse


def image_to_patches_and_save(image_path, patch_size=64, output_folder='patches'):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载图像
    image = Image.open(image_path)
    image = np.array(image)

    # 获取图像尺寸
    img_height, img_width = image.shape[:2]

    # 计算每个维度的块数量
    patch_height = patch_size
    patch_width = patch_size
    patches_per_row = img_width // patch_width
    patches_per_col = img_height // patch_height

    patch_number = 0  # 用于命名块文件
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # 遍历图像并提取块
    for col in range(patches_per_row):
        for row in range(patches_per_col):
            # 定义块边界
            start_row = row * patch_height
            end_row = start_row + patch_height
            start_col = col * patch_width
            end_col = start_col + patch_width

            # 提取块
            patch = image[start_row:end_row, start_col:end_col]

            # 转换为PIL图像以保存
            patch_image = Image.fromarray(patch)

            # 保存块
            patch_filename = os.path.join(output_folder, f'{base_filename}_{patch_number}.png')
            patch_image.save(patch_filename)
            patch_number += 1


def process_images_in_folder(input_folder, output_folder, patch_size=64):
    # 检查并创建输出文件夹
    os.makedirs(output_folder,exist_ok=True)

    # 遍历输入文件夹中的所有图片
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image_to_patches_and_save(image_path, patch_size, output_folder)


def main():
    # 创建命令行解析器
    parser = argparse.ArgumentParser(description="Split images into patches and save them.")
    parser.add_argument('--input_folder', type=str, help="Path to the input folder containing images.")
    parser.add_argument('--output_folder', type=str, help="Path to the output folder to save patches.")
    parser.add_argument('--patch_size', type=int,default=32, help="Patch size as two integers (height, width). Default is 64x64.")
    # 解析命令行参数
    args = parser.parse_args()
    # 调用处理函数
    process_images_in_folder(args.input_folder, args.output_folder, args.patch_size)

    print(f"Patches are saved in folder: {args.output_folder}")

if __name__ == '__main__':
    main()

