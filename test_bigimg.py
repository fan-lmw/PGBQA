import cv2
import os
import numpy as np
import torch
from torchvision import transforms
import shutil
import torch.nn as nn
import argparse
from skimage.metrics import normalized_root_mse

# 输入：图片路径(path+filename)，裁剪获得小图片的列数、行数（也即宽、高）
# 输出：无
def compute_gram_matrix(image):
    """计算给定图像的 Gram 矩阵."""
    # 确保图像的形状是 (H, W, C)，其中 C 是通道数
    height, width, channels = image.shape

    # 将图像展平为 (H*W, C) 的矩阵
    features = image.reshape(height * width, channels)#/255

    # 计算 Gram 矩阵 (C, C)
    gram_matrix = np.dot(features.T, features)

    return gram_matrix


def frobenius_norm(matrix1, matrix2):
    return np.linalg.norm(matrix1 - matrix2, 'fro')

def crop_one_picture(fake_path, slices_path,filename,cutshape_w, cutshape_h):
    # 灰度图读入
    img = cv2.imread(fake_path)
    h = img.shape[0]  # 高度
    w = img.shape[1]  # 宽度

    for i in range(int(w / cutshape_w)):
        for j in range(int(h / cutshape_h)):
            cv2.imwrite(
                os.path.join(slices_path,os.path.splitext(filename)[0] + '_' + str(j) + '_' + str(i) + os.path.splitext(filename)[
                    1]),
                img[j * cutshape_h:(j + 1) * cutshape_h, i * cutshape_w:(i + 1) * cutshape_w])


# 输入：图片路径(path+filename)，裁剪所的图片的列的数量、行的数量
# 输出：无
def merge_picture(recon_slices_path, save_path, num_of_cols, num_of_rows, rows, cols):
    recon_imgs = os.listdir(recon_slices_path)

    # 创建一个全零的 NumPy 数组
    dst = np.zeros((rows * int(num_of_rows), cols * int(num_of_cols),3), dtype=np.uint8)
    for recon_img in recon_imgs:
        img = cv2.imread(os.path.join(recon_slices_path,recon_img), cv2.IMREAD_COLOR)
        cols_th = int(recon_img.split("_")[-1].split('.')[0])
        rows_th = int(recon_img.split("_")[-2])
        roi = img[0:rows, 0:cols]

        dst[rows_th * rows:(rows_th + 1) * rows, cols_th * cols:(cols_th + 1) * cols] = roi

    cv2.imwrite(save_path, dst)


def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def main(args):
    # 根据 patchsize 选择不同的模型
    if args.patchsize == 32:
        from models.model_new_32 import CAE  # 如果 patchsize 为 32，加载 model_new_32 模型
    elif args.patchsize == 64:
        from models.model_new_64 import CAE  # 如果 patchsize 为 64，加载 model_new_64 模型
    else:
        raise ValueError("Unsupported patchsize. Please use 32 or 64.")  # 不支持的 patchsize

    weights_file = args.weights_file
    slices_path = args.slices_path  # 临时保存分块后切片的文件夹
    recon_slices_path = args.recon_slices_path  # 临时保存自编码器重构切片的文件夹
    os.makedirs(slices_path, exist_ok=True)
    os.makedirs(recon_slices_path, exist_ok=True)

    cutshape_w = args.patchsize  # 切片宽度
    cutshape_h = args.patchsize  # 切片高度
    dir_path = args.input_dir  # 输入测试图像所在文件夹
    save_path = os.path.join(args.output_dir,'reconimage')  # 保存重构图像的文件夹

    os.makedirs(args.output_dir,exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    # 判断是否有可用的 GPU
    device = torch.device("cuda" if (torch.cuda.is_available() and int(args.device_id) >=0) else "cpu")
    model = CAE(dropout_prob=0.2)
    # 加载模型权重，指定 map_location
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.eval()
    # 将模型移动到适当的设备
    model.to(device)  # 使用 `to` 方法将模型移动到 GPU 或 CPU
    print(device)

    recon_loss = []
    # 测试图像文件名
    files = os.listdir(dir_path)
    files.sort()

    for filename in files:
        # 将文件夹进行创建
        fake_path = os.path.join(dir_path, filename)
        IMG = cv2.imread(fake_path)
        # print(fake_path)
        # 图像分块，分块后所有切片已保存到slices_path
        crop_one_picture(fake_path, slices_path, filename, cutshape_w, cutshape_h)
        transform = transforms.ToTensor()
        img_files = os.listdir(slices_path)

        # nrmse1表示内容损失采用的是均方根误差， nrmse2表示的是采用gram矩阵的风格误差
        # nrmse3表示的内容损失加风格损失之和
        # nrmse4表示的是将内容损失和风格损失之积
        nrmse1 = 0.0
        nrmse2 = 0.0
        nrmse3 = 0.0
        for img_file in img_files:
            img_path = os.path.join(slices_path, img_file)
            img1 = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = transform(img1)
            # Add the batch dimension in the first axis

            img = img.unsqueeze(0)
            # 将 imgr 移动到指定设备
            img = img.to(device)
            _, reconstructed_img = model(img)

            reconstructed_img = reconstructed_img.squeeze(0).data.cpu().numpy().transpose((1, 2, 0))  # [0][0]
            img2 = reconstructed_img
            # 确保 img2 在 0 到 255 范围内
            img2 = (img2 * 255).astype(np.uint8)
            nrmse1 = normalized_root_mse(img1, img2, normalization='euclidean') + nrmse1

            img_1 = compute_gram_matrix(img1)
            img_2 = compute_gram_matrix(img2)

            nrmse2 = frobenius_norm(img_1, img_2) + nrmse2
            nrmse3 = normalized_root_mse(img1, img2, normalization='euclidean') * frobenius_norm(img_1, img_2) + nrmse3

            cv2.imwrite(os.path.join(recon_slices_path, img_file), reconstructed_img * 255)

        # print(len(img_files))
        nrmse1 = nrmse1
        nrmse2 = nrmse2
        nrmse3 = nrmse3
        nrmse4 = nrmse1 * nrmse2

        # 图像合并，输出重构大图
        num_of_cols = IMG.shape[0] / cutshape_w  # 列数
        num_of_rows = IMG.shape[1] / cutshape_h  # 行数
        merge_picture(recon_slices_path, os.path.join(save_path, filename), num_of_cols, num_of_rows, cutshape_h,
                      cutshape_w)

        # 删除两个临时的切片文件夹下所有切片
        del_file(slices_path)
        del_file(recon_slices_path)

        recon_loss.append({'fake_path': fake_path, 'content_Loss': nrmse1, 'style_loss': nrmse2, 'add_loss': nrmse3,
                           'mult_loss': nrmse4})

    shutil.rmtree(slices_path)
    shutil.rmtree(recon_slices_path)

    sorted_by_loss4 = sorted(recon_loss, key=lambda x: x['mult_loss'])
    # 计算所有结果的平均值
    loss_all4 = np.array([res['mult_loss'] for res in recon_loss])

    # print(dir_path)
    print(f"平均值： mult_loss : {loss_all4.mean()}")

    # 如果需要，输出排序后的损失
    output_log = os.path.join(args.output_dir, 'loss.txt')
    # 打开一个文本文件，以写入模式（'w'）打开
    with open(output_log, 'w') as f:
        for res in sorted_by_loss4:
            # 计算并格式化 mult_loss
            formatted_loss = res['mult_loss'] / 10000
            f.write(f"{res['fake_path']} - mult_loss: {formatted_loss:.3f}\n")

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Train and Evaluate the CAE model.")
    parser.add_argument('--device_id', type=str, default='0', help="CUDA device ID (e.g., '0' for GPU 0)")
    parser.add_argument('--weights_file', type=str, required=True, help="Path to the model weights file")
    parser.add_argument('--slices_path', type=str, default="./demo/tmp_slices/", help="Directory to save image slices")
    parser.add_argument('--recon_slices_path', type=str, default="./demo/tmp_recon_slices/",
                        help="Directory to save reconstructed slices")
    parser.add_argument('--patchsize', type=int, default=32, help="patchsize of each image slice")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing the input test_5.30 images")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the results")
    args = parser.parse_args()

    # 调用主函数
    main(args)


