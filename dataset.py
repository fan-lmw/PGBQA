from torch.utils.data import Dataset
from torchvision import transforms
import os
import cv2


class MyDataset(Dataset):

    def __init__(self, path, train=True, transform=None):
        self.path = path
        self.transform = transform
        self.train = train
        self.imgs_list = self.load_data()

    def __getitem__(self, index):
        if self.train:
            img_path = self.imgs_list[index]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if self.transform is not None:
                img = self.transform(img)
        else:
            img_path = self.imgs_list[index]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if self.transform is not None:
                img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs_list)

    def load_data(self):
        img_path_list = []
        img_files = os.listdir(self.path)
        for img_file in img_files:
            img_path = os.path.join(self.path, img_file)
            img_path_list.append(img_path)
        return img_path_list


if __name__ == "__main__":
    myDataset = MyDataset(path="./data", transform=transforms.ToTensor())
    print("myDataset 的类型：", type(myDataset))
    print("myDataset 的长度：", len(myDataset))
    print("myDataset[0]：", myDataset[0])
