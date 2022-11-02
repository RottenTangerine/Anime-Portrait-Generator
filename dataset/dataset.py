from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
import os


class ImageData(Dataset):
    def __init__(self, dataset_name):
        self.path = os.path.join('./data', dataset_name)
        self.image_list = os.listdir(self.path)

    def read_data(self, index):
        img_a = Image.open(os.path.join(self.path, self.image_list[index]))
        img_a = img_a.convert('RGB')
        img_a = T.Compose([
            T.ToTensor(),
            T.Resize((160, 160)),
            T.CenterCrop(128),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])(img_a)
        return img_a

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index: int):
        try:
            img = self.read_data(index)
        except Exception as e:
            # open random image
            print('Cannot open the image, use the'
                  ' first image as the default image')
            img = self.read_data(0)
        return img


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = ImageData('anime')
    DataLoader = DataLoader(dataset, batch_size=6, shuffle=True)

    img = next(iter(DataLoader))[0]
    img = img.permute(1, 2, 0)
    plt.imshow(img)
    plt.show()


