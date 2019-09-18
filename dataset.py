import torch
import torchvision.transforms as transforms
import os
import random
import numpy as np
from PIL import Image

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, train=True):
        #指定する場合は前処理クラスを受け取る
        self.transform = transform
        #画像とラベルの一覧を保持するリスト
        self.images = []
        self.images2 = []
        self.sounds = []
        self.labels = []
  
        #画像を読み込むファイルパスを取得
        if train == True:
            data_path = os.path.join(root, 'train')
        else:
            data_path = os.path.join(root, 'val')

        #ペアデータのディレクトリ一覧を取得
        pair_data = os.listdir(data_path)

        #1個のリストにする
        for id_dir in pair_data:
            r = random.randint(0,1)
            id_list = np.array([])
            data_list = os.listdir(os.path.join(data_path, id_dir))

            #ディレクトリに入っているidを2つ取得
            for data in data_list:
                data_id = data.split('.')[0]
                if data_id not in id_list:
                    id_list = np.append(id_list, data_id)

            id1 = id_list[0]
            id2 = id_list[1]

            #rの方を正解データとして出力する
            self.images.append(os.path.join(data_path, id_dir, id1+'.jpg'))
            self.images2.append(os.path.join(data_path, id_dir, id2+'.jpg'))
            if r==0:
                self.sounds.append(os.path.join(data_path, id_dir, id1+'.npy'))
                self.labels.append(0)
            else:
                self.sounds.append(os.path.join(data_path, id_dir, id2+'.npy'))
                self.labels.append(1)

    def __getitem__(self, index):
        #インデックスをもとに画像のファイルパスとラベルを取得
        image = self.images[index]
        image2 = self.images2[index]
        sound = self.sounds[index]
        label = self.labels[index]
        #画像ファイルパスから画像を読み込む
        with open(image, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        with open(image2, 'rb') as ff:
            image2 = Image.open(ff)
            image2 = image2.convert('RGB')    
        with open(sound, 'rb') as fs:
            sound = np.load(sound)
            sound = np.array([sound])
        #前処理がある場合は前処理を入れる
        if self.transform is not None:
            image = self.transform(image)
            image2 = self.transform(image2)
        #画像とラベルのペアを返却
        return image, image2, sound, label

    def __len__(self):
        #データ数を指定
        return len(self.images)