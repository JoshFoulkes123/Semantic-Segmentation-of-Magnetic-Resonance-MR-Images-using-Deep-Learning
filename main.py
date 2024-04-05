# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.utils.data as data
import cv2
import os
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import time
from datetime import datetime


# Finds the device the data will be computed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"The current device is {device}")

# Create variables which stores the location of the training images and validation images.
train_data_path = './data/train'

val_data_path = './data/val'

test_data_path = './data/test'

torch.set_printoptions(edgeitems=100)

# It visualises the actual MR image and its mask.
def show_image_mask(img, mask, cmap='gray'):
    fig = plt.figure(figsize=(5, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap=cmap)
    plt.axis('off')


# image = cv2.imread(os.path.join(folder_data_train,'image','cmr10.png'), cv2.IMREAD_UNCHANGED) # A random image to find later the size of it.
# mask = cv2.imread(os.path.join(folder_data_train,'mask','cmr10_mask.png'), cv2.IMREAD_UNCHANGED)
# show_image_mask(image, mask, cmap='gray')
# plt.pause(1)
# cv2.imwrite(os.path.join('./','cmr1.png'), mask*85)
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# It loads all the images for the training files.
class CustomDataset(data.Dataset):
    def __init__(self, root=''):
        super(CustomDataset, self).__init__()
        self.img_files = glob(os.path.join(root, 'image', '*.png'))
        self.mask_files = []
        for img_path in self.img_files:
            basename = os.path.basename(img_path)
            self.mask_files.append(os.path.join(root, 'mask', basename[:-4] + '_mask.png'))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        label = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return len(self.img_files)


class TestDataset(data.Dataset):
    def __init__(self, root=''):
        super(TestDataset, self).__init__()
        self.img_files = glob(os.path.join(root, 'image', '*.png'))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        return torch.from_numpy(data).float()

    def __len__(self):
        return len(self.img_files)


# Since every image is black and white, they have depth 1. So the unedit image has a weight and heigh 96 and depth 1.
# So we need the output to have channel 4.

# The model is the encode of the image in the CNN power point pg78. First of all, the model will take an image with size 96x96 with depth 1
# and it will be convoluted to channel 4, it will be convoluted to a matrix of size 48x48 with depht 32 using
# max pooling, again it will be convoluted to a new matrix of size 24x24 with depht 64 using max pooling, again it will be convoluted
# to a matrix of size 12x12 with depht 128 using max pooling. Then, the last matrix will be convoluted to a matrix
# of size 24x24 with depth 64 using max unpooling, it will be convoluted to matrix of size 48x48 with depth 32 using
# max unpooling and it will be convoluted to a matrix of size 96x96 with depht 4 using max unpooling. So, first it pools
# the matrix and then unpool the matrix to give back the same size.
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class Unet2(nn.Module):
    def __init__(self, in_channels, classes):
        super(Unet2, self).__init__()
        self.n_channels = in_channels
        self.n_classes = classes

        self.inc = InConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


def categorical_dice(mask1, mask2, label_class=1):
    """
    Dice score of a specified class between two volumes of label masks.
    (classes are encoded but by label class number not one-hot )
    Note: stacks of 2D slices are considered volumes.
    Args:
        mask1: N label masks, numpy array shaped (H, W, N)
        mask2: N label masks, numpy array shaped (H, W, N)
        label_class: the class over which to calculate dice scores
    Returns:
        volume_dice
    """
    mask1_pos = (mask1 == label_class).astype(np.float32)
    mask2_pos = (mask2 == label_class).astype(np.float32)
    dice = 2 * np.sum(mask1_pos * mask2_pos) / (np.sum(mask1_pos) + np.sum(mask2_pos))
    return dice


model = Unet2(1,4).to(device)

Loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

num_workers = 0
batch_size = 5

# Saves all the images in the variables train_set and valid_set.
train_set = CustomDataset(train_data_path)
valid_set = CustomDataset(val_data_path)
test_set = TestDataset(test_data_path)

# train_set = CustomDataset(folder_mask_train)
training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)

# test_set = CustomDataset(folder_data_val)
valid_loader = DataLoader(dataset=valid_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)




mini_batch =1

for epoch in range(20):  # loop over the dataset multiple times
    model.train()
    running_loss = 0.0
    running_dice_loss = 0.0
    for iteration, sample in enumerate(training_data_loader):
        img, mask = sample
        img = img.to(device)
        mask = mask.to(device)
        im = img.unsqueeze(1)

        # The optimised gradients set to zero. https://pytorch.org/docs/stable/optim.html
        optimizer.zero_grad()

        mask_p = model(im)  # Returns the predicted mask. Forward probacation

        # Calculate the cross entropy loss for the predicted mask and the actual mask.
        #print(mask.size())
        #print(mask_p.size())
        loss = Loss(mask_p, mask.long())

        loss.backward()  # Backward probacation
        optimizer.step()


        running_loss += loss.item()
        if iteration % mini_batch == mini_batch - 1:  # print every 2000 mini-batches
            print('[%d, %5d] training loss: %.3f' %
                  (epoch + 1, iteration + 1, running_loss / mini_batch))
            running_loss = 0.0

    for i, data in enumerate(valid_loader, 0):
        model.eval()
        with torch.no_grad():
            # get the inputs; data is a list of [inputs, labels] and move them to
            # the current device
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            im = inputs.unsqueeze(1)
            # print(labels)

            # forward + backward + optimize
            outputs = model(im)
            out = torch.argmax(outputs, 1)
            loss = Loss(outputs, labels.long())
            Dice_loss = 0
            for i in range(0,4):
                Dice_loss = Dice_loss+categorical_dice(out.detach().numpy(),labels.detach().numpy(),i)
            Dice_loss = Dice_loss/4
            #print(categorical_dice(outputs, labels))
            running_loss += loss.item()
            running_dice_loss += Dice_loss

            if i % mini_batch == mini_batch - 1:  # print every 2000 mini-batches
                print('[%d, %5d] valid loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / mini_batch))
                print('[%d, %5d] dice loss: %.3f' %
                      (epoch + 1, i + 1, running_dice_loss/mini_batch))
                running_loss = 0.0
                running_dice_loss = 0.0

now = datetime.now()
current_time = now.strftime("%H-%M_%d-%m-%y")

PATH = './networks/cifar_net_'+ current_time+'.pth'
torch.save(model.state_dict(), PATH)
print("Saved as: "+str(PATH))
print('Finished Training')

print("Displaying output")
# PATH = './networks/cifar_net_13-15_23-11-21.pth'
# model.load_state_dict(torch.load(PATH))


with torch.no_grad():
    model.eval()
    for i,data in enumerate(test_loader,0):
        inputs = data
        print(inputs.size())
        im = inputs.unsqueeze(1)
        #imshow(torchvision.utils.make_grid(inputs))
        inputs = inputs.to(device)
        outputs = model(im)
        #print(outputs.size())
        out = torch.argmax(outputs,1)
        out = out *0.196
        output = out[0]
        input = inputs[0]
        for a in range(1,out.size(0)):
            output = torch.cat((output,out[a]),axis=1)
            input = torch.cat((input, inputs[a]), axis=1)
        fig, axes = plt.subplots(nrows=2, ncols=1)
        im = axes[0].imshow(input,cmap='gray')
        clim = im.properties()['clim']
        axes[1].imshow(output,cmap='gray')

        plt.show()
        imshow(torchvision.utils.make_grid(outputs))

