#%matplotlib inline
import argparse
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
import PIL
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as tvt
import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
import torch.nn as nn
import cv2
import torchvision.models as models
import torch.nn.functional as F
import urllib
import torch.optim as optim
import PIL
from PIL import Image, ImageDraw, ImageFont
import time
import logging
from operator import itemgetter
import cv
parser = argparse.ArgumentParser ( description = 'HW04 COCO downloader')
parser.add_argument ( '--root_path' , required = True , type =str )
parser.add_argument ( '--coco_json_path_train', required = True ,type = str )
parser.add_argument ( '--class_list' , required = True , nargs ='*' , type = str )
parser.add_argument ( '--images_per_class' , required = True ,type = int )
args , args_other = parser.parse_known_args ()
class coco_downloader():
    def __init__(self,x,y,z,w,tv,transform):
        #pylab.rcParams['figure.figsize'] = (8.0, 10.0)
        annFile=w
        self.coco=COCO(annFile)
        self.images_per_class=z
        self.y=y
        self.x=x
        self.tv=tv
        self.img_dict={}
        self.im={}
        self.class_list_index = 0
        dtype = torch.float64
        img_id_dictionary={}
        for id in self.y:
         catIds = self.coco.getCatIds(catNms=[id]);
         imgIds = self.coco.getImgIds(catIds=catIds);
         img_id_dictionary[id] =set(imgIds)
        common_in_all=set.intersection(*img_id_dictionary.values())
        common_in_01=set.intersection(img_id_dictionary['horse'],img_id_dictionary['person'])
        common_in_01.difference(common_in_all)
        common_in_02 = set.intersection(img_id_dictionary['horse'], img_id_dictionary['dog'])
        common_in_02.difference(common_in_all)
        common_in_03 = set.intersection(img_id_dictionary['dog'], img_id_dictionary['person'])
        common_in_03.difference(common_in_all)
        all_used_id=set.union(common_in_all,common_in_01,common_in_02,common_in_03)
        self.all_used_id = list(all_used_id)
        self.Cat_Ids = self.coco.getCatIds(catNms=[self.y[0], self.y[1], self.y[2]]);  # get cat id of the instances

    def __getitem__(self,index):
        for i in range(self.images_per_class):
            ann_id = self.coco.getAnnIds(imgIds=self.all_used_id[i], catIds=self.Cat_Ids)
            load_dog_ann_id = self.coco.loadAnns(ann_id)
            load_dog_ann_id=sorted(load_dog_ann_id, key=lambda k: k['area'],reverse=True)
            self.bbox_num_img_cat = np.asarray([cat['bbox'] for cat in load_dog_ann_id])
            cat_id_index = [cat['category_id'] for cat in load_dog_ann_id]
            catid = list(set([cat['category_id'] for cat in load_dog_ann_id]))
            num_obj = len(cat_id_index)
            load_cat_id = self.coco.loadCats(cat_id_index)
            self.label = np.asarray([cat['name'] for cat in load_cat_id])
            self.label_all = np.zeros((len(self.label), len(self.y)))
            for l in range(len(self.label)):
                if self.label[l] == self.y[0]:
                    self.label_all[l] = [1, 0, 0]
                if self.label[l] == self.y[1]:
                    self.label_all[l] = [0, 1, 0]
                if self.label[l] == self.y[2]:
                    self.label_all[l] = [0, 0, 1]
            img = self.coco.loadImgs(self.all_used_id[i])[0]
            img_height = [cat['height'] for cat in self.coco.loadImgs(self.all_used_id[i])]
            img_width = [cat['width'] for cat in self.coco.loadImgs(self.all_used_id[i])]
            image1 = io.imread(img['coco_url'])
            if (len(image1.shape) < 3):
                image1=cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(image1)
            r, j = (img.size)
            img = img.resize((128, 128), Image.BOX)
            self.demo_img = transform(img)
            r, j = r / 128, j / 128
            img_path = os.path.join(self.x, self.tv, "allbbx")
            if not os.path.exists(self.x):
                os.mkdir(self.x, mode=0o666)
            if not os.path.exists(img_path):
                os.makedirs(img_path, mode=0o666)
            img.save(img_path + "/" + str(self.all_used_id[i]) + ".jpg")
            for length_of_bbox in range(self.bbox_num_img_cat.shape[0]):
                b0 = (int(self.bbox_num_img_cat[length_of_bbox][0]) / r)
                b1 = ((self.bbox_num_img_cat[length_of_bbox][1]) / j)
                b2 = ((self.bbox_num_img_cat[length_of_bbox][2]) / r)
                b3 = ((self.bbox_num_img_cat[length_of_bbox][3]) / j)
                self.bbox_num_img_cat[length_of_bbox] = [b0, b1, b2, b3]
                '''
                with Image.open(img_path + "/" + str(self.all_used_id[i]) + ".jpg") as im:
                    draw = ImageDraw.Draw(im)
                    draw.rectangle([(b0, b1), (b0 + b2, b1 + b3)], fill=None, outline=None)
                    im.save(img_path + "/" + str(self.all_used_id[i]) + ".jpg")
                '''
            bbox_tensor = np.zeros((5, 4))
            bbox_label_tensor = np.zeros((5, 3))
            if num_obj>5:
                num_obj=5
            for ind in range(num_obj):
                bbox_tensor[ind] = self.bbox_num_img_cat[ind]
                bbox_label_tensor[ind] = self.label_all[ind]
            self.img_dict[i] = self.demo_img, bbox_tensor, bbox_label_tensor, num_obj, ann_id
            self.im=list(self.img_dict.values())
        return self.im[index][0] ,self.im[index][1],self.im[index][2],self.im[index][3]

    def __len__(self):
        self.len = len(self.img_dict)
        #self.dog_length = len(self.dog_images)
        return self.images_per_class
transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataserver_train=coco_downloader(args.root_path,args.class_list,args.images_per_class,args.coco_json_path_train,"Train",transform)
#total=len(dataserver_train)
train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                                                        batch_size=10, shuffle=True,
                                                        num_workers=0)
z=dataserver_train[0]
print("downloader part complete")


class model(nn.Module):
    class SkipBlock(nn.Module):

        def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
            super(model.SkipBlock, self).__init__()
            self.downsample = downsample
            self.skip_connections = skip_connections
            self.in_ch = in_ch
            self.out_ch = out_ch
            self.convo1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
            self.convo2 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
            norm_layer1 = nn.BatchNorm2d
            norm_layer2 = nn.BatchNorm2d
            self.bn1 = norm_layer1(out_ch)
            self.bn2 = norm_layer2(out_ch)
            if downsample:
                self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)

        def forward(self, x):
            identity = x
            out = self.convo1(x)
            out = self.bn1(out)
            out = torch.nn.functional.relu(out)
            if self.in_ch == self.out_ch:
                out = self.convo2(out)
                out = self.bn2(out)
                out = torch.nn.functional.relu(out)
            if self.downsample:
                out = self.downsampler(out)
                identity = self.downsampler(identity)
            if self.skip_connections:
                if self.in_ch == self.out_ch:
                    out += identity
                else:
                    out[:, :self.in_ch, :, :] += identity
                    out[:, self.in_ch:, :, :] += identity
            return out

    class NetForYolo(nn.Module):
        """
        The YOLO approach to multi-instance detection is based entirely on regression.  As
        was mentioned earlier in the comment block associated with the enclosing
        class, each image is represented by a 1440 element tensor that consists
        of 8-element encodings for each anchor box for every cell in the SxS
        gridding of an image.  The network I show below is a modification of the
        network class LOADnet presented earlier for the case that all we want to
        do is regression.
        """

        def __init__(self, skip_connections=True, depth=8):
            super(model.NetForYolo, self).__init__()
            if depth not in [8, 10, 12, 14, 16]:
                sys.exit("This network has only been tested for 'depth' values 8, 10, 12, 14, and 16")
            self.depth = depth // 2
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(256)
            self.skip64_arr = nn.ModuleList()
            for i in range(self.depth):
                self.skip64_arr.append(model.SkipBlock(64, 64,
                                                       skip_connections=skip_connections))
            self.skip64ds = model.SkipBlock(64, 64, downsample=True,
                                            skip_connections=skip_connections)
            self.skip64to128 = model.SkipBlock(64, 128,
                                               skip_connections=skip_connections)
            self.skip128_arr = nn.ModuleList()
            for i in range(self.depth):
                self.skip128_arr.append(model.SkipBlock(128, 128, skip_connections=skip_connections))
            self.skip128ds = model.SkipBlock(128, 128, downsample=True, skip_connections=skip_connections)
            self.skip128to256 = model.SkipBlock(128, 256, skip_connections=skip_connections)
            self.skip256_arr = nn.ModuleList()
            for i in range(self.depth):
                self.skip256_arr.append(model.SkipBlock(256, 256, skip_connections=skip_connections))
            self.skip256ds = model.SkipBlock(256, 256, downsample=True, skip_connections=skip_connections)
            self.fc = nn.Linear(8192, 1440)

        def forward(self, x):
            x = self.pool(torch.nn.functional.relu(self.conv1(x)))
            x = nn.MaxPool2d(2, 2)(torch.nn.functional.relu(self.conv2(x)))
            for i, skip64 in enumerate(self.skip64_arr[:self.depth // 4]):
                x = skip64(x)
            x = self.skip64ds(x)
            for i, skip64 in enumerate(self.skip64_arr[self.depth // 4:]):
                x = skip64(x)
            x = self.bn1(x)
            x = self.skip64to128(x)
            for i, skip128 in enumerate(self.skip128_arr[:self.depth // 4]):
                x = skip128(x)
            x = self.bn2(x)
            x = self.skip128ds(x)
            x = x.view(-1, 8192)
            x = torch.nn.functional.relu(self.fc(x))
            return x


def training(model, train_dataloader, epoch, yolo_interval):
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-6, momentum=0.98)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    model = model.to(device)
    num_yolo_cells = (128 // yolo_interval) * (128 // yolo_interval)
    num_anchor_boxes = 5  # (height/width)   1/5  1/3  1/1  3/1  5/1
    # yolo_tensor = torch.zeros(self.rpg.batch_size, num_yolo_cells, num_anchor_boxes, 8)
    '''
    class AnchorBox:
        def __init__(self, AR, tlc, ab_height, ab_width):  # toc : top_left_corner      ab : anchor box
            self.AR = AR
            self.tlc = tlc
            self.ab_height = ab_height
            self.ab_width = ab_width
    '''
    filename_for_out1 = "performance_numbers_for_Training" + str(epoch) + "label.txt"
    filename_for_out2 = "performance_numbers_for_Training" + str(epoch) + "regres.txt"
    FILE1 = open(os.path.join(args.root_path, filename_for_out1), 'w')
    FILE2 = open(os.path.join(args.root_path, filename_for_out2), 'w')
    elapsed_time = 0.0
    start_time = time.perf_counter()
    Loss_tally = []
    Loss_tally_label = []
    model = model.to(device)
    for epoch in range(epoch):
        model.train()
        running_loss = 0.0
        running_loss_label = 0.0
        print("starting epoch", epoch)
        for iter, data in enumerate(train_dataloader):
            print("in iter", iter)
            im_tensor, bbox_tensor, bbox_label_tensor, num_objects_in_image = data
            im_tensor = im_tensor.to(device)
            bbox_tensor = bbox_tensor.to(device)
            bbox_label_tensor = bbox_label_tensor.to(device)
            num_objects_in_image = num_objects_in_image.to(device)
            yolo_tensor = torch.zeros(im_tensor.shape[0], num_yolo_cells, num_anchor_boxes, 8)
            num_cells_image_width = 128 // yolo_interval
            num_cells_image_height = 128 // yolo_interval
            for ibx in range(im_tensor.shape[0]):
                if num_objects_in_image[ibx] > 5:
                    num_objects_in_image[ibx] = 5
                    temp = []
                for idx in range(num_objects_in_image[ibx]):
                    height_center_bb = (bbox_tensor[ibx][idx][1].item() + ((bbox_tensor[ibx][idx][3].item()) // 2))
                    width_center_bb = (bbox_tensor[ibx][idx][0].item() + ((bbox_tensor[ibx][idx][2].item()) // 2))
                    obj_bb_height = bbox_tensor[ibx][idx][3].item()
                    obj_bb_width = bbox_tensor[ibx][idx][2].item()
                    label = (bbox_label_tensor[ibx][idx])
                    AR = float(obj_bb_height) / float(obj_bb_width)
                    if (obj_bb_height < 4) or (obj_bb_width < 4):
                        continue

                    if AR <= 0.2:
                        anchbox = 0
                        # print("the chosen anchbox is",0)
                    elif AR <= 0.5:
                        anchbox = 1
                        # print("the chosen anchbox is", 1)
                    elif AR <= 1.5:
                        anchbox = 2
                        # print("the chosen anchbox is", 2)
                    elif AR <= 4:
                        anchbox = 3
                        # print("the chosen anchbox is", 3)
                    elif AR > 4:
                        anchbox = 4
                        # print("the chosen anchbox is", 4)
                    cell_row_idx = height_center_bb // yolo_interval  ## for the i coordinate
                    cell_col_idx = width_center_bb // yolo_interval  ## for the j coordinates     cell_row_idx = height_center_bb // yolo_interval  ## for the i coordinate
                    bh = float(obj_bb_height) / float(yolo_interval)
                    bw = float(obj_bb_width) / float(yolo_interval)
                    obj_center_x = float(bbox_tensor[ibx][idx][2].item() / 2.0 + bbox_tensor[ibx][idx][0].item())
                    obj_center_y = float(bbox_tensor[ibx][idx][3].item() / 2.0 + bbox_tensor[ibx][idx][1].item())
                    # print("the obj", (obj_center_x), (obj_center_y))
                    '''
                    x = im_tensor[ibx]
                    save_image(x,"x.jpg")
                    imagenew = cv2.imread('x.jpg')
                    imagenew = cv2.circle(imagenew, (int(obj_center_x), int(obj_center_y)), 10, (0, 0, 255), -1)
                    cv2.imshow('fig', imagenew)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    '''
                    yolocell_center_i = cell_row_idx * yolo_interval + float(yolo_interval) / 2.0
                    yolocell_center_j = cell_col_idx * yolo_interval + float(yolo_interval) / 2.0
                    del_x = float(obj_center_x - yolocell_center_j) / yolo_interval
                    del_y = float(obj_center_y - yolocell_center_i) / yolo_interval
                    yolo_vector = [1, del_x, del_y, bh, bw, label[0], label[1], label[2]]
                    yolo_cell_index = int(cell_row_idx * num_cells_image_width + cell_col_idx)
                    yolo_tensor[ibx, yolo_cell_index, anchbox] = torch.FloatTensor(yolo_vector)
                    yolo_tensor_flattened = yolo_tensor.view(im_tensor.shape[0], -1)
            optimizer.zero_grad()
            output = model(im_tensor)
            outreshaped = output.view(im_tensor.shape[0], 36, 5, 8)
            yolo_tensor_resh = yolo_tensor_flattened.view(im_tensor.shape[0], 36, 5, 8)
            yolo_tensor_flattened = yolo_tensor_flattened.to(device)
            loss = criterion2(output, yolo_tensor_flattened)
            outreshaped = outreshaped[:, :, :, -3:]
            yolo_tensor_resh = yolo_tensor_resh[:, :, :, -3:]
            # x=torch.argmax(outreshaped,dim=3)
            x = outreshaped
            y = torch.argmax(yolo_tensor_resh, dim=3)
            x = x.type(torch.DoubleTensor)
            x = x.view(im_tensor.shape[0] * 36 * 5, -1)
            y = y.view(im_tensor.shape[0] * 36 * 5)
            loss_label = criterion1(x, y)
            loss_label.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_loss_label += loss_label.item()
            if iter % 50 == 49:
                current_time = time.perf_counter()
                elapsed_time = current_time - start_time
                avg_loss = running_loss / float(100)
                avg_loss_label = running_loss_label / float(100)
                print("[epoch:%d/%d, iter=%4d  elapsed_time=%5d secs]      mean MSE loss: %7.4f" %
                      (epoch + 1, epoch, iter + 1, elapsed_time, avg_loss))
                print("[epoch:%d/%d, iter=%4d  elapsed_time=%5d secs]      mean cross entropy for label loss: %7.4f" %
                      (epoch + 1, epoch, iter + 1, elapsed_time, avg_loss_label))
                Loss_tally.append(running_loss)
                Loss_tally_label.append(running_loss_label)
                print("the loss tally", Loss_tally, Loss_tally_label)
                FILE1.write("the bbox loss %.3f\n" % avg_loss)
                FILE1.write("the label loss %.3f\n" % avg_loss_label)
                FILE1.flush()
                running_loss = 0.0
                running_loss_label = 0.0

    print("\nFinished Training\n")
    plt.figure(figsize=(10, 5))
    plt.title("Loss vs. Iterations")
    plt.plot(Loss_tally)
    plt.xlabel("iterations")
    plt.ylabel("Loss YOLO")
    plt.savefig("training_loss_YOLO.png")
    plt.show()
    print("this is the epoch", epoch)
    plt.figure(figsize=(10, 5))
    plt.plot(Loss_tally_label)
    plt.xlabel("iterations")
    plt.ylabel("Loss label")
    plt.savefig("training_loss_label.png")
    plt.show()
    torch.save(model.state_dict(), os.path.join(args.root_path, "trainmodel.pth"))



model = model.NetForYolo()
training(model, train_dataloader, 4, 20)










