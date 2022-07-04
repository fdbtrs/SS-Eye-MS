import cv2
import numpy as np
import os

def resize_img_with_border(im):
     desired_size =256
     if (len(im.shape) == 2):
         new_im = np.zeros((im.shape[0], im.shape[1], 3))
         new_im[:, :, 0] = im
         new_im[:, :, 1] = im
         new_im[:, :, 2] = im
         im = new_im
     old_size = im.shape[:2]  # old_size is in (height, width) format

     ratio = float(desired_size) / max(old_size)
     new_size = tuple([int(x * ratio) for x in old_size])

     # new_size should be in (width, height) format

     im = cv2.resize(im, (new_size[1], new_size[0]))
     print(im.shape)

     delta_w = desired_size - new_size[1]
     delta_h = desired_size - new_size[0]
     top, bottom = delta_h // 2, delta_h - (delta_h // 2)
     left, right = delta_w // 2, delta_w - (delta_w // 2)

     color = [0, 0, 0]
     new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_REPLICATE)
     return new_im















def resize_image(img):
     if (len(img.shape)==2):
         row= img.shape[0]
         column=img.shape[1]
         result=np.zeros((max(row,column),max(row,column)))
         x =int ((max(row,column) - row ) // 2)
         y= int ((max(row,column) -column) // 2)
         w = max(row, column) - x
         h = max(row, column) - y
         '''
         if (column > row):
             w = w + 1
         elif row > column:
             h = h + 1
         '''
         if (abs(row - column) == 1):
             if (row > column):
                 y = 1
             else:
                 x = 1

         result[x: w, y: h] = img
     else:
         row = img.shape[0]
         column = img.shape[1]
         result = np.zeros((max(row, column), max(row, column), img.shape[2]))
         x = int((max(row, column) - row) // 2)
         y = int((max(row, column) - column) // 2)
         w =max(row, column) - x
         h=max(row, column) - y
         '''
         if(column> row):
             w = w+1
         elif row > column:
             h=h+1
          '''

         if (abs(row - column) == 1):
              if (row > column):
                  y = 1
              else:
                  x = 1
         result[x: w, y: h,:] = img
     return result

base_dir='data/TRAIN/'
target_dir='data/TRAIN1'
label_dir=os.path.join(base_dir, 'input')
#input_dir=os.path.join(base_dir,'input')

file_list=os.listdir(label_dir)
for idm in file_list:
    if('DS' in idm):
        continue
    img_list=os.listdir(os.path.join(label_dir,idm))
    for f in img_list:
        img=cv2.imread(os.path.join(label_dir,idm,f))
        new_img = resize_img_with_border(img)
        #new_img = (new_img - np.min(new_img)) / (np.max(new_img) - np.min(new_img))
        save_f=os.path.join(target_dir,'input',idm)
        if(not os.path.isdir(save_f)):
            os.makedirs(save_f)
        cv2.imwrite(os.path.join(save_f,f),new_img )

