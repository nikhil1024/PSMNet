from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from models import *
from PIL import Image
from scipy.io import savemat
import preprocess

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default='./trained/pretrained_model_KITTI2015.tar',
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.KITTI == '2015':
   from dataloader import KITTI_submission_loader as DA
else:
   from dataloader import KITTI_submission_loader2012 as DA  

# test_left_img, test_right_img = DA.dataloader(args.datapath)
test_left_img, test_right_img, test_left_disp = DA.dataloader_val(args.datapath)

if args.model == 'stackhourglass':
    model = stackhourglass(int(args.maxdisp))
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

# if args.loadmodel is not None:
#     state_dict = torch.load(args.loadmodel)
#     model.load_state_dict(state_dict['state_dict'])
#     print("Loaded model")
# exit()
# print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
    model.eval()

    if args.cuda:
        imgL = imgL.cuda()
        imgR = imgR.cuda()     

    with torch.no_grad():
        output = model(imgL,imgR)
    output = torch.squeeze(output).data.cpu().numpy()
    return output

def main():
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(**normal_mean_var)])
    transform2 = transforms.Compose([transforms.ToTensor()])
    
    best_rate = 9999.0
    best_model = 0
    for model_no in range(1, 300):
        if args.loadmodel is not None:
            # print(args.loadmodel + str(model) + '.tar')
            # state_dict = torch.load(args.loadmodel)
            state_dict = torch.load(args.loadmodel + str(model_no) + '.tar')
            model.load_state_dict(state_dict['state_dict'])
        
        avg_rate = 0
        for inx in range(len(test_left_img)):

            imgL_o = Image.open(test_left_img[inx]).convert('RGB')
            imgR_o = Image.open(test_right_img[inx]).convert('RGB')
            dispL_o = Image.open(test_left_disp[inx])
            
            w, h = left_img.size

            left_img = left_img.crop((w-1232, h-368, w, h))
            right_img = right_img.crop((w-1232, h-368, w, h))
            w1, h1 = left_img.size
            
            dataL = dataL.crop((w-1232, h-368, w, h))
            dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
            
            processed = preprocess.get_transform(augment=False)  
            imgL = processed(left_img)
            imgR = processed(right_img)

#             imgL = infer_transform(imgL_o)
#             imgR = infer_transform(imgR_o)
#             dispL = transform2(dispL_o).squeeze(0).detach().numpy()
          
          
          

            # pad to width and hight to 16 times
#             if imgL.shape[1] % 16 != 0:
#                 times = imgL.shape[1]//16       
#                 top_pad = (times+1)*16 -imgL.shape[1]
#             else:
#                 top_pad = 0

#             if imgL.shape[2] % 16 != 0:
#                 times = imgL.shape[2]//16                       
#                 right_pad = (times+1)*16-imgL.shape[2]
#             else:
#                 right_pad = 0    

#             imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
#             imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

            start_time = time.time()
            img = test(imgL,imgR)
            # print('time = %.2f' %(time.time() - start_time))

#             if top_pad !=0 or right_pad != 0:
#                 img = pred_disp[top_pad:,:-right_pad]
#             else:
#                 img = pred_disp

#             img = (img*256).astype('uint16')

            mask = np.logical_and(dispL >= 0.001, dispL <= int(args.maxdisp))
            rate = np.sum(np.abs(img[mask] - dispL[mask]) > 3.0) / np.sum(mask)
            
            avg_rate += rate

            # save_path = r'./predictions_png/'
            # os.makedirs(save_path, exist_ok=True)
            
            # --- SAVE AS MAT FILE ---
            # save_dict = {
            #   'prediction': img
            # }
            # savemat(save_path + "{}.mat".format(test_left_img[inx].split('/')[-1].split('.')[0]), save_dict)
            
            # --- SAVE AS PNG IMAGE ---
            # img = Image.fromarray(img)
            # img.save(save_path + test_left_img[inx].split('/')[-1])
    
        avg_rate = avg_rate / len(test_left_img)
        print("===> Model {} -- Total {} Frames ==> AVG 3 Px Error Rate: {:.4f}".format(model_no, len(test_left_img), avg_rate))
        
        if avg_rate < best_rate:
            best_rate = avg_rate
            best_model = model_no
    
    print("\n\n===> Best Model: {} ==> Best AVG 3 Px Error Rate: {:.4f}".format(best_model, best_rate))


if __name__ == '__main__':
    main()






