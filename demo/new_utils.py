import os, pickle
import re
import sys
sys.path.append('.')
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter

from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.config import cfg, update_config

def get_pose_sparse_img(video_name, index, model):
    path = '/home/molly/UCF_data/jpegs_256/v_'+video_name
    img_path = path+'/frame' + str(index).zfill(6)+'.jpg'
    print(img_path)
    img = cv2.imread(img_path)
    shape_dst = np.min(img.shape[0:2])
    with torch.no_grad():
        paf, heatmap, im_scale = get_outputs(img, model,  'rtpose')
    humans = paf_to_pose_cpp(heatmap, paf, cfg)
    image_h, image_w = img.shape[:2]
    pose_image = np.zeros((image_h, image_w), dtype = "uint8")
    pose_image = cv2.cvtColor(pose_image, cv2.COLOR_GRAY2BGR)
    centers = {}
    for human in humans:
        # draw point
        for i in range(CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue
            body_part = human.body_parts[i]
            #print(body_part)
            center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
            centers[i] = center
            cv2.circle(pose_image, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)

        # draw line
        for pair_order, pair in enumerate(CocoPairsRender):
            if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                continue
            cv2.line(pose_image, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)
    pose_img_resize = cv2.resize(pose_image, (224,224))
    #print(pose_img_resize.shape)
    if not os.path.exists('/home/molly/UCF_data/pose_flow/'+video_name+'/'):
        os.mkdir('/home/molly/UCF_data/pose_flow/'+video_name+'/')
    cv2.imwrite('/home/molly/UCF_data/pose_flow/'+video_name+'/frame'+ str(index).zfill(6)+'.jpg', pose_img_resize)
    #cv2.imshow('',pose_img_resize)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #print(pose_image)


def load_frame_count():
    #print '==> Loading frame number of each video'
    frame_count = {}
    with open('/home/molly/two-stream-action-recognition/dataloader/dic/frame_count.pickle','rb') as file:
        dic_frame = pickle.load(file)
    file.close()
    for line in dic_frame :
        videoname = line.split('_',1)[1].split('.',1)[0]
        #print(videoname) >> TableTennisShot_g23_c04
        n,g = videoname.split('_',1)
        #print(n) >> TableTennisShot
        #print(g) >> g23_c04
        if n == 'HandStandPushups':
            videoname = 'HandstandPushups_'+ g
        frame_count[videoname]=dic_frame[line]
    return frame_count
