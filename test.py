import argparse
import os
import random

import numpy as np
from PIL import Image, ImageDraw
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torchvision import transforms

from model_oxford import NetG, RNN_ENCODER

from datasets_oxford import TextDataset, prepare_data
from torch.autograd import Variable
from tqdm.auto import tqdm


from random import randint
import cv2
169, 193, 118

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='flower')
parser.add_argument('--root', type=str, default='./data/cub-200-2011')
parser.add_argument('--CAPTIONS_PER_IMAGE', type=int, default=10)
parser.add_argument('--checkpoint', type=str, default='./snapshots/ckpt/',
                    help='The filename of pickle checkpoint.')
parser.add_argument('--WORDS_NUM', type=int, default=16)
parser.add_argument('--BRANCH_NUM', type=int, default=1)
parser.add_argument('--base_size', type=int, default=256)
parser.add_argument('--save_dir', type=str, default='./test/cub-200-2011/20_50/three')
parser.add_argument('--save_only', type=str, default='./test/cub-200-2011/20_50/only')
parser.add_argument('--save_real', type=str, default='./test/cub-200-2011/20_50/real')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--gpu', type=str, default="0,1,2")
args = parser.parse_args()


#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
if use_cuda:
    torch.backends.cudnn.benchmark = True

print(device)


writer = SummaryWriter()

size = (args.image_size, args.image_size)
train_tf = transforms.Compose([
        transforms.Resize(size)
])

dataset_test = TextDataset(args.root, 'test',
                           base_size=args.base_size, 
                           CAPTIONS_PER_IMAGE=args.CAPTIONS_PER_IMAGE,
                           WORDS_NUM=args.WORDS_NUM, 
                           transform=train_tf)
assert dataset_test
test_set = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, drop_last=True) 

print(len(test_set))

ixtoword_test = dataset_test.ixtoword

text_encoder = RNN_ENCODER(dataset_test.n_words, nhidden=args.image_size)
text_encoder_path = './DAMSMencoders/cub-200-2011/text_encoder200.pth'
state_dict = torch.load(text_encoder_path, map_location=lambda storage, loc: storage)
text_encoder.load_state_dict(state_dict)
for p in text_encoder.parameters():
    p.requires_grad = False
print('Load text encoder from:', text_encoder_path)
text_encoder.eval()

if use_cuda:
    text_encoder = text_encoder.cuda()

def random_irregular_mask(img):
    """Generates a random irregular mask with lines, circles and elipses"""
    transform = transforms.Compose([transforms.ToTensor()])
    mask_shape = img[:, 0, :, :].unsqueeze(1)
    mask = torch.ones_like(mask_shape)
    size = img[0].size()
    img = np.zeros((size[1], size[2], 1), np.uint8)

    # Set size scale
    max_width = 30
    if size[1] < 64 or size[2] < 64:
        raise Exception("Width and Height of mask must be at least 64!")

    number = random.randint(2, 7)

    thickness_base = randint(20, 80)

    for _ in range(number):

        model = random.random()

        if model < 0.3:
            # Draw random lines
            x1, x2 = randint(1, size[1]), randint(1, size[1])
            y1, y2 = randint(1, size[2]), randint(1, size[2])
            
        else:
        # Draw random lines
            x1, y1 = randint(int(0.2*size[1]), int(0.5*size[1])), randint(int(0.2*size[1]), int(0.8*size[1]))

            xm, ym = randint(int(0.4*size[1]), int(0.5*size[1])), randint(int(0.4*size[1]), int(0.6*size[1]))

            x2, y2 = 2*xm-x1, 2*ym-y1

        
        a = random.randint(-10, 10)
    
        cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness_base + a)


    img = img.reshape(size[2], size[1])
    img = Image.fromarray(img*255)
    
    img_mask = transform(img)
    mask[:, :, :, :] = img_mask < 1
    mask = 1. - mask
   

    return mask


def load_mask(img, min=None, max=None):
    
    B, C, H, W = img.size()
    percentage = 0

    while (percentage < min) or (percentage > max):
       
        mask = random_irregular_mask(img)
        percentage = (mask[0][0].sum().item()/(H*W)) * 100
            
    return mask


nz = 100
noise = Variable(torch.FloatTensor(args.batch_size, nz))
if use_cuda:
    noise = noise.cuda()

for i in range():

    base = 0
    interval = 0

    if not os.path.exists(args.save_dir + str(base + i * interval)):
        os.makedirs('{:s}'.format(args.save_dir + str(base + i * interval)))

    if not os.path.exists(args.save_only + str(base + i * interval)):
        os.makedirs('{:s}'.format(args.save_only + str(base + i * interval)))

    if not os.path.exists(args.save_real + str(base + i * interval)):
        os.makedirs('{:s}'.format(args.save_real + str(base + i * interval)))

    g_model = nn.DataParallel(NetG().to(device))
    g_checkpoint = torch.load(args.checkpoint + 'G_' + str(base + i * interval) + '.pth', map_location=device)
    g_model.load_state_dict(g_checkpoint)
    g_model.eval()

    print('```````````````````````````````')
    print(args.checkpoint + 'G_' + str(base + i * interval) + '.pth')
    print('```````````````````````````````')
    
    for step, data_test in enumerate(tqdm(test_set), 0):

        imgs, captions, cap_lens, class_ids, keys = prepare_data(data_test)
        

        hidden = text_encoder.init_hidden(args.batch_size)
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
        text_mask = (captions == 0)
        num_words = words_embs.size(2)
        if text_mask.size(1) > num_words:
            text_mask = text_mask[:, :num_words]
            
        
        img = imgs[-1]
        mask = load_mask(img, min=20, max=50)       
        masked = img * (1. - mask)
        
        noise.data.normal_(0, 1)
        fake, fake_c_t, att = g_model(masked, mask, noise, sent_emb, words_embs, text_mask)
        
        for bb in range(args.batch_size):
            
            ims = torch.cat([masked, fake, img], dim=3)
            ims_test = ims.add(1).div(2).mul(255).clamp(0, 255).byte() # denorm
            ims_test = ims_test[bb].permute(1, 2, 0).data.cpu().numpy()
                
            cap_back = Image.new('RGB', (ims_test.shape[1], 30), (255, 255, 255))
            cap = captions[bb].data.cpu().numpy()

            sentence = []
            for j in range(len(cap)):
                if cap[j] == 0:
                    break
                word = ixtoword_test[cap[j]].encode('ascii', 'ignore').decode('ascii')
                sentence.append(word)
            sentence = ' '.join(sentence)
            
            draw = ImageDraw.Draw(cap_back)
            draw.text((0, 10), sentence, (0, 0, 0))
            cap_back = np.array(cap_back)
            
            ims_text = np.concatenate([ims_test, cap_back], 0)
            ims_out = Image.fromarray(ims_text)
            fullpath = '%s/%s.png' % (args.save_dir + str(base + i * interval), keys[bb].split('/')[-1])
            ims_out.save(fullpath)

             # only
            ims_test = fake.add(1).div(2).mul(255).clamp(0, 255).byte()
            ims_test = ims_test[bb].permute(1, 2, 0).data.cpu().numpy()
            ims_out = Image.fromarray(ims_test)
            fullpath = '%s/%s.png' % (args.save_only + str(base + i * interval), keys[bb].split('/')[-1])
            ims_out.save(fullpath)

            
            # real
            ims_test = img.add(1).div(2).mul(255).clamp(0, 255).byte()
            ims_test = ims_test[bb].permute(1, 2, 0).data.cpu().numpy()
            ims_out = Image.fromarray(ims_test)
            fullpath = '%s/%s.png' % (args.save_real + str(base + i * interval), keys[bb].split('/')[-1])
            ims_out.save(fullpath)
