import argparse
import os
import numpy as np
import random
from PIL import Image, ImageDraw
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from loss import MA_GP, predict_loss, words_loss, sent_loss, PerceptualLoss
from model_oxford import NetG, NetD, CNN_ENCODER, RNN_ENCODER
from datasets_oxford import TextDataset, prepare_data
from torch.autograd import Variable
from maskgenerator import random_regular_mask, center_mask, random_irregular_mask

import warnings

from random import randint
import cv2

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='flower')
parser.add_argument('--root', type=str, default='./data/cub-200-2011')
parser.add_argument('--CAPTIONS_PER_IMAGE', type=int, default=10)
parser.add_argument('--WORDS_NUM', type=int, default=16)
parser.add_argument('--BRANCH_NUM', type=int, default=1)
parser.add_argument('--base_size', type=int, default=256)
parser.add_argument('--save_dir', type=str, default='./snapshots')
parser.add_argument('--training_image', type=str, default='./training')
parser.add_argument('--lr_g', type=float, default=1e-4, help="adam: generator learning rate")
parser.add_argument('--lr_d', type=float, default=4e-4, help="adam: discriminator learning rate")
parser.add_argument("--b1", type=float, default=0.0, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of second order momentum of gradient")
parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_threads', type=int, default=2)
parser.add_argument('--save_model_interval', type=int, default=3000)
parser.add_argument('--vis_interval', type=int, default=500)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--resume', type=int, default=108000)
#parser.add_argument('--gpu', type=str, default="1")
args = parser.parse_args()


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print(device)
if use_cuda:
    torch.backends.cudnn.benchmark = True

if not os.path.exists(args.save_dir):
    os.makedirs('{:s}/ckpt'.format(args.save_dir))
if not os.path.exists(args.training_image):
    os.makedirs('{:s}'.format(args.training_image))

writer = SummaryWriter()

size = (args.image_size, args.image_size)
train_tf = transforms.Compose([
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
])

dataset_train = TextDataset(args.root, 'train',
                            base_size=args.base_size, 
                            CAPTIONS_PER_IMAGE=args.CAPTIONS_PER_IMAGE,
                            WORDS_NUM=args.WORDS_NUM, 
                            BRANCH_NUM=args.BRANCH_NUM,
                            transform=train_tf)
assert dataset_train
train_set = DataLoader(
        dataset_train, batch_size=args.batch_size,
        drop_last=True, shuffle=True, num_workers=args.n_threads, pin_memory=True) 

print(len(train_set))

ixtoword_train = dataset_train.ixtoword

netG = nn.DataParallel(NetG()).to(device)
netD = nn.DataParallel(NetD(64, 256, 3)).to(device)

l1 = nn.L1Loss().to(device)
perceptual_loss = nn.DataParallel(PerceptualLoss()).to(device)

start_epoch = 0
g_optimizer_t = torch.optim.Adam(
    netG.parameters(),
    args.lr_g, (args.b1, args.b2))
pd_optimizer_t = torch.optim.Adam(
    netD.parameters(),
    args.lr_d, (args.b1, args.b2))

if args.resume:
    g_checkpoint = torch.load(f'{args.save_dir}/ckpt/G_{args.resume}.pth', map_location=device)
    netG.load_state_dict(g_checkpoint)
    pd_checkpoint = torch.load(f'{args.save_dir}/ckpt/PD_{args.resume}.pth', map_location=device)
    netD.load_state_dict(pd_checkpoint)
    print('Models restored')

image_encoder = CNN_ENCODER(args.image_size)
img_encoder_path = './output/output/Model/image_encoder250.pth'
state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
image_encoder.load_state_dict(state_dict)
for p in image_encoder.parameters():
    p.requires_grad = False
print('Load image encoder from:', img_encoder_path)
image_encoder.eval()

text_encoder = RNN_ENCODER(dataset_train.n_words, nhidden=args.image_size)
text_encoder_path = './output/output/Model/text_encoder250.pth'
state_dict = torch.load(text_encoder_path, map_location=lambda storage, loc: storage)
text_encoder.load_state_dict(state_dict)
for p in text_encoder.parameters():
    p.requires_grad = False
print('Load text encoder from:', text_encoder_path)
text_encoder.eval()


# clip Loss
parallel = True


if use_cuda:
    text_encoder = text_encoder.cuda()
    image_encoder = image_encoder.cuda()


def prepare_labels():
    batch_size = args.batch_size
    real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
    fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
    match_labels = Variable(torch.LongTensor(range(batch_size)))
    if use_cuda:
        real_labels = real_labels.cuda()
        fake_labels = fake_labels.cuda()
        match_labels = match_labels.cuda()

    return real_labels, fake_labels, match_labels
    
real_labels, fake_labels, match_labels = prepare_labels()


def load_mask(img, min=None, max=None):
    """Load different mask types for training and testing"""
    mask_type = random.randint(0, 2)
    #mask_type = 0
    
    if (min is not None) and (max is not None):
        B, C, H, W = img.size()
        percentage = 0

        while (percentage < min) or (percentage > max):
            
            # center mask
            if mask_type == 0:
                mask = center_mask(img)

            # random regular mask
            if mask_type == 1:
                mask = random_regular_mask(img)

            # random irregular mask
            if mask_type == 2:
                mask = random_irregular_mask(img)
    
            percentage = (mask[0][0].sum().item()/(H*W)) * 100
            

    # center mask
    if mask_type == 0:
        return center_mask(img)

    # random regular mask
    if mask_type == 1:
        return random_regular_mask(img)

    # random irregular mask
    if mask_type == 2:
        return random_irregular_mask(img)
        
    
nz = 100
noise = Variable(torch.FloatTensor(args.batch_size, nz))
fixed_noise = Variable(torch.FloatTensor(args.batch_size, nz).normal_(0, 1))
if use_cuda:
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    

start_epoch = 133
for i in range(start_epoch, args.max_epoch):
    
    iterator_train = iter(train_set)
    train_step = 0
    num_batches = len(train_set)
    while train_step < num_batches:
        
        train_step = train_step + 1

        data_train = next(iterator_train)
        imgs, captions, cap_lens, class_ids, keys = prepare_data(data_train)

        hidden = text_encoder.init_hidden(args.batch_size)
        
        # words_embs: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        # words_embs: torch.Size([4, 256, WORDS_NUM])
        # sent_emb: torch.Size([4, 256])
        
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
        text_mask = (captions == 0)
        num_words = words_embs.size(2)
        if text_mask.size(1) > num_words:
            text_mask = text_mask[:, :num_words]
        
        img = imgs[-1]
        mask = load_mask(img)
        masked = img * (1. - mask)
        
        img = img.to(device).requires_grad_()
        sent_emb = sent_emb.to(device).requires_grad_()
        words_embs = words_embs.to(device).requires_grad_()
        
        
        real_features = netD(img)
        pred_real, errD_real = predict_loss(netD.module.COND_DNET, real_features, sent_emb, negtive=False)
        mis_features = torch.cat((real_features[1:], real_features[0:1]), dim=0)
        _, errD_mis = predict_loss(netD.module.COND_DNET, mis_features, sent_emb, negtive=True)
        # synthesize fake images
        noise = torch.randn(args.batch_size, 100).to(device)
        fake, fake_c_t, att = netG(masked, mask, noise, sent_emb, words_embs, text_mask)
        fake_features = netD(fake.detach())
        _, errD_fake = predict_loss(netD.module.COND_DNET, fake_features, sent_emb, negtive=True)
        # MA-GP
        errD_MAGP = MA_GP(img, sent_emb, pred_real)
        # whole D loss
        errD = errD_real + (errD_fake + errD_mis)/2.0 + errD_MAGP
        # update D
        pd_optimizer_t.zero_grad()
        errD.backward()
        pd_optimizer_t.step()
        # update G
        fake_features = netD(fake)
        output = netD.module.COND_DNET(fake_features, sent_emb)

        errG = -output.mean()# - sim
        
        ################### Text-Image Matching Loss ####################
        region_features, cnn_code = image_encoder(fake)
        w_loss0, w_loss1, _ = words_loss(region_features, words_embs, match_labels, cap_lens, class_ids, args.batch_size, use_cuda)
        w_loss = (w_loss0 + w_loss1) * 1.0

        s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb, match_labels, class_ids, args.batch_size, use_cuda)
        s_loss = (s_loss0 + s_loss1) * 1.0
        
        matching_loss = w_loss + s_loss
        
        ################## Total Loss ##########################
        gan_percep_loss = perceptual_loss(fake_c_t, img).mean()
        attn_loss = l1(fake * att, img * att)

        total_loss_t = 0.2*gan_percep_loss + attn_loss + errG + 0.01*matching_loss 
        
        g_optimizer_t.zero_grad()
        total_loss_t.backward()
        g_optimizer_t.step()

        
        num_save_interval = num_batches * i + train_step
    
        if num_save_interval % args.save_model_interval == 0:
            torch.save(netG.state_dict(), f'{args.save_dir}/ckpt/G_{num_save_interval}.pth')
            torch.save(netD.state_dict(), f'{args.save_dir}/ckpt/PD_{num_save_interval}.pth')
            print("model saved.")
    
        if num_save_interval % args.log_interval == 0:
            writer.add_scalar('g_loss_t/gan_percep_loss', gan_percep_loss.item(), num_save_interval)
            writer.add_scalar('g_loss_t/attn_loss', attn_loss.item(), num_save_interval)
            writer.add_scalar('g_loss_t/matching_loss', matching_loss.item(), num_save_interval)
            writer.add_scalar('g_loss_t/gan_loss_t', errG.item(), num_save_interval)
            writer.add_scalar('g_loss_t/total_loss_t', total_loss_t.item(), num_save_interval)
            writer.add_scalar('d_loss_t/errD', errD.item(), num_save_interval)
            writer.add_scalar('d_loss_t/errD_MAGP', errD_MAGP.item(), num_save_interval)
            
            print('\n', num_save_interval)
            
            print('g_loss_t/total_loss_t', total_loss_t.item())
            print('g_loss_t/gan_percep_loss', gan_percep_loss.item())
            print('g_loss_t/attn_loss', attn_loss.item())
            print('g_loss_t/matching_loss', matching_loss.item())
            print('g_loss_t/gan_loss_t', errG.item())
            print('d_loss_t/errD', errD.item())
            print('d_loss_t/errD_MAGP', errD_MAGP.item())
    
        def denorm(x):
            out = (x + 1) / 2 # [-1,1] -> [0,1]
            return out.clamp_(0, 1)
        if num_save_interval % args.vis_interval == 0:
            with torch.no_grad():
                ims = torch.cat([masked.cpu(), fake_c_t.cpu(), fake.cpu(), img.cpu()], dim=3)
                ims_train = ims.add(1).div(2).mul(255).clamp(0, 255).byte()
                ims_train = ims_train[0].permute(1, 2, 0).data.cpu().numpy()
                
                cap_back = Image.new('RGB', (ims_train.shape[1], 30), (255, 255, 255))
                cap = captions[0].data.cpu().numpy()
                sentence = []
                for j in range(len(cap)):
                    if cap[j] == 0:
                        break
                    word = ixtoword_train[cap[j]].encode('ascii', 'ignore').decode('ascii')
                    sentence.append(word)
                sentence = ' '.join(sentence)
                draw = ImageDraw.Draw(cap_back)
                draw.text((0, 10), sentence, (0, 0, 0))
                cap_back = np.array(cap_back)
                
                ims_text = np.concatenate([ims_train, cap_back], 0)
                ims_out = Image.fromarray(ims_text)  
                fullpath = '%s/epoch%d_iteration%d.png' % (args.training_image, i+1, num_save_interval)
                ims_out.save(fullpath)

                print("train image saved.")
      
writer.close()
