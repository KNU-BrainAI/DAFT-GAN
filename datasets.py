import numpy as np
import torchvision.transforms as transforms
import os
import pandas as pd
import pickle
import torch.utils.data as data
import numpy.random as random
import torch

from PIL import Image
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict




def prepare_data(data):
    imgs, captions, captions_lens, class_ids, keys = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    #print(len(imgs))
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if torch.cuda.is_available():
            real_imgs.append(imgs[i].cuda())
        else:
            real_imgs.append(imgs[i])

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if torch.cuda.is_available():
        captions = captions.cuda()
        sorted_cap_lens = sorted_cap_lens.cuda()
    else:
        captions = captions
        sorted_cap_lens = sorted_cap_lens


    return [real_imgs, captions, sorted_cap_lens, class_ids, keys]




'''data_dir = "/home/hdd1/hdkim/mmfl/data/birds"
split = "train"
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
])'''


def get_imgs(img_path, BRANCH_NUM, imsize, bbox,
            transform=None, normalize=None):
    
    # 출력은 base_size와 BRANCH_NUM을 이용하여 (base_size)x(base_size), (base_size x 2)x(base_size x 2), ... , (args.image_size)x(args.image_size) 의 크기를 갖는 이미지 텐서들
    # 예를 들어 base_size=64, BRANCH_NUM=2 일때 두가지 이미지 텐서 출력하며, 사이즈는 첫번째가 3x64x64, 두번째가 3x256x256 
    # offical 코드에서 train할 때는 인덱스에 -1을 무조건 주어서 3x(args.image_size)x(args.image_size) 이미지 텐서 이용
            
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    
    for i in range(BRANCH_NUM):
        #print(imsize[i])
        if i < (BRANCH_NUM - 1):
            re_img = transforms.Resize(imsize[i])(img)
        else:
            re_img = img
        ret.append(normalize(re_img))
    
    return ret

'''base_size = 64 
    
filenames = load_text_data(data_dir, split)[0]
key = filenames[0]
print(key)
img_path = '%s/CUB_200_2011/images/%s.jpg' % (data_dir, key)

BRANCH_NUM = 3
imsize = []
for i in range(BRANCH_NUM):
    imsize.append(base_size)
    base_size = base_size * 2

print(f"imsize: {imsize}")

bbox = load_bbox()
bbbox = bbox[key]

norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

get_imgs(img_path, BRANCH_NUM, imsize, bbbox, transform, norm)'''



class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=64, CAPTIONS_PER_IMAGE=10, WORDS_NUM=16, BRANCH_NUM=2,
                 transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = CAPTIONS_PER_IMAGE
        self.BRANCH_NUM = BRANCH_NUM
        self.WORDS_NUM = WORDS_NUM

        self.imsize = []
        for i in range(self.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('cub-200-2011') != -1:
            self.bbox = self.load_bbox()
            print(f"bbox is load")
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_bbox(self): 
        # {'200.Common_Yellowthroat/Common_Yellowthroat_0055_190967': [x-left, y-top, width, height], ' ' : []} 이런 형태의 원소 11788개의 dictionary
        
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'bounding_boxes.txt') # 11788개
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True, # delim_whitespace는 공백으로 구분된 파일 읽을때 사용
                                        header=None).astype(int) # 불러올 데이터가 header가 없을때
        
        filepath = os.path.join(data_dir, 'images.txt') # 11788개 
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        
        filenames = df_filenames[1].tolist() #df_filenames[0]은 1~11788까지의 index
        print('Total filenames: ', len(filenames))
        
        filename_bbox = {img_file[:-4]: [] for img_file in filenames} # [:-4]로 끝의 .jpg 제거. dictionary임
        # {'200.Common_Yellowthroat/Common_Yellowthroat_0055_190967': []} 이런 형태
        
        numImgs = len(filenames) # 11788
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4] # key는 .jpg 뗀 파일 이름
            filename_bbox[key] = bbox # {'200.Common_Yellowthroat/Common_Yellowthroat_0055_190967': [x-left, y-top, width, height]} 이런 형태
        
        return filename_bbox



    def load_filenames(self, data_dir, split):
        # split(train or test) 데이터의 filenames가 담긴 pickle 파일 불러와서 리스트에 저장. 
        # 출력은 [' ', '200.Common_Yellowthroat/Common_Yellowthroat_0055_190967', ' '] 이런 형태의 원소 8855개의 리스트
        
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames




    def load_captions(self, data_dir, filenames): 
        # 출력은 [[], ... , ['a', 'bird', 'with', 'a', 'very', 'long', 'wing', 'span', 'and', 'a', 'long', 'pointed', 'beak'], ..., []] 이런 형태의 리스트
        # train인 경우 리스트의 원소의 갯수는 88550
        

        
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text_flower/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().encode('utf-8').decode('utf8').split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                            % (filenames[i], cnt))
        return all_captions




    def build_dictionary(self, train_captions, test_captions):
        
        # 출력은 [train_captions_new, test_captions_new, ixtoword, wordtoix, len(ixtoword)] 이렇게 총 5가지
        # train_captions_new는 8855개의 train 이미지에서 각각 10개의 caption안의 단어를 dictionary에 매핑한 리스트. [[], ... , [18, 19, 1, 26, 8, 14, 2, 3, 1, 85, 22, 8, 52, 36, 11, 107, 108]], ... , []] 형태
        # test_captions_new도 마찬가지
        # ixtoword는 {0: '<end>', 1: 'a', 2: 'bird', 3: 'with', 4: 'very', ... , 5448: 'frontside', 5449: 'wrapped'} 형태의 dictionary
        # wordtoix는 {'<end>': 0, 'a': 1, 'bird': 2, 'with': 3, 'very': 4, ... , 'frontside': 5448, 'wrapped': 5449} 형태의 dictionary
        # len(ixtoword)는 5450
        
        word_counts = defaultdict(float) # defaultdict()는 dictionary에 없는 키에 접근해도 KeyError 발생시키지 않고 default값 반환
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)
            
        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]




    def load_text_data(self, data_dir, split):
        # 이 함수의 출력이 바로 TextDataset 클래스의 self.filenames, self.captions, self.ixtoword, self.wordtoix, self.n_words
        # 출력은 filenames에 build_dictionary()를 거치고 나온 captions, ixtoword, wordtoix, n_words(사전 길이)
        # buid_dictionary()의 결과를 pickle로 저장되어있지 않으면 저장하고 pick을 load하여 출력하는 함수
        
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        #print("train_names:", train_names)
        test_names = self.load_filenames(data_dir, 'test')
        #print("test_names:", test_names)
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                                ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
            
        print(f"num captions: {len(captions)}")

            
        return filenames, captions, ixtoword, wordtoix, n_words







    def load_class_id(self, data_dir, total_num): 
        # 출력은 학습 데이터인 경우 총 8850개로 [2, 2, 2, 2, ..., 2, 3, 3, 3, ... , 3, ... , 200, 200, ... , 200] 이런 형태. 1부터 200까지의 bird 클래스를 train과 test셋이 나눠가짐
        
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding='bytes')
        else:
            class_id = np.arange(total_num)
            
        
        return class_id



    def get_caption(self, sent_ix):
        
        # 출력은 [[args.WORDS_NUM], 1] 의 넘파이 배열과 word의 갯수(상한은 WORDS_NUM)으로 caption의 길이가 WORDS_NUM 보다 짧으면 나머지 뒷부분은 0으로, 길면 shuffle한 다음 딱 WORDS_NUM 까지만 남김
        
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64') # np.array는 copy=True가 디폴트, np.asarray는 copy=False가 디폴트
        
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((self.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= self.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = self.WORDS_NUM
            
        return x, x_len
    
    
    
    def __getitem__(self, index):
        #
        key = self.filenames[index].split("/")[-1]
        cls_id = self.class_id[index]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        # Lung dataset is in png format 
        try:
            img_name = '%s/flower/jpg/%s.jpg' % (data_dir, key)
            imgs = get_imgs(img_name, self.BRANCH_NUM, self.imsize,
                            bbox, self.transform, normalize=self.norm)
        except IOError:
            img_name = '%s/flower/jpg/%s.png' % (data_dir, key)
            imgs = get_imgs(img_name, self.BRANCH_NUM, self.imsize,
                            bbox, self.transform, normalize=self.norm)
        # random select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        #print(new_sent_ix)
        caps, cap_len = self.get_caption(new_sent_ix)
        return imgs, caps, cap_len, cls_id, key
    
    

    def __len__(self):
        return len(self.filenames)
