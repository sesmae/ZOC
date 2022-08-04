from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import os
from torchvision.datasets import CocoDetection
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from tqdm import tqdm
from transformers import BertGenerationTokenizer
import copy


class my_coco_detetction():
    def __init__(self, train=True):
        if train:
            filename = 'train2017'
        else:
            filename = 'val2017'
            print('file is val')

        super(my_coco_detetction, self).__init__()

        self.transform = Compose([
            Resize(224, interpolation=Image.BICUBIC),  # 224 for vit, 288 for res50x4
            CenterCrop(224),  # 224 for vit, 288 for res50x4
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])

        self.coco_dataset = CocoDetection(root=os.path.join('./data/MS-COCO/images', filename),
                                 annFile=os.path.join('./data/MS-COCO/annotations', 'captions_{}.json'.format(filename)))

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, index):
        img = self.transform(self.coco_dataset[index][0])
        captions = self.coco_dataset[index][1]
        cap_list=[]
        for i, caption in enumerate(captions):
            if i==5:
                #print('more than 5 captions for this image', index)
                break
            cap = caption['caption']
            cap_list.append(cap)
        if len(cap_list)<5:
           print('has less than 5 captions', index)
        return img, cap_list


def get_clip_image_features(coco_dataset, split, clip_backbone, device):
    clip_model = torch.jit.load(os.path.join('./trained_models', "{}.pt".format(clip_backbone))).to(device).eval()
    if os.path.isfile('./dataloaders/processed_coco/{}/5xCaptions/full_coco_clip_features_{}.npy'.format(clip_backbone, split)):
        with open('./dataloaders/processed_coco/{}/5xCaptions/full_coco_clip_features_{}.npy'.format(clip_backbone, split), 'rb') as e:
            clip_out_all = np.load(e, allow_pickle=True)
    else:
        print('calculating all clip image encoder features')
        loader = DataLoader(dataset=coco_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
        clip_out_all = []
        with torch.no_grad():
            for i, (images, annot) in enumerate(tqdm(loader)):
                #if i == 1: break
                images =torch.stack(images)
                clip_out = clip_model.encode_image(images.to(device))
                clip_out_all.append(clip_out.cpu().numpy())
            clip_out_all = np.concatenate(clip_out_all)
        with open('./dataloaders/processed_coco/{}/5xCaptions/full_coco_clip_features_{}.npy'.format(clip_backbone, split), 'wb') as e:
            np.save(e, clip_out_all, allow_pickle=True)

    return clip_out_all


def get_bos_sentence_eos(coco_dataset, berttokenizer, split, clip_backbone):
    if os.path.isfile('./dataloaders/processed_coco/{}/5xCaptions/full_coco_processed_annot_{}.npy'.format(clip_backbone, split)):
        with open('./dataloaders/processed_coco/{}/5xCaptions/full_coco_processed_annot_{}.npy'.format(clip_backbone, split), 'rb') as e:
            bos_sentence_eos = np.load(e, allow_pickle=True)
            bos_sentence_eos = bos_sentence_eos.tolist()
    else:
        print('preprocessing all sentences...')
        bos_sentence_eos = []
        for i, (image, captions) in enumerate(tqdm(coco_dataset)):
            #if i==128:break
            for caption in captions:
                bos_sentence_eos.append(berttokenizer.bos_token + ' ' + caption + ' ' + berttokenizer.eos_token)
        with open('./dataloaders/processed_coco/{}/5xCaptions/full_coco_processed_annot_{}.npy'.format(clip_backbone, split), 'wb') as e:
            np.save(e, bos_sentence_eos, allow_pickle=True)
    return bos_sentence_eos


def get_bert_training_features(coco_dataset, train, clip_backbone):
    berttokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')
    sentences = get_bos_sentence_eos(coco_dataset, berttokenizer, train, clip_backbone)
    print('tokenizing all processed sentences...')
    tokenized = berttokenizer(sentences, padding=True,
                              truncation=True, max_length=77,
                              return_token_type_ids=False, return_tensors='np')

    label_ids = copy.deepcopy(tokenized['input_ids'])
    label_ids[label_ids == 0] = -100
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    return input_ids, attention_mask, label_ids


def collate_fn(batch):
    return tuple(zip(*batch))


def get_loader(train, clip_backbone):
    if train:
        split='train'
    else:
        split='val'

    coco_dataset = my_coco_detetction(train)
    clip_features = get_clip_image_features(coco_dataset, split, clip_backbone, device='cuda')
    input_ids, attention_mask, label_ids = get_bert_training_features(coco_dataset, split, clip_backbone)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    label_ids = torch.tensor(label_ids, dtype=torch.long)
    clip_features = torch.tensor(clip_features, dtype=torch.long)
    print(input_ids.size(), attention_mask.size(), label_ids.size(), clip_features.size())
    hidden_size = clip_features.size(1)
    print(clip_features.repeat(1,5).view(-1, hidden_size).size())
    dataset = TensorDataset(input_ids, attention_mask, label_ids, clip_features.repeat(1,5).view(-1, hidden_size))
    loader = DataLoader(dataset=dataset, batch_size=128, num_workers=8, shuffle=True)
    return loader


if __name__=='__main__':
    #with open('./processed_coco/{}/coco_clip_features_{}.npy'.format('ViT-B32', 'train'),'rb') as e:
    #    clip_out_all = np.load(e, allow_pickle=True)
    #print(np.shape(clip_out_all))

    dset = my_coco_detetction(train=True)
    max_length=0
    for i, (image, captions) in enumerate(tqdm(dset)):
        pass
