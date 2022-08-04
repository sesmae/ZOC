import argparse
import torch
from transformers import BertGenerationTokenizer, BertGenerationDecoder, BertGenerationConfig
import os
from dataloaders.zeroshot_openset_COCO_loaders import get_loader
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
from transformers import AdamW
from PIL import Image
from tqdm import tqdm
import numpy as np
import gc
from sklearn.metrics import roc_auc_score
#from zeroshot_weights_generator import zeroshot_classifier
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage


class cifar10_isolated_class(Dataset):
    def __init__(self, class_label=None):
        assert class_label, 'a semantic label should be specified'
        super(cifar10_isolated_class, self).__init__()
        self.transform = Compose([
            ToPILImage(),
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])
        cifar10 = CIFAR10(root='/home/sesmae2/data', train=False)

        class_mask = np.array(cifar10.targets) == cifar10.class_to_idx[class_label]
        self.data = cifar10.data[class_mask]
        self.targets = np.array(cifar10.targets)[class_mask]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.transform(self.data[index])


def cifar10_single_isolated_class_loader():
    loaders_dict = {}
    cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for label in cifar10_labels:
       dataset = cifar10_isolated_class(label)
       loader = DataLoader(dataset=dataset, batch_size=1, num_workers=4)
       loaders_dict[label] = loader
    return loaders_dict


def tokenize_for_clip(batch_sentences, tokenizer):
    default_length = 77  # CLIP default
    sot_token = tokenizer.encoder['<|startoftext|>']
    eot_token = tokenizer.encoder['<|endoftext|>']
    tokenized_list = []
    for sentence in batch_sentences:
        text_tokens = [sot_token] + tokenizer.encode(sentence)[:75] + [eot_token]
        tokenized = torch.zeros((default_length), dtype=torch.long)
        tokenized[:len(text_tokens)] = torch.tensor(text_tokens)
        tokenized_list.append(tokenized)
    tokenized_list = torch.stack(tokenized_list)
    return tokenized_list


def get_label_recall(sentences, semantic_label):
    TP = 0
    for sentence in sentences:
        for word in sentence:
            if word == semantic_label:
                TP += 1
                #print('semantic label=({}), entity found in sentence'.format(semantic_label))
                break
    class_recall = TP/len(sentences)
    return class_recall


def train_decoder(bert_model, train_loader, eval_loader, cifar10_isolated_loaders, optimizer):
    num_batch = len(iter(train_loader))
    acc_loss = 0
    best_loss = 100
    for epoch in args.num_epochs:
        print('Training : epoch {}'.format(epoch))
        for i, batch in enumerate(tqdm(train_loader)):
            input_ids, attention_mask, label_ids, clip_embeds = batch

            # clip hidden size=512 but bert decoder hidden size is 1024, repeating the clip embedding twice to get 1024
            clip_extended_embed = clip_embeds.repeat(1, 2).type(torch.FloatTensor)

            N, seq_length = input_ids.shape
            position_ids = torch.arange(0, seq_length).expand(N, seq_length)
            bert_model.train()
            out = bert_model(input_ids=input_ids.to(device),
                             position_ids=position_ids.to(device),
                             attention_mask=attention_mask.to(device),
                             #encoder_attention_mask=torch.ones(batch_size, 1024).to(device),
                             encoder_hidden_states=clip_extended_embed.unsqueeze(1).to(device),
                             labels=label_ids.to(device))

            out.loss.backward(retain_graph=False)
            optimizer.step()
            optimizer.zero_grad()
            acc_loss += out.loss.detach().item()
            gc.collect()
            #print('loss in this batch:{}\n'.format(out.loss))
            if i % 100 == 0:
                train_results.write('This iteration loss={} recorded every 100 iteration\n'.format(acc_loss / num_batch))

        validation_loss = eval_decoder(bert_model, eval_loader)
        if validation_loss < best_loss :
            state = {'net': bert_model.state_dict(),
                     'epoch': epoch,
                     'validation loss': best_loss}
            torch.save(state, args.saved_model_path+'clip_bert_model.pt')

        print('Average loss on {} training batches in this epoch:{}\n'.format(num_batch, acc_loss/num_batch))
        train_results.write('Average loss in this epoch:{}\n'.format(acc_loss / num_batch))
        train_results.flush()
        os.fsync(train_results.fileno())

    recall, topk_recall = image_decoder( clip_model, berttokenizer, device, generated_cifar10, max_len=77, image_loaders=cifar10_isolated_loaders)

    return recall, topk_recall


def eval_decoder(bert_model, eval_loader):
    num_batch = len(iter(eval_loader))
    print('evaluating loss on validation data ...')
    acc_loss = 0
    bert_model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_loader)):
            input_ids, attention_mask, label_ids, clip_embeds = batch

            clip_extended_embed = clip_embeds.repeat(1, 2).type(torch.FloatTensor)

            N, seq_length = input_ids.shape
            position_ids = torch.arange(0, seq_length).expand(N, seq_length)
            out = bert_model(input_ids=input_ids.to(device),
                                 position_ids=position_ids.to(device),
                                 attention_mask=attention_mask.to(device),
                                 #encoder_attention_mask=torch.ones(batch_size, 1024).to(device),
                                 encoder_hidden_states=clip_extended_embed.unsqueeze(1).to(device),
                                 labels=label_ids.to(device))
            acc_loss += out.loss.detach().item()
            gc.collect()

    print('Average loss on {} validation batches={}\n'.format(num_batch, acc_loss/num_batch))
    train_results.write('Average validation loss in this epoch:{}\n'.format(acc_loss / num_batch))
    train_results.flush()
    os.fsync(train_results.fileno())

    return acc_loss

def greedysearch_generation_topk(clip_embed, max_len):
    N = 1  # batch has single sample
    target_list = [torch.tensor(berttokenizer.bos_token_id)]
    top_k_list = []
    bert_model.eval()
    for i in range(max_len):
        target = torch.LongTensor(target_list).unsqueeze(0)
        position_ids = torch.arange(0, len(target)).expand(N, len(target)).to(device)
        with torch.no_grad():
            out = bert_model(input_ids=target.to(device),
                             position_ids=position_ids,
                             attention_mask=torch.ones(len(target)).unsqueeze(0).to(device),
                             encoder_hidden_states=clip_embed.unsqueeze(1).to(device))

        pred_idx = out.logits.argmax(2)[:, -1]
        _, top_k = torch.topk(out.logits, dim=2, k=35)
        top_k_list.append(top_k[:, -1].flatten())
        target_list.append(pred_idx)
        if pred_idx == berttokenizer.eos_token_id or len(target_list)==10: #the entitiy word is in at most first 10 words
            break
    top_k_list = torch.cat(top_k_list)
    return target_list, top_k_list



def image_decoder(clip_model, berttokenizer, device, caption_results,  max_len=77, image_loaders=None):
    topk_recall_list = []
    splits = [['airplane', 'automobile', 'truck', 'horse', 'cat', 'bird', 'ship', 'deer', 'dog', 'frog'],
                   ['airplane', 'bird', 'deer', 'cat', 'horse', 'dog', 'ship', 'automobile', 'frog', 'truck'],
                   ['dog', 'automobile', 'truck', 'ship', 'horse', 'airplane', 'bird', 'cat', 'deer', 'frog'],
                   ['dog', 'horse', 'automobile', 'ship', 'deer', 'frog', 'airplane', 'truck', 'bird', 'cat'],
                   ['ship', 'automobile', 'dog', 'cat', 'deer', 'frog', 'airplane', 'truck', 'bird', 'horse']]

    print('calculating retrieval recall on cifar10 and cifar100')
    auc_list_mean = []
    auc_list_sum = []
    for split in splits:
        seen_labels = split[:6]
        seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]

        targets = torch.tensor(6000*[0] + 4000*[1])
        #targets = torch.tensor(60*[0] + 40*[1])

        max_num_entities=0
        max_probs = []
        ood_probs_sum = []
        ood_probs_mean = []
        recall_list = []
        for i, semantic_label in enumerate(split):
            loader = image_loaders[semantic_label]
            pred_list = []
            topk_pred_list = []
            for idx, image in enumerate(tqdm(loader)):
                #if idx==10:break
                with torch.no_grad():
                    clip_out = clip_model.encode_image(image.to(device)).float()
                clip_extended_embed = clip_out.repeat(1, 2).type(torch.FloatTensor)

                #greedy generation
                target_list, topk_list = greedysearch_generation_topk(clip_extended_embed, max_len)

                target_tokens = [berttokenizer.decode(int(pred_idx.cpu().numpy())) for pred_idx in target_list]
                topk_tokens = [berttokenizer.decode(int(pred_idx.cpu().numpy())) for pred_idx in topk_list]

                unique_entities = list(set(topk_tokens) - {semantic_label})  # TODO: changed to count for seen semantic among extracted entities
                if len(unique_entities) > max_num_entities:
                    max_num_entities = len(unique_entities)
                all_desc = seen_descriptions + [f"This is a photo of a {label}" for label in unique_entities]
                #print(all_desc)
                all_desc_ids = tokenize_for_clip(all_desc, cliptokenizer)

                image_feature = clip_model.encode_image(image.cuda()).float()
                image_feature /= image_feature.norm(dim=-1, keepdim=True)

                text_features = clip_model.encode_text(all_desc_ids.cuda()).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)
                # print(image_features.size(), text_features.size())
                zeroshot_probs = (100.0 * image_feature @ text_features.T).softmax(dim=-1).squeeze()

                # checked ensembling over prompts, gave very bad result!
                #text_features = zeroshot_classifier(clip_model, unique_entities)
                #zeroshot_probs = (100.0 * image_feature.float() @ text_features.float()).softmax(dim=-1).squeeze()

                #detection score is max prob among all seen and generated entities
                top_prob, _ = zeroshot_probs.cpu().topk(1, dim=-1)
                max_probs.append(top_prob.detach().numpy())

                #detection score is accumulative sum of probs of generated entities
                ood_prob_sum = np.sum(zeroshot_probs[6:].detach().cpu().numpy())
                #print(zeroshot_probs[6:])
                #print('ood probs sum', ood_prob_sum)
                ood_probs_sum.append(ood_prob_sum)

                #detection score is mean of probs of generated entities
                ood_prob_mean = np.mean(zeroshot_probs[6:].detach().cpu().numpy())
                #print('ood probs mean', ood_prob_mean)
                ood_probs_mean.append(ood_prob_mean)

                #caption_results.write(target_tokens)
                #print(target_tokens)
                pred_list.append(target_tokens)
                topk_pred_list.append(topk_tokens)
            recall = get_label_recall(pred_list, semantic_label)
            topk_recall = get_label_recall(topk_pred_list, semantic_label)
            recall_list.append(recall)
            topk_recall_list.append(topk_recall)
            print(' for {} : image to text Recall={}, topk_recall={}'.format(semantic_label, recall, topk_recall))
            train_results.write(' for {} : image to text Recall={}, topk_recall={}'.format(semantic_label, recall, topk_recall))
            print('maximum number of predicted entities', max_num_entities)
        print('Average retrieval Recall={}, topk_recall={}'.format(np.mean(recall_list), np.mean(topk_recall_list)))
        print('maximum number of predicted entities', max_num_entities)
        train_results.write('Average retrieval Recall={}, topk_recall={}'.format(np.mean(recall_list), np.mean(topk_recall_list)))
        auc_max = roc_auc_score(np.array(targets), np.squeeze(max_probs))
        auc_sum = roc_auc_score(np.array(targets), np.squeeze(ood_probs_sum))
        auc_mean = roc_auc_score(np.array(targets), np.squeeze(ood_probs_mean))
        auc_list_mean.append(auc_mean)
        auc_list_sum.append(auc_sum)

        print('max prob AUROC ={}, sum_ood AUROC={}, mean_ood AUROC={}'.format(auc_max, auc_sum, auc_mean))
    print('auc mean', np.mean(auc_list_mean), np.std(auc_list_mean))
    print('auc sum', np.mean(auc_list_sum), np.std(auc_list_sum))
    return np.mean(recall_list), np.mean(topk_recall_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=25, help="End epoch")  # trained with 200 epochs
    parser.add_argument('--trained_path', type=str, default='./trained_models/COCO/')
    args = parser.parse_args()

    train_results = open('coco_img2text_encoderdecoder.txt', 'w+')
    generated_cifar10 = open('cifar10_captions.txt', 'w+')
    generated_cifar100 = open('cifar100_captions.txt', 'w+')
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.saved_model_path = args.trained_path + '/fast_training/'

    if not os.path.exists(args.saved_model_path):
        os.makedirs(args.saved_model_path)

    # initialize tokenizers for clip and bert, these two use different tokenizers
    berttokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')

    # load COCO train and test captions
    train_loader = get_loader(train=True, clip_backbone='ViT-B32')
    eval_loader = get_loader(train=False, clip_backbone='ViT-B32')

    clip_model = torch.jit.load(os.path.join('./trained_models', "{}.pt".format('ViT-B32'))).to(device).eval()
    cliptokenizer = clip_tokenizer()

    bert_config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
    bert_config.is_decoder=True
    bert_config.add_cross_attention=True
    bert_model = BertGenerationDecoder.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder',
                                                       config=bert_config).to(device).train()

    # default optimizer used in the huggingface example
    optimizer = AdamW(bert_model.parameters(), lr=args.lr)

    bert_model.load_state_dict(torch.load(args.saved_model_path + 'clip_bert_model_new.pt')['net'])
    cifar10_loaders = cifar10_single_isolated_class_loader()

    recall, topk_recall = train_decoder(bert_model, train_loader, eval_loader, cifar10_loaders, optimizer)

    image_decoder(clip_model, berttokenizer, device, generated_cifar10,  max_len=77, image_loaders=cifar10_loaders)

