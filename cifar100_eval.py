import argparse
import torch
from transformers import BertGenerationTokenizer, BertGenerationDecoder, BertGenerationConfig
import os
from dataloaders.ZO_Clip_loaders import cifar100_single_isolated_class_loader
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import torchvision
import time

def tokenize_for_clip(batch_sentences, tokenizer):
    default_length = 77  # CLIP default
    sot_token = tokenizer.encoder['<|startoftext|>']
    eot_token = tokenizer.encoder['<|endoftext|>']
    tokenized_list = []
    for sentence in batch_sentences:
        text_tokens = [sot_token] + tokenizer.encode(sentence) + [eot_token]
        tokenized = torch.zeros((default_length), dtype=torch.long)
        tokenized[:len(text_tokens)] = torch.tensor(text_tokens)
        tokenized_list.append(tokenized)
    tokenized_list = torch.stack(tokenized_list)
    return tokenized_list


def greedysearch_generation_topk(clip_embed):
    max_len=77
    N = 1  # batch has single sample
    max_len=77
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
                             encoder_hidden_states=clip_embed.unsqueeze(1).to(device),
                             )

        pred_idx = out.logits.argmax(2)[:, -1]
        _, top_k = torch.topk(out.logits, dim=2, k=35)
        top_k_list.append(top_k[:, -1].flatten())
        target_list.append(pred_idx)
        if len(target_list) == 10:  # the entitiy word is in at most first 10 words
            break
    top_k_list = torch.cat(top_k_list)
    return target_list, top_k_list


def image_decoder(clip_model, berttokenizer, device, split, image_loaders=None):
    seen_labels = split[:20]
    seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]

    targets = torch.tensor(2000*[0] + 8000*[1])
    ood_probs_sum = []
    max_num_entities = 0
    for i, semantic_label in enumerate(split):
        loader = image_loaders[semantic_label]
        start_time = time.time()
        for idx, image in enumerate(tqdm(loader)):
            with torch.no_grad():
                clip_out = clip_model.encode_image(image.to(device)).float()
            clip_extended_embed = clip_out.repeat(1, 2).type(torch.FloatTensor)

            #greedy generation
            target_list, topk_list = greedysearch_generation_topk(clip_extended_embed)

            target_tokens = [berttokenizer.decode(int(pred_idx.cpu().numpy())) for pred_idx in target_list]
            topk_tokens = [berttokenizer.decode(int(pred_idx.cpu().numpy())) for pred_idx in topk_list]

            unique_entities = list(set(topk_tokens) - set(seen_labels))
            if len(unique_entities) > max_num_entities:
                max_num_entities = len(unique_entities)
            all_desc = seen_descriptions + [f"This is a photo of a {label}" for label in unique_entities]
            all_desc_ids = tokenize_for_clip(all_desc, cliptokenizer)

            image_feature = clip_model.encode_image(image.cuda()).float()
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
            text_features = clip_model.encode_text(all_desc_ids.cuda()).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            zeroshot_probs = (100.0 * image_feature @ text_features.T).softmax(dim=-1).squeeze()

            #detection score is accumulative sum of probs of generated entities
            ood_prob_sum = np.sum(zeroshot_probs[20:].detach().cpu().numpy())
            ood_probs_sum.append(ood_prob_sum)
        end_time=time.time()
    auc_sum = roc_auc_score(np.array(targets), np.squeeze(ood_probs_sum))
    print('sum_ood AUROC={}'.format(auc_sum))
    return auc_sum


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_path', type=str, default='./trained_models/COCO/')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.saved_model_path = args.trained_path + '/ViT-B32/'

    if not os.path.exists(args.saved_model_path):
        os.makedirs(args.saved_model_path)

    # initialize tokenizers for clip and bert, these two use different tokenizers
    berttokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')

    clip_model = torch.jit.load(os.path.join('./trained_models', "{}.pt".format('ViT-B-32'))).to(device).eval()
    cliptokenizer = clip_tokenizer()

    bert_config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
    bert_config.is_decoder=True
    bert_config.add_cross_attention=True
    bert_model = BertGenerationDecoder.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder', config=bert_config).to(device).train()
    bert_model.load_state_dict(torch.load(args.saved_model_path + 'model.pt')['net'])

    cifar100_loaders = cifar100_single_isolated_class_loader()
    dset = torchvision.datasets.CIFAR100(root='./data/', train=False, download=True)
    idx2class = {v:k for k,v in dset.class_to_idx.items()}

    splits = [list(range(20)), list(range(20, 40)), list(range(40, 60)), list(range(60, 80)), list(range(80, 100))]
    auc_list = []
    idx_splits = []
    for seen_split in splits:
        split=seen_split+list(set(list(range(100)))-set(seen_split))
        label_split = [idx2class[idx] for idx in split]
        print(label_split)
        auc = image_decoder(clip_model, berttokenizer, device, label_split, image_loaders=cifar100_loaders)
        auc_list.append(auc)
    print(auc_list)
    print('mean AUC={}'.format(np.mean(auc_list)), ' std={}'.format(np.std(auc_list)))
