import argparse
import torch
from transformers import BertGenerationTokenizer, BertGenerationDecoder, BertGenerationConfig
import os
from dataloaders.ZO_Clip_loaders import cifarplus_loader
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
from tqdm import tqdm
import copy
import numpy as np
from sklearn.metrics import roc_auc_score


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
        #if pred_idx == berttokenizer.eos_token_id or len(target_list)==10: #the entitiy word is in at most first 10 words
        if len(target_list) == 10:  # the entitiy word is in at most first 10 words
            break
    top_k_list = torch.cat(top_k_list)
    return target_list, top_k_list


def image_decoder(clip_model, berttokenizer, device, in_loader, out_loaders):
    seen_labels = ['airplane', 'automobile', 'ship', 'truck']
    seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]

    in_probs_sum = []
    max_num_entities = 0

    for idx, (image, label_idx) in enumerate(tqdm(in_loader)):
        #if idx==10:break
        with torch.no_grad():
            clip_out = clip_model.encode_image(image.to(device)).float()
        clip_extended_embed = clip_out.repeat(1, 2).type(torch.FloatTensor)

        #greedy generation
        target_list, topk_list = greedysearch_generation_topk(clip_extended_embed)

        target_tokens = [berttokenizer.decode(int(pred_idx.cpu().numpy())) for pred_idx in target_list]
        topk_tokens = [berttokenizer.decode(int(pred_idx.cpu().numpy())) for pred_idx in topk_list]

        unique_entities = list(set(topk_tokens))
        if len(unique_entities) > max_num_entities:
            max_num_entities = len(unique_entities)
        all_desc = seen_descriptions + [f"This is a photo of a {label}" for label in unique_entities]
        all_desc_ids = tokenize_for_clip(all_desc, cliptokenizer)

        image_feature = clip_model.encode_image(image.cuda()).float()
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        text_features = clip_model.encode_text(all_desc_ids.cuda()).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # print(image_features.size(), text_features.size())
        zeroshot_probs = (100.0 * image_feature @ text_features.T).softmax(dim=-1).squeeze()

        #detection score is accumulative sum of probs of generated entities
        ood_prob_sum = np.sum(zeroshot_probs[len(seen_labels):].detach().cpu().numpy())
        in_probs_sum.append(ood_prob_sum)
    print('maximum number of predicted entities', max_num_entities)

    ood_probs_sum_list = [[],[],[],[],[],[]]
    for i, out_loader_name in enumerate(list(out_loaders.keys())):
        print(out_loader_name)
        out_loader = out_loaders[out_loader_name]
        for idx, (image, label_idx) in enumerate(tqdm(out_loader)):
            #if idx==10:break
            with torch.no_grad():
                clip_out = clip_model.encode_image(image.to(device)).float()
            clip_extended_embed = clip_out.repeat(1, 2).type(torch.FloatTensor)

            #greedy generation
            target_list, topk_list = greedysearch_generation_topk(clip_extended_embed)

            target_tokens = [berttokenizer.decode(int(pred_idx.cpu().numpy())) for pred_idx in target_list]
            topk_tokens = [berttokenizer.decode(int(pred_idx.cpu().numpy())) for pred_idx in topk_list]

            unique_entities = list(set(topk_tokens))
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

            #detection score is accumulative sum of probs of generated entities
            ood_prob_sum = np.sum(zeroshot_probs[len(seen_labels):].detach().cpu().numpy())
            ood_probs_sum_list[i].append(ood_prob_sum)

    print('maximum number of predicted entities', max_num_entities)

    for i, out_loader_name in enumerate(out_loaders.keys()):
        out_loader = out_loaders[out_loader_name]
        targets = torch.tensor(len(in_loader.dataset)*[0] + len(out_loader.dataset)*[1])
        #targets = torch.tensor(10*[0] + 10*[1])

        probs_sum = copy.deepcopy(in_probs_sum)
        probs_sum.extend(ood_probs_sum_list[i])
        auc_sum = roc_auc_score(np.array(targets), np.squeeze(probs_sum))

        print(' OOD dataset : {}, sum_ood AUROC={}'.format(out_loader_name, auc_sum))


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
    bert_model = BertGenerationDecoder.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder',
                                                       config=bert_config).to(device).train()
    bert_model.load_state_dict(torch.load(args.saved_model_path + 'model.pt')['net'])

    in_loader, out_loaders = cifarplus_loader()
    image_decoder(clip_model, berttokenizer, device, in_loader=in_loader, out_loaders=out_loaders)