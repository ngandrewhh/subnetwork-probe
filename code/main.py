import argparse
import os

import numpy as np
from matplotlib import pyplot as plt

from bert import WordLevelBert
from classifiers import BertEncoder
from masked_bert import MaskedWordLevelBert
from train import train_ner, train_ud, train_pos, save_mask_scores
from visualize import visualize_head_sparsity, visualize_dense_sparsity, visualize_layer_attn_sparsity


# conda create -n probe python=3.7 -y
# conda activate probe
# conda install pytorch torchvision -c pytorch
# conda install matplotlib
# conda install transformers -c huggingface
# conda install nltk

# return torch.from_numpy(ndarray).pin_memory().cuda(async=True)
#   >> change to return torch.from_numpy(ndarray).pin_memory().cuda(non_blocking=True)
# from transformers.modeling_bert import ...
#   >> change to from transformers.models.bert.modeling_bert


def run(huggingface_model,
        tasks=('UPOS', 'NER', 'UD'),
        settings=('pretrained', 'resetenc', 'resetall'),
        methods=('prune', 'mlp1'),
        exp_name='new',
        **kwargs):

    for task in tasks:
        for setting in settings:
            for method in methods:

                if method == 'mlp1':
                    params_list = (1, 2, 5, 10, 25, 50, 125, 250, 768)
                elif method == 'finetune':
                    params_list = (1,)
                elif method == 'prune':
                    params_list = ((768, 768), (768, 192), (768, 24), (768, 6),
                                   (768, 1), (192, 1), (24, 1), (6, 1), (1, 1))

                for params in params_list:
                    print(task)
                    print(method)
                    if method == 'mlp1':
                        rank = params
                        print("MLP1 - Rank: {}".format(rank))
                        bert = WordLevelBert(huggingface_model)
                        bert.freeze_bert()
                        bert_encoder = BertEncoder(bert, mlp1=True, rank=rank)
                        masked = False
                    elif method == 'finetune':
                        print("Fine-tuning")
                        bert = WordLevelBert(huggingface_model)
                        bert_encoder = BertEncoder(bert, mlp1=False)
                        masked = False
                    elif method == 'prune':
                        out_w_per_mask, in_w_per_mask = params
                        print("Prune - (out,in)_w_per_mask: {}".format((out_w_per_mask, in_w_per_mask)))
                        bert = MaskedWordLevelBert(huggingface_model, out_w_per_mask, in_w_per_mask)
                        bert.freeze_bert()
                        bert_encoder = BertEncoder(bert, mlp1=False)
                        masked = True

                    if setting == 'pretrained':
                        print("Keeping pre-trained")
                    elif setting == 'resetenc':
                        print("Resetting encoder!")
                        bert.reset_weights(encoder_only=True)
                    elif setting == 'resetall':
                        print("Resetting all!")
                        bert.reset_weights(encoder_only=False)

                    kwargs.update({'masked': masked})
                    print(kwargs)

                    print("Finding subnetwork...")
                    if task == "NER":
                        log, model = train_ner(bert_encoder, './data/CoNLL-NER/eng.train',
                                               './data/CoNLL-NER/eng.testa', **kwargs)
                    elif task == "UD":
                        log, model = train_ud(bert_encoder, './data/UD_English/en-ud-train.conllu',
                                              './data/UD_English/en-ud-dev.conllu', **kwargs)
                    elif task == "UPOS":
                        log, model = train_pos(bert_encoder, './data/UD_English/en-ud-train.conllu',
                                               './data/UD_English/en-ud-dev.conllu', **kwargs)

                    os.makedirs('./out', exist_ok=True)
                    path = f"./out/exp_{exp_name}_task_{task}_set_{setting}_met_{method}_params_{str(params).replace(', ', '_')}"
                    save_mask_scores(model, log, base=path)

                    print("Final results: {}".format(log[-1]))
                    for key in log[0].keys():
                        plt.plot(np.arange(len(log)), [a[key] for a in log], color='blue')
                        plt.title(key)
                        plt.savefig('{}_{}_graph.png'.format(path, key))
                        plt.show(block=False)
                        plt.close()

                    if masked:
                        visualize_head_sparsity(bert, path, block=False)
                        visualize_dense_sparsity(bert, path, block=False)
                        visualize_layer_attn_sparsity(bert, path, block=False)
                        print("Percentage of elements within 0.01 of 0 or 1: {:.5f}".format(bert.compute_binary_pct()))


if __name__ == "__main__":
    import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--huggingface_model', default='bert-base-uncased')
    parser.add_argument('-n', '--exp_name', default=datetime.datetime.now().strftime('%Y%m%d_%H%M'))
    parser.add_argument('--lambda_init', default=0)
    parser.add_argument('--lambda_final', default=1)
    parser.add_argument('-e', '--epochs', default=5)
    parser.add_argument('-b', '--batch_size', default=100)
    parser.add_argument('-s', '--subbatch_size', default=25)
    parser.add_argument('-lr', '--lr_base', default=1e-3)
    parser.add_argument('-lm', '--mask_lr_base', default=0.2)
    parser.add_argument('-v', '--verbose', default=True)
    args = parser.parse_args()

    plt.switch_backend('agg')
    run(**args.__dict__)


