from typing import List, Dict

import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertConfig

from word_embeddings import WordEmbeddings
from wsi import Word


class SimpleBertEmbeddings(WordEmbeddings):
    tokenizer: BertTokenizer
    model: BertModel
    special_tokens = []

    def __init__(self, bert_model_path: str):
        self.tokenizer = BertTokenizer(vocab_file=bert_model_path + '/vocab.txt')
        config = BertConfig.from_pretrained(bert_model_path + '/config.json', output_hidden_states=True)
        self.model = BertModel.from_pretrained(bert_model_path, config=config)
        self.model.eval()

    def convert(self, text: str) -> Dict[Word, List[float]]:
        lower_text = text.lower().replace("й", "и").replace("ё", "е").replace("́", "")
        token_ids = self.tokenizer.encode(lower_text)
        print(token_ids)

        encoded_layers = self.model(input_ids=torch.tensor([token_ids]))
        hidden_layers = encoded_layers[2][1:]
        token_embeddings = torch.stack(hidden_layers, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)
        result: Dict[Word, List[float]] = {}
        text_pos = 0
        prev = None
        for i, token_vec in enumerate(token_embeddings):
            # todo: try only -12 layer: https://github.com/hanxiao/bert-as-service#q-so-which-layer-and-which-pooling-strategy-is-the-best
            # combine last 4 layers (best F1 score)
            cat_vec = torch.cat((token_vec[-1], token_vec[-2], token_vec[-3], token_vec[-4]), dim=0)
            if token_ids[i] in self.tokenizer.all_special_ids:
                continue
            token: str = self.tokenizer.convert_ids_to_tokens(token_ids[i])
            if token.startswith("##") and prev is not None:
                clear_token = token.replace("##", "")
                word = Word(prev.text + clear_token, prev.start, prev.end + len(clear_token))
                result.update({word:  np.add(result[prev], cat_vec.tolist()).tolist()})
                del result[prev]
                prev = word
                continue
            start = lower_text.find(token, text_pos)
            if start == -1:
                continue
            end = start + len(token)
            word = Word(token, start, end)
            text_pos = end
            prev = word
            result.update({word: cat_vec.tolist()})

        return result
