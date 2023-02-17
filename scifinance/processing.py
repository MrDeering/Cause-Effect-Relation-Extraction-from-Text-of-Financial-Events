from typing import List

import jieba
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from transformers import BertTokenizer
from zhon.hanzi import punctuation #标点符号

import torch
import numpy as np
import jieba.posseg as psg

pos_tags = {
    '[PAD]': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6,
    'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13,
    'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19,
    't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26,
    '[CLS]': 27, '[SEP]': 28
}#填充，开始，分隔
pos_num_tags = len(pos_tags)#多少个键值对

# R: Reason, C: Consequence
out_tags = {
    '[PAD]': 0, 'O': 1, 'B-RC': 2, 'I-RC': 3, 'B-RP': 4, 'I-RP': 5,
    'B-C': 6, 'I-C': 7, 'B-CC': 8, 'I-CC': 9, 'B-CP': 10, 'I-CP': 11,
    '[CLS]': 12, '[SEP]': 13
}
out_num_tags = len(out_tags)
out_id2tag = dict([(ix, tag) for tag, ix in out_tags.items()])


def getPartOfSpeech(s: str, seq_len: int) -> List[int]:#返回值为List[int]类型,part of speech说明是词性
    pos_list, cnt = [0] * seq_len, 1
    pos_list[0] = pos_tags['[CLS]']
    for word, pos in psg.cut(s):

        for _ in word: #_为占位符，只是为了说明遍历这么多次
            pos_list[cnt] = pos_tags[pos[0].lower()]
            cnt = cnt + 1
    pos_list[cnt] = pos_tags['[SEP]']
    return pos_list #返回词性列表对应的数值


def get_1hot(pos_list: List[int]) -> List[List[int]]:
    r"""
        Use it with getPartOfSpeech
    """
    eye = np.eye(pos_num_tags)
    return eye[pos_list, :]#等同于return eye[pos_list],就是相当于在对应pos_list的数值上面0变成1,结果是二维数组


def batchPosProcessing(texts: List[str], seq_len) -> torch.Tensor:
    processed = []
    for text in texts:
        pos_list = getPartOfSpeech(text, seq_len)
        processed.append(get_1hot(pos_list))   #一个列表中不断地添加二维数组的元素
    return torch.tensor(processed, dtype=torch.float)


class Tokenizer(object):
    def __init__(self, bert_path):
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

    @staticmethod
    def is_chinese(ch) -> bool:
        return '\u4e00' <= ch <= '\u9fa5' or ch in punctuation#是中文就返回True

    def fit(self, texts, seq_len) -> torch.Tensor:
        outputs = []
        for _text in texts:
            text = ""
            for ch in _text:
                if self.is_chinese(ch):
                    text += ch
                else:
                    text += '[UNK]'  #unknown token
            tokens = self.tokenizer.encode(text)
            if (k := seq_len - len(tokens)) > 0:
                tokens += [0] * k
            outputs.append(tokens)
        #print(outputs)
        return torch.LongTensor(outputs)  #返回编码


def getDataLoader(raw_input1, raw_input2, raw_targets, batch_size=32, split_required=True, split_ratio=0.8):
    if split_required:
        dataset = TensorDataset(raw_input1, raw_input2, raw_targets)   #对三个tensor进行打包，类似于zip
        train_len = int(split_ratio * (length := len(dataset)))
        valid_len = length - train_len
        train_dataset, valid_dataset = random_split(dataset, (train_len, valid_len)) #设置百分之八十训练，百分之二十是用来作验证
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, valid_loader
    else:
        dataset = TensorDataset(raw_input1, raw_input2, raw_targets)
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader


def generate_tags_from_txt(txt_path: str, seq_len: int) -> (List[str], torch.Tensor):
    batch_text, batch_tags = [], []
    with open(txt_path, 'r', encoding='utf-8') as f:
        while f.readline() != '':
            text = f.readline()[4:] #文本：空格，前面总共四个字符
            r_cores = f.readline()[9:].split()
            r_preds = f.readline()[10:].split()
            centers = f.readline()[4:].split()
            c_cores = f.readline()[9:].split()
            c_preds = f.readline()[10:].split()
            if (length := len(text)) > seq_len - 2: continue  #超过最大长度的直接就过滤不做处理了
            tags = ['[CLS]'] + ['O'] * length + \
                   ['[SEP]'] + ['[PAD]'] * (seq_len - length - 2)
            pos = -1
            for r_core in r_cores:
                pos = text.find(r_core, pos + 1)
                if pos == -1: continue
                tags[pos] = 'B-RC'
                for i in range(pos + 1, pos + len(r_core)):
                    tags[i] = 'I-RC'
            pos = -1
            for r_pred in r_preds:
                pos = text.find(r_pred, pos + 1)
                if pos == -1: continue
                tags[pos] = 'B-RP'
                for i in range(pos + 1, pos + len(r_pred)):
                    tags[i] = 'I-RP'
            pos = -1
            for center in centers:
                pos = text.find(center, pos + 1)
                if pos == -1: continue
                tags[pos] = 'B-C'
                for i in range(pos + 1, pos + len(center)):
                    tags[i] = 'I-C'
            pos = -1
            for c_core in c_cores:
                pos = text.find(c_core, pos + 1)
                if pos == -1: continue
                tags[pos] = 'B-CC'
                for i in range(pos + 1, pos + len(c_core)):
                    tags[i] = 'I-CC'
            pos = -1
            for c_pred in c_preds:
                pos = text.find(c_pred, pos + 1)
                if pos == -1: continue
                tags[pos] = 'B-CP'
                for i in range(pos + 1, pos + len(c_pred)):
                    tags[i] = 'I-CP'
            batch_text.append(text)
            batch_tags.append(tags)
        f.close()
    #print(batch_tags)
    batch_tags = batch_tags2tensor(batch_tags)
    return batch_text, batch_tags


def get_out_tags_id(tags: List[str]):
    ids = [0] * len(tags)
    for ix, tag in enumerate(tags):
        ids[ix] = out_tags[tag]
    return ids


def batch_tags2tensor(batch_tags: List[List[str]]):
    targets = []
    for tags in batch_tags:
        ids = get_out_tags_id(tags)
        targets.append(ids)
    return torch.LongTensor(targets)

if __name__ == "__main__":
    seq_len = 142
    # bert_path = 'bert/bert-base-chinese'
    bert_path = r'bert/bert-base-chinese'
    data_path = 'test.txt'
    batch_text,batch_tags  =generate_tags_from_txt(data_path,seq_len)
    processed =[]
    for text in batch_text:
        pos_list = getPartOfSpeech(text, seq_len)
        print(pos_list)
        print(get_1hot(pos_list))
        processed.append(get_1hot(pos_list))
        print("   ")
    print(processed)


    # print(batch_text)
    # print(" ")
    # print(batch_tags)





# from scifinance.models import AdvancedNER

# if __name__ == "__main__":

#     seq_len = 12
#     s = ["上海自来水来自海上",
#          "海门中学适合看海"]
#     # s = [" "]
#     bert_path = r'C:\Users\LENOVO\Desktop\scifinance-master\bert-base-chinese'
#     tokenizer = Tokenizer(bert_path)
#
#     # model = AdvancedNER(bert_path, out_num_tags, pos_num_tags)
#     input1 = tokenizer.fit(s, seq_len)
#     print(input1)
#     print(input1.shape[0])

    # input2 = batchPosProcessing(s, seq_len)
    # # targets = [[12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 13, 0],
    # #            [12, 1, 1, 1, 1, 1, 1, 1, 1, 13, 0, 0]]
    # targets = [[12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 13, 0]]
    # targets = torch.tensor(targets)
    # print(input1.shape, input2.shape)
    # outputs = model(input1, input2, targets)
    # print(outputs)
    # predicts = model.predict(input1, input2)
    # print(predicts)
