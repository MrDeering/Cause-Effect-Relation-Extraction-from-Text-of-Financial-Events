from scifinance.models import AdvancedNER
from scifinance.processing import out_num_tags
from scifinance.processing import pos_num_tags
from scifinance.processing import batchPosProcessing
from scifinance.processing import Tokenizer
from scifinance.processing import out_id2tag

import scifinance as sf

import json
import copy
import torch

device = sf.getDevice()
bert_path = 'bert/bert-base-chinese'
model_path = 'saved/model.pkl'
seq_len= 144
real_len = seq_len - 2


class Predictor:
    def __init__(self):
        self.tokenizer = Tokenizer(bert_path)
        self.model = AdvancedNER(bert_path, out_num_tags, pos_num_tags)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)

    def predict(self, content: dict) -> dict:
        _text = content['document'][0]['text']
        text = [_text[:real_len]]
        input1 = self.tokenizer.fit(text, seq_len).to(device)
        input2 = batchPosProcessing(text, seq_len).to(device)
        self.model.eval()
        with torch.no_grad():
            predicts = self.model.predict(input1, input2)[0]
        for ix, tag in enumerate(predicts):
            predicts[ix] = out_id2tag[tag]
        length = min(len(_text), real_len)
        predicts = predicts[1:length + 1]
        reason_n_b, reason_n_e = [], []
        reason_p_b, reason_p_e = [], []
        result_n_b, result_n_e = [], []
        result_p_b, result_p_e = [], []

        def tagging(begin, end, tag_b, tag_i):
            pos = 0
            for i in range(length):
                if predicts[i] == tag_i:
                    if i == 0: predicts[i] = tag_b
                    elif predicts[i-1] != tag_b and predicts[i-1] != tag_i:
                        predicts[i] = tag_b
                elif predicts[i] == tag_b:
                    if i == 0: continue
                    elif predicts[i-1] == tag_b:
                        predicts[i] = tag_i
            while pos < length:
                if predicts[pos] == tag_b:
                    begin.append(pos)
                    pos += 1
                    while pos < length and predicts[pos] == tag_i: pos += 1
                    end.append(pos - 1)
                else: pos += 1

        tagging(reason_n_b, reason_n_e, 'B-RC', 'I-RC')
        tagging(reason_p_b, reason_p_e, 'B-RP', 'I-RP')
        tagging(result_n_b, result_n_e, 'B-CC', 'I-CC')
        tagging(result_p_b, result_p_e, 'B-CP', 'I-CP')

        tmp = copy.deepcopy(content['qas'][0][0])
        content['qas'][0] = [copy.deepcopy(tmp) for _ in range(5)]
        content['qas'][0][0]['question'] = '原因中的核心名词'
        content['qas'][0][1]['question'] = '原因中的谓语或状态'
        content['qas'][0][3]['question'] = '结果中的核心名词'
        content['qas'][0][4]['question'] = '结果中的谓语或状态'

        def fill_in(begin, end, i):
            tmp = copy.deepcopy(content['qas'][0][i]['answers'][0])
            content['qas'][0][i]['answers'] = [copy.deepcopy(tmp) for _ in range(len(begin))]
            for ix, (b, e) in enumerate(zip(begin, end)):
                content['qas'][0][i]['answers'][ix]['start'] = b
                content['qas'][0][i]['answers'][ix]['end'] = e
                content['qas'][0][i]['answers'][ix]['text'] = _text[b:e+1]

        fill_in(reason_n_b, reason_n_e, 0)
        fill_in(reason_p_b, reason_p_e, 1)
        fill_in(result_n_b, result_n_e, 3)
        fill_in(result_p_b, result_p_e, 4)

        return content


if __name__ == "__main__":
    example_input = '{"document": [{"block_id": "0", "text": "08年4月，郑煤集团拟以非公开发行的方式进行煤炭业务整体上市，解决与郑州煤电同业竞争问题，但之后由于股市的大幅下跌导致股价跌破发行价而被迫取消整体上市。"}], "key": "79c29068d30a686", "qas": [[{"question": "中心词", "answers": [{"start_block": "0", "start": 57, "end_block": "0", "end": 58, "text": "导致", "sub_answer": null}]}]]}'
    example_output = '{"document": [{"block_id": "0", "text": "08年4月，郑煤集团拟以非公开发行的方式进行煤炭业务整体上市，解决与郑州煤电同业竞争问题，但之后由于股市的大幅下跌导致股价跌破发行价而被迫取消整体上市。"}], "key": "79c29068d30a686", "qas": [[{"question": "原因中的核心名词", "answers": [{"start_block": "0", "start": 50, "end_block": "0", "end": 51, "text": "股市", "sub_answer": null}]}, {"question": "原因中的谓语或状态", "answers": [{"start_block": "0", "start": 53, "end_block": "0", "end": 56, "text": "大幅下跌", "sub_answer": null}]}, {"question": "中心词", "answers": [{"start_block": "0", "start": 57, "end_block": "0", "end": 58, "text": "导致", "sub_answer": null}]}, {"question": "结果中的核心名词", "answers": [{"start_block": "0", "start": 59, "end_block": "0", "end": 60, "text": "股价", "sub_answer": null}]}, {"question": "结果中的谓语或状态", "answers": [{"start_block": "0", "start": 61, "end_block": "0", "end": 65, "text": "跌破发行价", "sub_answer": null}]}]]}'
    obj = Predictor()
    output = obj.predict(json.loads(example_input))
    print(output != json.loads(example_output))
