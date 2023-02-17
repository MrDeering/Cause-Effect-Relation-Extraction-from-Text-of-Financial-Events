from scifinance.processing import generate_tags_from_txt
from scifinance.processing import Tokenizer
from scifinance.processing import batchPosProcessing
from scifinance.processing import getDataLoader

import scifinance as sf
import scifinance.models as models
import scifinance.processing as ps

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import sys
import getopt
import torch

if __name__ == "__main__":
    seq_len = 142
    # bert_path = 'bert/bert-base-chinese'
    bert_path = r'D:\Documents\Python\Jupyter\WriteLab\NLP\bert-base-chinese'
    data_path = 'train.txt'
    device = sf.getDevice()
    options, _ = getopt.getopt(sys.argv[1:], "s:l:e:d:")
    save_path, load_path = "", ""
    epochs = 100
    for option, arg in options:
        if option == '-s': save_path = arg
        elif option == '-l': load_path = arg
        elif option == '-e': epochs = int(arg)
        elif option == '-d': data_path = arg
    model = models.AdvancedNER(bert_path, ps.out_num_tags, ps.pos_num_tags)
    if load_path != "":
        model.load_state_dict(torch.load(load_path, map_location=device))
    model = model.to(device)
    tokenizer = Tokenizer(bert_path)
    texts, targets = generate_tags_from_txt(data_path, seq_len)
    input1 = tokenizer.fit(texts, seq_len)
    input2 = batchPosProcessing(texts, seq_len)
    train_loader, valid_loader = getDataLoader(input1, input2, targets)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=10)
    writer = SummaryWriter()
    dummy_input1 = input1.clone()[:5].to(device)
    dummy_input2 = input2.clone()[:5].to(device)
    dummy_targets = targets.clone()[:5].to(device)
    writer.add_graph(model, [dummy_input1, dummy_input2, dummy_targets])
    sf.train(model, train_loader, valid_loader, optimizer, scheduler,
             device, epochs=epochs, tensorboard=writer, save_path=save_path)
