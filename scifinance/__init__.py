__all__ = ['io', 'models', 'processing']

from scifinance.io import read_json, write_json

from tqdm import tqdm   #进度条库
import torch
import numpy as np


def getDevice():
    if torch.cuda.is_available():
        return torch.device("cuda:0")  #选择GPU做运算
    else:
        return torch.device("cpu")


def train(model, train_loader, valid_loader,
          optimizer, scheduler=None, device="cpu",
          epochs=20, early_stop=True, save_path="",
          tensorboard=None) -> None:
    print("Training start...")
    prev_valid_losses, eps = [], 1e-6
    best_state = None
    for epoch in tqdm(range(1, epochs + 1)):

        model.train()
        train_loss, valid_loss = 0, 0
        iter_times = 0
        for input1, input2, targets in train_loader:
            iter_times += 1
            optimizer.zero_grad()
            input1 = input1.to(device)
            input2 = input2.to(device)
            targets = targets.to(device)
            loss = model(input1, input2, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= iter_times

        model.eval()
        iter_times, accuracy = 0, 0
        best_loss = np.inf
        with torch.no_grad():
            for input1, input2, targets in valid_loader:
                iter_times += 1
                input1 = input1.to(device)
                input2 = input2.to(device)
                targets = targets.to(device)
                loss = model(input1, input2, targets)
                if scheduler is not None:
                    scheduler.step(loss)
                valid_loss += loss.item()
            valid_loss /= iter_times

        show_str = "Epoch: {}, training_loss: {:.2f}, validation_loss: {:.2f}"
        print(show_str.format(epoch, train_loss, valid_loss))
        if tensorboard is not None:
            tensorboard.add_scalar('training_loss', train_loss, epoch)
            tensorboard.add_scalar('validation_loss', valid_loss, epoch)
        if best_loss > valid_loss:
            best_loss = valid_loss
            best_state = model.state_dict()
        if early_stop:
            if len(prev_valid_losses) < 10:
                prev_valid_losses.append(valid_loss)
            else:
                loss_mean = np.mean(prev_valid_losses)
                loss_std = np.std(prev_valid_losses)
                if valid_loss > loss_mean + loss_std - eps:
                    print("Early Stopped.")
                    break
                prev_valid_losses = prev_valid_losses[1:] + [valid_loss]
    if save_path != "":
        torch.save(best_state, save_path)


def test(model, test_loader, device):
    model.eval()
    total_num, acc_num = 0, 0
    with torch.no_grad():
        for input1, input2, targets in test_loader:
            total_num += input1.shape[0]
            input1 = input1.to(device)
            input2 = input2.to(device)
            predicts = model.predict(input1, input2)
            predicts = torch.tensor(predicts)
            acc_num += torch.sum(torch.all(targets.eq(predicts), 1))
        accuracy = acc_num / total_num
    show_str = "Accuracy on test set: {:.2f}"
    print(show_str.format(accuracy))
