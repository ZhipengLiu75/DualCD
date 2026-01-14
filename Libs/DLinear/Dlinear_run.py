import argparse
import time
import torch.nn as nn
import numpy as np
import torch
import os
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from Libs.utils.Continual import forgetting, average_accuracy, performance_aware_forgetting
from DLinear import DLinear
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from torch import optim
import os

RED = '\033[91m'
RESET = '\033[0m'

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', default='../../data/HAR', type=str)
parser.add_argument('--tasks', default=10, type=int)
parser.add_argument('--num_class', default=6, type=int)
parser.add_argument('--seq_len', type=int, default=128)
parser.add_argument('--enc_in', type=int, default=9)

parser.add_argument('--DIL', type=bool, default=True, help='=== If use our strategy ===')
parser.add_argument('--alpha', type=float, default=.5)
parser.add_argument('--batchsize', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epoches', default=100, type=int)
parser.add_argument('--model', type=str, default='DLinear')
parser.add_argument('--moving_avg', type=int, default=25)


class EarlyStopping:
    def __init__(self, patience=15, verbose=False, dump=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_ls = None
        if args.model == 'DLinear':
            self.best_model = DLinear(args).float().to('cuda')

        self.early_stop = False
        self.dump = dump
        self.val_loss_min = np.Inf
        self.delta = 0
        self.trace_func = print

    def __call__(self, val_loss, model, epoch):
        ls = val_loss
        if self.best_ls is None:
            self.best_ls = ls
            self.best_model.load_state_dict(model.state_dict())

        elif ls > self.best_ls + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} with {epoch}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_ls = ls
            self.best_model.load_state_dict(model.state_dict())
            self.counter = 0


class Load_Dataset(Dataset):
    def __init__(self, dataset):
        super(Load_Dataset, self).__init__()
        X_train = dataset["samples"]
        y_train = dataset["labels"]
        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train).float().to('cuda')
            self.y_data = torch.from_numpy(y_train).long().to('cuda')
        else:
            self.x_data = X_train.float().to('cuda')
            self.y_data = y_train.long().to('cuda')
        self.len = X_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator(args, task):
    train_datasets = torch.load(os.path.join(args.data_path, "train.pt"))
    val_datasets = torch.load(os.path.join(args.data_path, "val.pt"))
    test_datasets = torch.load(os.path.join(args.data_path, "test.pt"))

    train_dataset = train_datasets[task]
    val_dataset = val_datasets[task]
    test_dataset = test_datasets[task]

    train_dataset = Load_Dataset(train_dataset)
    val_dataset = Load_Dataset(val_dataset)
    test_dataset = Load_Dataset(test_dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batchsize,
                                               shuffle=True, drop_last=False,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batchsize,
                                             shuffle=False, drop_last=False,
                                             num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batchsize,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, val_loader, test_loader


def TestTest(test_loader, bestmodel, final):
    predicts = None
    labels = None
    bestmodel.eval()

    for data, label in test_loader:
        out, out2, enc_out = bestmodel(data, label)
        out = torch.argmax(out.cpu().detach(), dim=-1).reshape(-1)
        label = label.cpu().detach().reshape(-1)
        if predicts == None:
            predicts = out
            labels = label
        else:
            predicts = torch.cat([predicts, out])
            labels = torch.cat([labels, label])

    X = None
    if final == True:
        num_classes = torch.max(labels).item() + 1
        correct = torch.zeros(num_classes)
        total = torch.zeros(num_classes)
        for i in range(num_classes):
            mask = (labels == i)
            total[i] = mask.sum()
            correct[i] = (predicts[mask] == labels[mask]).sum()
        per_class_acc = correct / total.clamp(min=1)
        X = []
        for i in range(num_classes):
            X.append(per_class_acc[i].item())

    acc = accuracy_score(predicts, labels)
    printtext = "ACC:{:.4f}".format(acc)
    print(printtext)
    return acc, X


def Test(test_loader, bestmodel, final):
    predicts = None
    labels = None
    bestmodel.eval()
    for data, label in test_loader:
        out, out2, _ = bestmodel(data, label)
        out = torch.argmax(out.cpu().detach(), dim=-1).reshape(-1)
        label = label.cpu().detach().reshape(-1)
        if predicts == None:
            predicts = out
            labels = label
        else:
            predicts = torch.cat([predicts, out])
            labels = torch.cat([labels, label])
    X = None
    if final == True:
        num_classes = torch.max(labels).item() + 1
        correct = torch.zeros(num_classes)
        total = torch.zeros(num_classes)
        for i in range(num_classes):
            mask = (labels == i)
            total[i] = mask.sum()
            correct[i] = (predicts[mask] == labels[mask]).sum()
        per_class_acc = correct / total.clamp(min=1)
        X = []
        for i in range(num_classes):
            X.append(per_class_acc[i].item())

    acc = accuracy_score(predicts, labels)
    printtext = "ACC:{:.4f}".format(acc)
    print(printtext)
    return acc, X


def Trainer(args, model):
    PredLossFun = nn.CrossEntropyLoss()
    all_test_loaders = []
    accuracy_matrix = []
    XX = []
    for task in range(args.tasks):
        print('=' * 20, task + 1, '=' * 30)
        early_stopping = EarlyStopping(patience=15, verbose=True, dump=False)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(.9, .99), weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=.5, patience=3)
        train_loader, val_loader, test_loader = data_generator(args, task)
        all_test_loaders.append(test_loader)

        for epoch in range(1, args.epoches + 1):
            train_losses, val_losses, test_losses = [], [], []
            model.train()
            for data, label in train_loader:
                optimizer.zero_grad()
                outv, outc, _ = model(data, label)
                if args.DIL:
                    loss = args.alpha * PredLossFun(outv, label) + (1 - args.alpha) * PredLossFun(outc, label)
                else:
                    loss = PredLossFun(outv, label)

                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                model.eval()
                for data, label in val_loader:
                    out, out2, _ = model(data, label)
                    loss = PredLossFun(out, label)
                    val_losses.append(loss.item())

            with torch.no_grad():
                model.eval()
                for data, label in test_loader:
                    out, out2, _ = model(data, label)
                    loss = PredLossFun(out, label)
                    test_losses.append(loss.item())

            print('epoch:{0:}, train_loss:{1:.5f}, val_loss:{2:.5f}, test_loss:{3:.5f}'.format(epoch,
                                                                                               np.mean(train_losses),
                                                                                               np.mean(val_losses),
                                                                                               np.mean(test_losses)))

            scheduler.step(np.mean(val_losses))
            early_stopping(np.mean(val_losses), model, epoch)
            if early_stopping.early_stop:
                print("Early stopping with best_ls:{}".format(early_stopping.best_ls))
                break
            if np.isnan(np.mean(val_losses)) or np.isnan(np.mean(train_losses)):
                break


        accs = []
        for i, test_loader in enumerate(all_test_loaders):
            acc, xx = Test(test_loader, early_stopping.best_model, True)
            print(f"Accuracy on Domain {i + 1}: {acc:.4f}")
            accs.append(acc)
            if task == args.tasks - 1:
                XX.append(xx)
        accuracy_matrix.append(accs)


    avg_acc = average_accuracy(accuracy_matrix)
    avg_forget = forgetting(accuracy_matrix)
    avg_pa_forget = performance_aware_forgetting(accuracy_matrix)
    XX = torch.tensor(XX)
    XX = torch.mean(XX, dim=0)
    return avg_acc, avg_forget, avg_pa_forget, XX.cpu().numpy()


def main(args, i=0):
    SEED = i
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = None
    if args.model == 'DLinear':
        model = DLinear(args).float().to(args.device)
    return Trainer(args, model)


if __name__ == '__main__':
    args = parser.parse_args()

    ACC, Forget, PAForget= [], [], []

    avg_acc, avg_forget, avg_pa_forget, _ = main(args)

    print(f"\n{RED}====================================={RESET}")
    print(f"=== Data: {args.data_path[11:]} ===")
    print(f"{RED}=== Average Accuracy: {avg_acc:.4f} ==={RESET}")
    print(f"{RED}=== Average Forgetting: {avg_forget:.4f} ==={RESET}")
    print(f"{RED}=== Performance Aware Forgetting: {avg_pa_forget:.4f} ==={RESET}")

