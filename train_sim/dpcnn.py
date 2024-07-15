# -*-coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import math
import random
from tqdm import tqdm
import torch.optim as optim
from dataset.sim.dataset import sim_dataset
from torch.autograd import Variable
from torch.nn import TransformerEncoder
from torch.utils.tensorboard import SummaryWriter


# 超参数
class Config(object):
    max_seq_len = 128
    embed_size = 256
    kernel_num = 250
    output_size = 2
    dropout_p = 0.5


input_dim = 8684
hidden_dim = 256
output_dim = 2
max_len = 128
dim_ffn = 512
dropout = 0.1

train_batch = 32
test_batch = 64
epochs = 1000
lr = 1e-4

pretrain = False
seed = 2023


class AWP_fast:
    def __init__(self, model, optimizer, *, adv_param='weight',
                 adv_lr=0.001, adv_eps=0.001):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}

    def perturb(self):
        """
        Perturb model parameters for AWP gradient
        Call before loss and loss.backward()
        """
        self._save()  # save model parameters
        self._attack_step()  # perturb weights

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                grad = self.optimizer.state[param]['exp_avg']
                if grad.type() != 'torch.cuda.CharTensor':
                    norm_grad = torch.norm(grad)
                    norm_data = torch.norm(param.detach())

                    if norm_grad != 0 and not torch.isnan(norm_grad):
                        # Set lower and upper limit in change
                        limit_eps = self.adv_eps * param.detach().abs()
                        param_min = param.data - limit_eps
                        param_max = param.data + limit_eps

                        # Perturb along gradient
                        # w += (adv_lr * |w| / |grad|) * grad
                        param.data.add_(grad, alpha=(self.adv_lr * (norm_data + e) / (norm_grad + e)))
                        # Apply the limit to the change
                        param.data.clamp_(param_min, param_max)

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.clone().detach()
                else:
                    self.backup[name].copy_(param.data)

    def restore(self):
        """
        Restore model parameter to correct position; AWP do not perturbe weights, it perturb gradients
        Call after loss.backward(), before optimizer.step()
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])


class ResnetBlock(nn.Module):
    def __init__(self, config: Config):
        super(ResnetBlock, self).__init__()
        # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html#torch.nn.MaxPool1d
        self.half_max_pooling = nn.Sequential(
            nn.ConstantPad1d(padding=(0, 1), value=0),  # 用0填充
            nn.MaxPool1d(kernel_size=3, stride=2)  # 1/2池化
        )

        # 两个等长卷积层：步长=1，卷积核大小=k,两端补0数量p为(k-1)/2时，卷积后序列长度不变
        # 卷积核大小k=3,因此p=1
        self.equal_width_conv = nn.Sequential(
            nn.BatchNorm1d(num_features=config.kernel_num),
            nn.ReLU(),
            # padding-->卷积
            nn.Conv1d(config.kernel_num, config.kernel_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=config.kernel_num),
            nn.ReLU(),
            nn.Conv1d(config.kernel_num, config.kernel_num, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        half_pooling_x = self.half_max_pooling(x)
        conv_x = self.equal_width_conv(half_pooling_x)
        final_x = half_pooling_x + conv_x
        return final_x


class DPCNN(nn.Module):
    def __init__(self, vocab_size: int, config: Config, embedding_pretrained=None):
        super(DPCNN, self).__init__()
        # 定义各种网络层
        # Embedding
        if not embedding_pretrained:
            self.embedding = nn.Embedding(vocab_size, config.embed_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(embeddings=embedding_pretrained, freeze=False)
        self.embedding.weight.requires_grad = True

        # region embedding
        self.region_embedding = nn.Sequential(
            nn.Conv1d(config.embed_size, config.kernel_num, kernel_size=3, stride=1),
            # BatchNormalization
            nn.ReLU(),
            nn.Dropout(p=config.dropout_p)
        )

        # 两个等长卷积层：步长=1，卷积核大小=k,两端补0数量p为(k-1)/2时，卷积后序列长度不变
        # 卷积核大小k=3,因此p=1
        self.equal_width_conv = nn.Sequential(
            nn.BatchNorm1d(num_features=config.kernel_num),
            nn.ReLU(),
            # padding-->卷积
            nn.Conv1d(config.kernel_num, config.kernel_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=config.kernel_num),
            nn.ReLU(),
            nn.Conv1d(config.kernel_num, config.kernel_num, kernel_size=3, stride=1, padding=1)
        )

        # ResNet_Block
        self.resnet_block = ResnetBlock(config)
        self.fc = nn.Linear(config.kernel_num, config.output_size)

    def forward(self, x):
        # x.shape: (batch_size, max_seq_len)
        x = self.embedding(x)  # x.shape: (batch_size, max_seq_len, embed_size)
        x = x.permute(0, 2, 1)  # x.shape: (batch_size, embed_size, max_seq_len)
        x = self.region_embedding(x)  # x.shape: (batch_size, 250, max_seq_len-3+1)
        x = self.equal_width_conv(x)  # x.shape: (batch_size, 250, max_seq_len-3+1)
        while x.size()[2] > 2:  # 当序列长度大于2时，一直迭代
            x = self.resnet_block(x)
        # x.shape: (batch_size, 250, 1)
        x = x.squeeze()  # x.shape: (batch_size, 250)
        # output = self.fc(x)#output.shape: (batch_size, 10)
        # output_p = F.softmax(output, dim=1)
        return x


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class custom_loss(nn.Module):
    def __init__(self, device, temperature=0.7):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, x, target):
        x = x.reshape(-1, 2, 2, x.shape[-1])
        batch_size = x.shape[0]

        question_feature = x[:, 0, 0, :]
        answer1_feature = x[:, 0, 1, :]
        answer2_feature = x[:, 1, 1, :]

        # 归一化
        question_feature = F.normalize(question_feature, dim=1)
        anchor_feature = F.normalize(answer1_feature, dim=1)
        contrast_feature = F.normalize(answer2_feature, dim=1)

        mask = torch.eye(batch_size).to(self.device)

        qa = torch.sum(mask * torch.matmul(question_feature, anchor_feature.transpose(1, 0)), dim=-1)
        qa = torch.div(qa, self.temperature)

        qc = torch.sum(mask * torch.matmul(question_feature, contrast_feature.transpose(1, 0)), dim=-1)
        qc = torch.div(qc, self.temperature)

        negated_tensor = torch.logical_not(target).float()

        up = (negated_tensor[0] * torch.exp(qa)) + (negated_tensor[1] * torch.exp(qc))
        epsilon = 1e-8
        clamped_up = torch.clamp(up, epsilon, float('inf'))

        # print(up)
        down = (target[0] * torch.exp(qa)) + (target[1] * torch.exp(qc))
        clamped_down = torch.clamp(down, epsilon, float('inf'))
        # print(down)
        loss = torch.log(clamped_up) - torch.log(clamped_down)

        # calculate correct
        correct = torch.gt(qa, qc) + 0

        return loss.mean(), torch.eq(correct, target[0])


def trans_data(path):
    with open(path, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    vocab2id = {}
    id2vocab = {}
    for i, line in enumerate(lines):
        vocab2id[line] = i
        id2vocab[i] = line
    return vocab2id, id2vocab


def make_data(vocab_path, data, max_len):
    vocab2id, id2vocab = trans_data(vocab_path)

    output_data = [[] for _ in range(len(data[0]))]
    for i in range(len(data[0])):

        prompt_text = data[0][i].split(' ')

        if len(prompt_text) > 5:
            random_prompt = random.sample(range(len(prompt_text)), k=1)
            prompt_text[random_prompt[0]] = 'unk'  # 随机替换为不认识的字
        prompt_text = [vocab2id[word] if word in vocab2id else vocab2id['unk'] for word in prompt_text]

        answer1_text = data[1][i].split(' ')
        if len(answer1_text) > 5:
            random_answer1 = random.sample(range(len(answer1_text)), k=1)
            answer1_text[random_answer1[0]] = 'unk'  # 随机替换为不认识的字
        answer1_text = [vocab2id[word] if word in vocab2id else vocab2id['unk'] for word in answer1_text]

        answer2_text = data[3][i].split(' ')
        if len(answer2_text) > 5:
            random_answer2 = random.sample(range(len(answer2_text)), k=1)
            answer2_text[random_answer2[0]] = 'unk'  # 随机替换为不认识的字
        answer2_text = [vocab2id[word] if word in vocab2id else vocab2id['unk'] for word in answer2_text]
        output_data[i] = [prompt_text, answer1_text, prompt_text, answer2_text]

    # 零值填充或者截断
    output = []
    for dt in output_data:
        d_ls = []
        for d in dt:
            if len(d) <= max_len:
                d_ls.append(d + ([0] * (max_len - len(d))))
            else:
                d_ls.append(d[:max_len])
        output.append(d_ls)

    return torch.tensor(output)


def main(pretrain, seed, train_batch, test_batch, lr, epochs):
    # log日志文件
    writer = SummaryWriter('tensorboard')  # 可视化训练过程

    # 数据路径
    train_data_path = "/root/competition/dataset/sim/train_data.csv"
    test_data_path = "/root/competition/dataset/sim/test_data.csv"
    vocab_path = "/root/competition/data/vocab.txt"

    train_dataset = sim_dataset(train_data_path)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch, shuffle=True, num_workers=4)
    print('Total number of train data:', len(train_dataset))

    test_dataset = sim_dataset(test_data_path)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch, shuffle=False)
    print('Total number of test data:', len(test_dataset))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(seed)

    # model = TransformerClassifier(input_dim, hidden_dim, device, num_layers, num_heads, dropout, max_len)
    model = DPCNN(input_dim, Config)
    model.to(device)

    print(model)
    loss_fn = custom_loss(device)
    loss_fn.to(device)

    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[175, 250, 375], gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5, last_epoch=-1)

    attack_func = AWP_fast(model, optimizer, adv_lr=0.001, adv_eps=0.001)

    # print(model)
    accuracy = 0

    for epoch in range(epochs):
        model, f1score, train_loss = train(model, optimizer, train_dataloader, epoch, device, loss_fn, scheduler,
                                           attack_func, vocab_path)
        correct, test_loss = test(model, test_dataloader, device, loss_fn, vocab_path)

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', f1score, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', correct, epoch)

        save_name = f"sim_dpcnn_{epoch}.pth"
        # accuracy = correct
        torch.save(model.state_dict(), f'saved_model/{save_name}')
    writer.close()


def train(model, optimizer, train_loader, epoch, device, loss_fn, scheduler, attack_func, vocab_path):
    model.train()
    train_loss = 0
    correct = 0

    for data, target in tqdm(train_loader, desc=f"epoch_{epoch}"):
        data = make_data(vocab_path, data, max_len)
        target = torch.stack(target)

        data = data.to(device)
        target = target.to(device)

        attack_func.perturb()

        optimizer.zero_grad()  # 优化器梯度初始化为零

        data = data.reshape(-1, data.shape[-1])
        output = model(data)  # 把数据输入网络并得到输出，即进行前向传播

        loss, correct_batch = loss_fn(output, target)  # 交叉熵损失函数
        correct += torch.sum(correct_batch)

        train_loss += loss.item() * data.shape[0]  # 计算训练误差
        loss.backward(retain_graph=True)  # 反向传播梯度
        attack_func.restore()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        optimizer.step()  # 结束一次前传+反传之后，更新参数

    scheduler.step()

    print("\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
        train_loss / len(train_loader.dataset), correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)
    ))
    return model, correct / len(train_loader.dataset), train_loss / len(train_loader.dataset)


def test(model, loader, device, loss_fn, vocab_path):
    model.eval()  # 设置为test模式
    test_loss = 0  # 初始化测试损失值为0
    correct = 0  # 初始化预测正确的数据个数为0

    for data, target in loader:
        data = make_data(vocab_path, data, max_len)
        target = torch.stack(target)

        data = data.to(device)
        target = target.to(device)

        data = data.reshape(-1, data.shape[-1])
        with torch.no_grad():
            output = model(data)

        loss, correct_batch = loss_fn(output, target)  # sum up batch loss 把所有loss值进行累加
        test_loss += loss.item() * data.shape[0]

        correct += torch.sum(correct_batch)

    test_loss /= len(loader.dataset)  # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))
    return correct, test_loss


if __name__ == '__main__':
    main(pretrain=pretrain, seed=seed, train_batch=train_batch, test_batch=test_batch, lr=lr, epochs=epochs)
