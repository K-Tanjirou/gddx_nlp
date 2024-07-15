# -*-coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import math
import random
import copy
from tqdm import tqdm
import torch.optim as optim
from dataset.sim.dataset import sim_dataset
from transformers import BertModel, BertConfig
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.tensorboard import SummaryWriter

# 超参数
custom_config = BertConfig(
    vocab_size=8684,  # 词汇表大小，通常为30522
    hidden_size=256,   # 隐藏层大小，通常为768
    num_hidden_layers=4,  # 隐藏层的数量，通常为12
    num_attention_heads=8,  # 注意力头的数量，通常为12
    intermediate_size=1024,  # Feedforward 层的中间大小，通常为3072
    hidden_dropout_prob=0.2,  # 隐藏层的 dropout 概率，通常为0.1
    attention_probs_dropout_prob=0.2,  # 注意力机制的 dropout 概率，通常为0.1
    layer_norm_eps=1e-7,  # 增加 Layer Normalization 的 epsilon
)

max_len = 128

train_batch = 32
test_batch = 64
epochs = 1000
lr = 1e-7

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


class CustomBERTClassifier(nn.Module):
    def __init__(self):
        super(CustomBERTClassifier, self).__init__()
        self.bert = BertModel(config=custom_config)
        self.dropout = nn.Dropout(0.1)
        # self.fc = nn.Linear(custom_config.hidden_size, num_classes)

        # 初始化权重
        # self.init_weights()

    def init_weights(self):
        # 在这里可以添加自定义的权重初始化逻辑
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        # logits = self.fc(pooled_output)
        return pooled_output


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class custom_loss(nn.Module):
    def __init__(self, device, temperature=1):
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

        ac = torch.sum(mask * torch.matmul(anchor_feature, contrast_feature.transpose(1, 0)), dim=-1)
        ac = torch.div(ac, self.temperature)

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
        loss = torch.log(clamped_up + torch.exp(ac)) - torch.log(clamped_down)

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
    input_ids = torch.tensor(output)
    mask_ids = (input_ids != 0).float()
    return input_ids, mask_ids


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
    # model = TransformerModel(input_dim, hidden_dim, num_heads, dim_ffn, num_layers, dropout)
    model = CustomBERTClassifier()
    model.load_state_dict(torch.load('saved_model/sim_transformer_326.pth', map_location='cpu'))
    model.to(device)

    print(model)
    loss_fn = custom_loss(device)
    loss_fn.to(device)

    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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

        save_name = f"sim_transformer_{epoch}.pth"
        # accuracy = correct
        torch.save(model.state_dict(), f'/hy-tmp/sim/{save_name}')
    writer.close()


def train(model, optimizer, train_loader, epoch, device, loss_fn, scheduler, attack_func, vocab_path):
    model.train()
    train_loss = 0
    correct = 0

    for data, target in tqdm(train_loader, desc=f"epoch_{epoch}"):
        input_ids, mask_ids = make_data(vocab_path, data, max_len)
        target = torch.stack(target)

        input_ids = input_ids.to(device)
        mask_ids = mask_ids.to(device)
        target = target.to(device)

        attack_func.perturb()

        optimizer.zero_grad()  # 优化器梯度初始化为零

        input_ids = input_ids.reshape(-1, input_ids.shape[-1])
        mask_ids = mask_ids.reshape(-1, mask_ids.shape[-1])
        output = model(input_ids, mask_ids)  # 把数据输入网络并得到输出，即进行前向传播

        loss, correct_batch = loss_fn(output, target)  # 交叉熵损失函数
        correct += torch.sum(correct_batch)

        train_loss += loss.item() * input_ids.shape[0]  # 计算训练误差
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
        input_ids, mask_ids = make_data(vocab_path, data, max_len)
        target = torch.stack(target)

        input_ids = input_ids.to(device)
        mask_ids = mask_ids.to(device)
        target = target.to(device)

        input_ids = input_ids.reshape(-1, input_ids.shape[-1])
        mask_ids = mask_ids.reshape(-1, mask_ids.shape[-1])
        with torch.no_grad():
            output = model(input_ids, mask_ids)

        loss, correct_batch = loss_fn(output, target)  # sum up batch loss 把所有loss值进行累加
        test_loss += loss.item() * input_ids.shape[0]

        correct += torch.sum(correct_batch)

    test_loss /= len(loader.dataset)  # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))
    return correct, test_loss


if __name__ == '__main__':
    main(pretrain=pretrain, seed=seed, train_batch=train_batch, test_batch=test_batch, lr=lr, epochs=epochs)