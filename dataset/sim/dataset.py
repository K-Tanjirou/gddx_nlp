import random
import re
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
from nltk.tokenize import WordPunctTokenizer


class sim_dataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = pd.read_csv(data_path, sep='\t')

        data_length = self.data[['prompt', 'answer1', 'answer2']].applymap(len)
        # print(data_length.describe())

    def __getitem__(self, index):
        return self.data.loc[index, ['prompt', 'answer1', 'prompt', 'answer2']].values.tolist(), \
               self.data.loc[index, ['answer1_target', 'answer2_target']].values.tolist()

    def __len__(self):
        return len(self.data)


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


def reduce_spaces(text):
    # 使用正则表达式将多个空格替换为一个空格
    reduced_text = re.sub(r'\s+', ' ', text)
    return reduced_text


def make_data(vocab_path, data, max_len):
    vocab2id, id2vocab = trans_data(vocab_path)

    output_data = [[] for _ in range(len(data[0]))]
    for i in range(len(data[0])):

        prompt_text = data[0][i].split(' ')

        if len(prompt_text) > 5:
            random_prompt = random.sample(range(len(prompt_text)), k=1)
            prompt_text[random_prompt[0]] = 'unk'       # 随机替换为不认识的字
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


def loss(x, target, temperature):
    x = x.reshape(-1, 2, 2, x.shape[-1])
    batch_size = x.shape[0]

    question_feature = x[:, 0, 0, :]
    answer1_feature = x[:, 0, 1, :]
    answer2_feature = x[:, 1, 1, :]

    # 归一化
    question_feature = F.normalize(question_feature, dim=1)
    anchor_feature = F.normalize(answer1_feature, dim=1)
    contrast_feature = F.normalize(answer2_feature, dim=1)

    mask = torch.eye(batch_size)

    qa = torch.sum(mask * torch.matmul(question_feature, anchor_feature.transpose(1, 0)), dim=-1)
    qa = torch.div(qa, temperature)

    # ac = torch.sum(mask * torch.matmul(anchor_feature, contrast_feature.transpose(1, 0)), dim=-1)
    # ac = torch.div(ac, self.temperature)

    qc = torch.sum(mask * torch.matmul(question_feature, contrast_feature.transpose(1, 0)), dim=-1)
    qc = torch.div(qc, temperature)

    negated_tensor = torch.logical_not(target).float()

    up = (negated_tensor[0] * torch.exp(qa)) + (negated_tensor[1] * torch.exp(qc))
    down = (target[0] * torch.exp(qa)) + (target[1] * torch.exp(qc))
    loss = torch.log(up) - torch.log(down)

    # calculate correct
    correct = torch.gt(qa, qc) + 0

    return loss.mean(), torch.eq(correct, target)


if __name__ == '__main__':
    train_path = '/root/competition/dataset/sim/train_data.csv'
    test_path = '/root/competition/dataset/sim/test_data.csv'
    vocab_path = "/root/competition/data/vocab.txt"

    train_dataset = sim_dataset(train_path)
    test_dataset = sim_dataset(test_path)

    vocab2id, id2vocab = trans_data(vocab_path)
    # print(vocab2id)

    prompt, answer1, _, answer2 = train_dataset[490][0]

    # --------------------------- dataloader -------------------------------
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    data, target = next(iter(train_dataloader))

    data = make_data(vocab_path, data, 128).float()
    target = torch.stack(target).float()

    print(loss(data, target, 1))


