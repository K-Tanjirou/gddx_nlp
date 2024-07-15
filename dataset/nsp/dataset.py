import random
import re
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
from nltk.tokenize import WordPunctTokenizer


class nsp_dataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = pd.read_csv(data_path, sep='\t')

        data_length = self.data[['prompt', 'answer1', 'answer2']].applymap(len)
        # print(data_length.describe())

    def __getitem__(self, index):
        return self.data.loc[index, ['prompt', 'answer1', 'answer2']].values.tolist(), \
               self.data.loc[index, ['target']].values.tolist()

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


def make_data(vocab_path, data, target, max_len):
    vocab2id, id2vocab = trans_data(vocab_path)

    output_data = []
    output_target = []
    segment_data = []

    for i in range(len(data[0])):

        if i == 0:
            rand_idx = random.sample((list(range(1, len(data[0])))), k=3)
        else:
            rand_idx = random.sample(list(range(0, i)) + list(range(i+1, len(data[0]))), k=3)

        prompt_text = data[0][i].split(' ')
        prompt_text = [vocab2id[word] if word in vocab2id else vocab2id['unk'] for word in prompt_text]
        prompt_text_seg = [0] * len(prompt_text)

        answer1_text = data[1][i].split(' ')
        answer1_text = [vocab2id[word] if word in vocab2id else vocab2id['unk'] for word in answer1_text]
        answer1_text_seg = [1] * len(answer1_text)

        answer2_text = data[2][i].split(' ')
        answer2_text = [vocab2id[word] if word in vocab2id else vocab2id['unk'] for word in answer2_text]
        answer2_text_seg = [2] * len(answer2_text)

        output_text = [vocab2id['cls']] + prompt_text + [vocab2id['sep']] + answer1_text + \
                      [vocab2id['sep']] + answer2_text + [vocab2id['sep']]
        output_segment = [0] + prompt_text_seg + [0] + answer1_text_seg + [1] + answer2_text_seg + [2]

        output_data.append(output_text)
        output_target.append(target[0, i])
        segment_data.append(output_segment)

        for idx in rand_idx:
            if target[0, i] == 1:
                answer2_text = data[2][idx].split(' ')
                answer2_text = [vocab2id[word] if word in vocab2id else vocab2id['unk'] for word in answer2_text]
                answer2_text_seg = [2] * len(answer2_text)
                output_text = [vocab2id['cls']] + prompt_text + [vocab2id['sep']] + answer1_text + \
                              [vocab2id['sep']] + answer2_text + [vocab2id['sep']]
                output_segment = [0] + prompt_text_seg + [0] + answer1_text_seg + [1] + answer2_text_seg + [2]
            else:
                answer1_text = data[2][idx].split(' ')
                answer1_text = [vocab2id[word] if word in vocab2id else vocab2id['unk'] for word in answer1_text]
                answer1_text_seg = [1] * len(answer1_text)
                output_text = [vocab2id['cls']] + prompt_text + [vocab2id['sep']] + answer1_text + \
                              [vocab2id['sep']] + answer2_text + [vocab2id['sep']]
                output_segment = [0] + prompt_text_seg + [0] + answer1_text_seg + [1] + answer2_text_seg + [2]
            output_data.append(output_text)
            output_target.append(target[0, i])
            segment_data.append(output_segment)

    # 零值填充或者截断
    input_ids = []
    segment_ids = []
    for dt in output_data:
        if len(dt) <= max_len:
            input_ids.append(dt + ([0] * (max_len - len(dt))))
        else:
            input_ids.append(dt[:max_len])

    for st in segment_data:
        if len(st) <= max_len:
            segment_ids.append(st + ([0] * (max_len - len(st))))
        else:
            segment_ids.append(st[:max_len])

    input_ids = torch.tensor(input_ids, dtype=torch.float)
    segment_ids = torch.tensor(segment_ids, dtype=torch.float)
    mask_ids = (input_ids != 0).float()

    return input_ids, segment_ids, mask_ids, torch.tensor(output_target, dtype=torch.float)


if __name__ == '__main__':
    train_path = '/root/competition/dataset/nsp/train_data.csv'
    test_path = '/root/competition/dataset/nsp/test_data.csv'
    vocab_path = "/root/competition/data/vocab.txt"

    train_dataset = nsp_dataset(train_path)
    test_dataset = nsp_dataset(test_path)

    vocab2id, id2vocab = trans_data(vocab_path)
    # print(vocab2id)

    print(train_dataset[490])
    prompt, answer1, answer2 = train_dataset[490][0]

    # --------------------------- dataloader -------------------------------
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    data, target = next(iter(train_dataloader))
    target = torch.stack(target)

    print(target.shape)
    input_ids, segment_ids, mask_ids, target = make_data(vocab_path, data, target, 256)
    # target = torch.stack(target).float()

    print(input_ids)
    print(segment_ids)
    print(mask_ids)
    print(target)

