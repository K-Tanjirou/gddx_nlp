import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertConfig
import pandas as pd
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.linalg import norm


# 超参数
custom_config = BertConfig(
    vocab_size=8684,  # 词汇表大小，通常为30522
    type_vocab_size=3,
    hidden_size=256,   # 隐藏层大小，通常为768
    num_hidden_layers=2,  # 隐藏层的数量，通常为12
    num_attention_heads=4,  # 注意力头的数量，通常为12
    intermediate_size=1024,  # Feedforward 层的中间大小，通常为3072
    hidden_dropout_prob=0.2,  # 隐藏层的 dropout 概率，通常为0.1
    attention_probs_dropout_prob=0.2,  # 注意力机制的 dropout 概率，通常为0.1
    layer_norm_eps=1e-7,  # 增加 Layer Normalization 的 epsilon
)

max_len = 256

train_batch = 32
test_batch = 64
epochs = 1000
lr = 1e-4


class test_dataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = pd.read_csv(data_path, sep='\t')

        # data_length = self.data[['prompt', 'answer1', 'answer2']].applymap(len)
        # print(data_length.describe())

    def __getitem__(self, index):
        return self.data.loc[index, ['prompt', 'answer1', 'answer2']].values.tolist()

    def __len__(self):
        return len(self.data)


class CustomBERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CustomBERTClassifier, self).__init__()
        self.bert = BertModel(config=custom_config)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(custom_config.hidden_size, num_classes)

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        # 在这里可以添加自定义的权重初始化逻辑
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits


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

    output_data = []
    segment_data = []

    for i in range(len(data[0])):

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

    input_ids = torch.tensor(input_ids)
    segment_ids = torch.tensor(segment_ids)
    mask_ids = (input_ids != 0).float()

    return input_ids, segment_ids, mask_ids


def calculate(x, device, temperature):
    x = x.reshape(-1, 2, 2, x.shape[-1])
    batch_size = x.shape[0]

    question_feature = x[:, 0, 0, :]
    answer1_feature = x[:, 0, 1, :]
    answer2_feature = x[:, 1, 1, :]

    # 归一化
    question_feature = F.normalize(question_feature, dim=1)
    anchor_feature = F.normalize(answer1_feature, dim=1)
    contrast_feature = F.normalize(answer2_feature, dim=1)

    mask = torch.eye(batch_size).to(device)

    qa = torch.sum(mask * torch.matmul(question_feature, anchor_feature.transpose(1, 0)), dim=-1)
    qa = torch.div(qa, temperature)

    qc = torch.sum(mask * torch.matmul(question_feature, contrast_feature.transpose(1, 0)), dim=-1)
    qc = torch.div(qc, temperature)

    outputs = torch.stack([qa, qc], dim=1)

    correct = torch.gt(qa, qc) + 0
    return correct, F.softmax(outputs, dim=1).detach().cpu()


def tfidf_similarity(s1, s2):
    def add_space(s):
        # 分词，词语间使用空格分隔
        return ' '.join(jieba.lcut(s))

    # 将词语中间加入空格，方便转为矩阵
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为矩阵
    cv = TfidfVectorizer(tokenizer=lambda s: s.split(' '))
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 计算TF-IDF系数
    sim = np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))
    return int(sim >= 0.6)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = '/root/competition/data/test_process.csv'
    vocab_path = "/root/competition/data/vocab.txt"
    submission_path = "/root/competition/data/sample_submission.csv"
    model_path = "/hy-tmp/nsp/sim_bert_89.pth"

    # model = TextCNN(num_classes=hidden_dim, num_embeddings=input_dim)
    model = CustomBERTClassifier(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)

    testdataset = test_dataset(data_path)
    test_dataloader = DataLoader(testdataset, batch_size=test_batch, shuffle=False)

    preds = []
    prob_all = []
    for data in test_dataloader:
        input_ids, segment_ids, mask_ids, = make_data(vocab_path, data, max_len)

        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        mask_ids = mask_ids.to(device)

        output = model(input_ids, mask_ids, segment_ids)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        prob = F.softmax(output, dim=1).detach().cpu()
        print(prob)
        prob_all.append(prob)
        preds += pred.flatten().tolist()
    prob_all = torch.cat(prob_all).numpy()

    test_data = pd.read_csv(data_path, sep='\t')
    # test_data['pred'] = preds
    print(preds)

    # test_data[['prob1', 'prob2']] = prob_all
    # print(test_data)

    # test_data['pred'] = test_data['prob2'].apply(lambda x: 0 if x > 0.62 else 1)
    test_data['pred'] = preds

    test_data['target'] = test_data.apply(lambda row: row['tfidf'] if row['tfidf'] == 1 else row['pred'], axis=1)
    sub_data = pd.read_csv(submission_path)
    sub_data.loc[:, 'answer1_target'] = test_data['target'].values.tolist()
    sub_data['answer1_target'] = sub_data['answer1_target'].apply(lambda x: 'chosen' if x == 1 else 'rejected')
    print(sub_data)
    sub_data.to_csv('BERT_submission.csv', index=False)


