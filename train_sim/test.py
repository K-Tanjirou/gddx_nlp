import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertModel, BertConfig
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.linalg import norm


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


class test_dataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = pd.read_csv(data_path, sep='\t')

        # data_length = self.data[['prompt', 'answer1', 'answer2']].applymap(len)
        # print(data_length.describe())

    def __getitem__(self, index):
        return self.data.loc[index, ['prompt', 'answer1', 'prompt', 'answer2']].values.tolist()

    def __len__(self):
        return len(self.data)


class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        return F.max_pool1d(x, kernel_size=x.shape[2])  # shape: (batch_size, channel, 1)


class TextCNN(nn.Module):
    def __init__(self, num_classes, num_embeddings=-1, embedding_dim=128, kernel_sizes=[3, 4, 5, 6],
                 num_channels=[256, 256, 256, 256], embeddings_pretrained=None):
        """
        :param num_classes: 输出维度(类别数num_classes)
        :param num_embeddings: size of the dictionary of embeddings,词典的大小(vocab_size),
                               当num_embeddings<0,模型会去除embedding层
        :param embedding_dim:  the size of each embedding vector，词向量特征长度
        :param kernel_sizes: CNN层卷积核大小
        :param num_channels: CNN层卷积核通道数
        :param embeddings_pretrained: embeddings pretrained参数，默认None
        :return:
        """
        super(TextCNN, self).__init__()
        self.num_classes = num_classes
        self.num_embeddings = num_embeddings
        # embedding层
        if self.num_embeddings > 0:
            # embedding之后的shape: torch.Size([200, 8, 300])
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            if embeddings_pretrained is not None:
                self.embedding = self.embedding.from_pretrained(embeddings_pretrained, freeze=False)
        # 卷积层
        self.cnn_layers = nn.ModuleList()  # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            cnn = nn.Sequential(
                nn.Conv1d(in_channels=embedding_dim,
                          out_channels=c,
                          kernel_size=k),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
            )
            self.cnn_layers.append(cnn)
        # 最大池化层
        self.pool = GlobalMaxPool1d()
        # 输出层
        self.classify = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(sum(num_channels), self.num_classes)
        )

    def forward(self, input):
        """
        :param input:  (batch_size, context_size, embedding_size(in_channels))
        :return:
        """
        if self.num_embeddings > 0:
            # 得到词嵌入(b,context_size)-->(b,context_size,embedding_dim)
            input = self.embedding(input)
            # (batch_size, context_size, channel)->(batch_size, channel, context_size)
        input = input.permute(0, 2, 1)
        y = []
        for layer in self.cnn_layers:
            x = layer(input)
            x = self.pool(x).squeeze(-1)
            y.append(x)
        y = torch.cat(y, dim=1)
        out = self.classify(y)
        return out


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
        prompt_text = [vocab2id[word] if word in vocab2id else vocab2id['unk'] for word in prompt_text]

        answer1_text = data[1][i].split(' ')
        answer1_text = [vocab2id[word] if word in vocab2id else vocab2id['unk'] for word in answer1_text]

        answer2_text = data[3][i].split(' ')
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
    model_path = "/hy-tmp/sim/sim_textcnn_96.pth"

    model = CustomBERTClassifier()
    model.load_state_dict(torch.load('saved_model/sim_transformer_75.pth', map_location='cpu'))
    # model = TextCNN(num_classes=hidden_dim, num_embeddings=input_dim)
    # model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)

    testdataset = test_dataset(data_path)
    test_dataloader = DataLoader(testdataset, batch_size=test_batch, shuffle=False)

    preds = []
    prob_all = []
    for data in test_dataloader:
        data, mask = make_data(vocab_path, data, max_len)
        data = data.to(device)
        mask = mask.to(device)

        data = data.reshape(-1, data.shape[-1])
        mask = mask.reshape(-1, mask.shape[-1])
        output = model(data, mask)
        pred, prob = calculate(output, device, 0.7)
        prob_all.append(prob)
        preds += pred.tolist()
    prob_all = torch.cat(prob_all).numpy()

    test_data = pd.read_csv(data_path, sep='\t')
    # test_data['pred'] = preds
    print(preds)
    print(test_data)

    test_data['pred'] = preds

    test_data['target'] = test_data.apply(lambda row: row['tfidf'] if row['tfidf'] == 1 else row['pred'], axis=1)
    sub_data = pd.read_csv(submission_path)
    sub_data[['prob1', 'prob2']] = prob_all
    sub_data.loc[:, 'answer1_target'] = test_data['target'].values.tolist()
    sub_data['answer1_target'] = sub_data['answer1_target'].apply(lambda x: 'chosen' if x == 1 else 'rejected')
    print(sub_data)
    sub_data.to_csv('submission.csv', index=False)


