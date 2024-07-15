import pandas as pd
import random
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.linalg import norm
import numpy as np

train_path = "/root/competition/data/train_process.csv"


# 定义计算相似度的方法
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


# 分离词
def sep_data(data):
    out = ''
    for word in data:
        if re.match('[^a-z0-9]', word):
            out = out + ' ' + word + ' '
        else:
            out += word

    reduced_text = re.sub(r'\s+', ' ', out)
    return reduced_text.strip()


if __name__ == '__main__':
    data = pd.read_csv(train_path, sep='\t')
    data['tfidf'] = data.apply(lambda row: tfidf_similarity(row['answer1'], row['answer2']), axis=1)
    data['answer1_target'] = data['answer1_target'].apply(lambda x: 1 if x == 'chosen' else 0)
    data['answer2_target'] = data.apply(lambda row: int(row['answer1_target'] == 0)
                                        if row['tfidf'] == 0 else row['tfidf'], axis=1)
    print(len(data[data['tfidf'] == 1]))

    data['prompt'] = data['prompt'].apply(lambda x: sep_data(x.lower()))
    data['answer1'] = data['answer1'].apply(lambda x: sep_data(x.lower()))
    data['answer2'] = data['answer2'].apply(lambda x: sep_data(x.lower()))

    chosen_data = data[data['answer1_target'] == 1]
    reject_data = data[data['answer1_target'] == 0]

    # 负样本索引
    test_chosen_index = random.sample(range(len(chosen_data)), k=350)
    test_reject_index = random.sample(range(len(reject_data)), k=350)

    # 正样本索引
    train_chosen_index = list(set(range(len(chosen_data))) - set(test_chosen_index))
    train_reject_index = list(set(range(len(reject_data))) - set(test_reject_index))

    print(len(chosen_data), len(test_chosen_index), len(train_chosen_index))
    print(len(reject_data), len(test_reject_index), len(train_reject_index))

    train_data_chosen = data.loc[train_chosen_index, ['prompt', 'answer1', 'answer1_target',
                                                      'answer2', 'answer2_target', 'tfidf']]
    train_data_reject = data.loc[train_reject_index, ['prompt', 'answer1', 'answer1_target',
                                                      'answer2', 'answer2_target', 'tfidf']]

    test_data_chosen = data.loc[test_chosen_index, ['prompt', 'answer1', 'answer1_target',
                                                    'answer2', 'answer2_target', 'tfidf']]
    test_data_reject = data.loc[test_reject_index, ['prompt', 'answer1', 'answer1_target',
                                                    'answer2', 'answer2_target', 'tfidf']]

    train_data = pd.concat([train_data_chosen, train_data_reject])
    test_data = pd.concat([test_data_chosen, test_data_reject])

    print(len(train_data))
    print(len(test_data))

    train_data.to_csv('/root/competition/dataset/sim/train_data.csv', sep='\t')
    test_data.to_csv('/root/competition/dataset/sim/test_data.csv', sep='\t')
