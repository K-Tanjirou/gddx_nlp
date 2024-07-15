import re
import pandas as pd
from nltk.tokenize import WordPunctTokenizer
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.linalg import norm
import numpy as np


data = pd.read_csv("/root/competition/data/test.csv", sep='\t')

data_columns = ['prompt', 'answer1', 'answer2']


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


def clean_text(text):
    # 使用正则表达式替换制表符和标点符号为空格
    cleaned_text = re.sub(r'[-…⌃●•‘’「 」\\“”\n\t,;:!：！\._，。/、?<>《》@#$%\^&\*（）+=——【】？；·\|\'"()\[\]{}]', ' ', text)
    return cleaned_text


def separate_chinese_english(text):
    # 使用正则表达式分离中文和英文
    chinese_text = re.sub(r'[^\u4e00-\u9fa5]', '', text)  # 仅保留中文字符
    english_text = re.sub(r'[\u4e00-\u9fa5]', ' ', text)  # 将中文字符替换为空格

    return chinese_text, english_text


def reduce_spaces(text):
    # 使用正则表达式将多个空格替换为一个空格
    reduced_text = re.sub(r'\s+', ' ', text)
    return reduced_text


def sep_data(data):
    out = ''
    for word in data:
        if re.match('[^a-z0-9]', word):
            out = out + ' ' + word + ' '
        else:
            out += word

    reduced_text = re.sub(r'\s+', ' ', out)
    return reduced_text.strip()


data['tfidf'] = data.apply(lambda row: tfidf_similarity(row['answer1'], row['answer2']), axis=1)


# all_text_ls = []
for columns in data_columns:
    # 清理符号
    data[columns] = data[columns].apply(lambda x: 'unk' if x == '...' else x)
    data[columns] = data[columns].apply(lambda x: clean_text(x))
    data[columns] = data[columns].apply(lambda x: reduce_spaces(x))
    data[columns] = data[columns].apply(lambda x: sep_data(x.lower()))
    # all_text_ls += data[columns].values.tolist()

data = data.fillna('unk')
data.to_csv('/root/competition/data/test_process.csv', sep='\t', index=False)

# all_text = reduce_spaces(' '.join(all_text_ls))
# chinese_text, english_text = separate_chinese_english(all_text)
# english_text = reduce_spaces(english_text).lower()
#
# tokenizer = WordPunctTokenizer()
# sep_english = list(set(tokenizer.tokenize(english_text)))
#
# sep_chinese = list(set(chinese_text))
#
# vocab = sep_chinese + sep_english
# vocab += list(range(10))
# vocab = ['pad', 'unk', 'mask', 'sep', 'cls'] + vocab
# print(vocab)

# with open('vocab.txt', 'w', encoding='UTF-8') as f:
#     for word in vocab:
#         f.write(str(word))
#         f.write('\n')
#
# with open('vocab.txt', 'r', encoding='UTF-8') as f:
#     lines = f.readlines()
#     lines = [line.strip() for line in lines]

# print(lines)

# vocab2id = {}
# id2vocab = {}
# for i, line in enumerate(lines):
#     print(i, line)
#     vocab2id[line] = i
#     vocab2id[i] = line
#
# print(vocab2id)

# data_process = pd.read_csv('/root/competition/data/train_process.csv', sep='\t')
# print(data_process.loc[:, ['prompt', 'answer1', 'answer2']])
