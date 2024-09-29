import requests
from koeda import EDA
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from sklearn.utils import shuffle
import torch
import pandas as pd
import re

def swap_sentences(df):
    swapped_df = df.copy()
    swapped_df['sentence_1'], swapped_df['sentence_2'] = df['sentence_2'], df['sentence_1']
    return pd.concat([df, swapped_df], ignore_index=True)

def remove_special_characters(df):
    df['sentence_1'] = df['sentence_1'].str.replace('[^a-zA-Z0-9가-힣]', ' ', regex=True)
    df['sentence_2'] = df['sentence_2'].str.replace('[^a-zA-Z0-9가-힣]', ' ', regex=True)
    return df

def koeda(df, add=False):
    eda = EDA(morpheme_analyzer="Okt", alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, prob_rd=0.1)

    copy_df = df.copy()
    diff_list = []
    sentence_1 = copy_df['sentence_1'].tolist()

    res = eda(sentence_1)

    copy_df['sentence_1'] = res

    df = pd.concat([df, copy_df], ignore_index=True)
    df = df.drop_duplicates(subset=['sentence_1'], keep='first')

    # Drop any row that has empty columns
    df = df.dropna()

    return df if not add else diff_list

def copy_sentence(df, index_min=250, index_max=750) -> pd.DataFrame:
    df_copied = df[df["label"] == 0][index_min:index_max].copy()
    df_copied["sentence_1"] = df_copied["sentence_2"]  # sentence 2를 sentence 1으로 복사
    df_copied["label"] = 5.0  # 라벨 5로 설정

    df_copied = df_copied.drop_duplicates(subset=['sentence_1'], keep='first')

    return df_copied

def under_sampling(df) -> pd.DataFrame:
    """
    label 값이 0인 데이터를 under sampling하는 함수
        Args:
            data_path (str): 증강하고자 하는 데이터의 경로
        Returns:
            df_new (DataFrame): under sampling된 데이터
    """

    df_0 = df[df["label"] == 0][1000:2000].copy()  # 라벨 0의 일부 데이터 선택
    df_new = df[df["label"] != 0].copy()  # 라벨 0이 아닌 데이터 선택
    df_new = pd.concat([df_new, df_0])  # 라벨 0 데이터와 결합
    return df_new

def remove_stopwords(df):
    # Load stopwords list
    def sub_func(text):
        url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ko/master/stopwords-ko.txt"
        response = requests.get(url)
        stop_words = set(response.text.splitlines())
        # print(stop_words)

        return ' '.join([word for word in text.split() if word not in stop_words])

    df['sentence_1'] = df['sentence_1'].apply(sub_func)
    df['sentence_2'] = df['sentence_2'].apply(sub_func)

    return df

def clean_text(df):
    # Load stopwords list
    def sub_func(sentence):
        sentence = re.sub(r'<[^>]+>', '', sentence)
        sentence = re.sub(r'[^가-힣a-zA-Z\s]', '', sentence)
        return sentence

    df['sentence_1'] = df['sentence_1'].apply(sub_func)
    df['sentence_2'] = df['sentence_2'].apply(sub_func)

    return df

def normalize_numbers(df):
    # Load stopwords list
    def sub_func(sentence):
        sentence = re.sub(r'\d+', 'NUM', sentence)
        return sentence

    df['sentence_1'] = df['sentence_1'].apply(sub_func)
    df['sentence_2'] = df['sentence_2'].apply(sub_func)

    return df


def train_val_split(train_df, dev_df, ratio=0.8):
    # Concat two df and split random in 8:2 ratio
    train_val_concat = concat_train_val(train_df, dev_df)
    train_df, dev_df = train_test_split(train_val_concat, test_size=1-ratio, stratify=train_val_concat['binary-label'],
                                        random_state=0)
    return train_df, dev_df


def concat_train_val(train_path, val_path):
    df1 = pd.read_csv(train_path)
    df2 = pd.read_csv(val_path)

    return pd.concat([df1, df2], ignore_index=True)


def k_fold_split(train_val_concat, kf, tokenizer):

    x = train_val_concat[['sentence_1', 'sentence_2']]
    y = train_val_concat['label']

    train_dataset, val_dataset = None, None

    for train_index, val_index in kf.split(x):
        x_train, x_val = x.iloc[train_index], x.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        train_token = tokenizer_pair(tokenizer, x_train['sentence_1'], x_train['sentence_2'])
        val_token = tokenizer_pair(tokenizer, x_val['sentence_1'], x_val['sentence_2'])

        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

        train_dataset = TensorDataset(train_token['input_ids'], train_token['attention_mask'], y_train_tensor)
        val_dataset = TensorDataset(val_token['input_ids'], val_token['attention_mask'], y_val_tensor)

    return train_dataset, val_dataset

def tokenizer_pair(tokenizer, s1, s2):
    return tokenizer(s1.tolist(), s2.tolist(), return_tensors='pt', padding='max_length', truncation=True, max_length=128)

if __name__ == '__main__':
    train_path = '../data/train.csv'

    df = pd.read_csv(train_path)

    ko_df = koeda(df, add=True)

    for i in range(10):
        print(ko_df[i])