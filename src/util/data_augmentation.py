from koeda import EDA
from torch.utils.data import TensorDataset
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

    # translate each sequence using list due to koeda does not support type pandas series
    # for i, each_sentence in enumerate(sentence_1):
    #     eda_sentence = eda(each_sentence)
    #
    #     # remove special characters by using regex
    #     each_sentence_remove = re.sub('[^a-zA-Z가-힣]', ' ', str(each_sentence))
    #     eda_sentence_remove = re.sub('[^a-zA-Z가-힣]', ' ', str(eda_sentence))
    #
    #     if each_sentence_remove != eda_sentence_remove:
    #         copy_df.loc[i, 'sentence_1'] = eda_sentence
    #         diff_list.append([each_sentence, eda_sentence])

    copy_df['sentence_1'] = res

    df = pd.concat([df, copy_df], ignore_index=True)
    df = df.drop_duplicates(subset=['sentence_1'], keep='first')

    # Drop any row that has empty columns
    df = df.dropna()

    return df if not add else diff_list

# import pandas as pd
#
# def line_swap(path):
#     df = pd.read_csv(path)
#     swap_df = pd.read_csv(path)
#
#     swap_df['sentence_1'], swap_df['sentence_2'] = swap_df['sentence_2'], swap_df['sentence_1']
#
#     df = pd.concat([df, swap_df])
#     return df
#
# if __name__ == '__main__':
#     df = line_swap('../data/train.csv')
#
#     print(df)
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