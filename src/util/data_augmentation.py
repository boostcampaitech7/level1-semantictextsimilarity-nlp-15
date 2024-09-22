from koeda import EDA
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

def koeda(df):
    eda = EDA(morpheme_analyzer="Okt", alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, prob_rd=0.1)

    copy_df = df.copy()
    sentence_1 = copy_df['sentence_1'].tolist()

    # translate each sequence using list due to koeda does not support type pandas series
    for i, each_sentence in enumerate(sentence_1):
        eda_sentence = eda(each_sentence)

        # remove special characters by using regex
        each_sentence_remove = re.sub('[^a-zA-Z가-힣]', ' ', str(each_sentence))
        eda_sentence_remove = re.sub('[^a-zA-Z가-힣]', ' ', str(eda_sentence))

        if each_sentence_remove != eda_sentence_remove:
            copy_df.loc[i, 'sentence_1'] = eda_sentence

    df = pd.concat([df, copy_df], ignore_index=True)
    df = df.drop_duplicates(subset=['sentence_1'], keep='first')

    return df

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