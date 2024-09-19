import pandas as pd

def line_swap(path):
    df = pd.read_csv(path)
    swap_df = pd.read_csv(path)

    swap_df['sentence_1'], swap_df['sentence_2'] = swap_df['sentence_2'], swap_df['sentence_1']

    df = pd.concat([df, swap_df])
    return df

if __name__ == '__main__':
    df = line_swap('../../data/train.csv')

    print(df)