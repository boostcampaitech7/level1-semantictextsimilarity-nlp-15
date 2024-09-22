import pandas as pd
def swap_sentences(df):
    swapped_df = df.copy()
    swapped_df['sentence_1'], swapped_df['sentence_2'] = df['sentence_2'], df['sentence_1']
    return pd.concat([df, swapped_df], ignore_index=True)