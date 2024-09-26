import pandas as pd
from sklearn.model_selection import train_test_split

# 두 개의 CSV 파일을 합치고, binary-label의 비율을 유지하면서 8:2로 분할

# train.dev와 dev.csv 파일 읽기
train_df = pd.read_csv('./data/train.csv')
dev_df = pd.read_csv('./data/dev.csv')

# 두 DataFrame 합치기
combined_df = pd.concat([train_df, dev_df], ignore_index=True)

# binary-label 비율 유지하기
train_df, dev_df = train_test_split(combined_df, test_size=0.2, stratify=combined_df['binary-label'], random_state=42)

# 새로운 CSV 파일로 저장
train_df.to_csv('./data/new_train.csv', index=False)
dev_df.to_csv('./data/new_dev.csv', index=False)
