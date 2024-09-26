import pandas as pd
import sys
import os
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def swap_sentences(df, concatFlag=False):


def swap_sentences(df):
    """
    데이터프레임의 sentence_1과 sentence_2 컬럼을 스왑하여 데이터를 증강합니다.
    
    Args:
    df (pd.DataFrame): 원본 데이터프레임. 'sentence_1', 'sentence_2'
    
    Returns:
    pd.DataFrame: 증강된 데이터프레임
    """
    
    # 원본 데이터 복사
    swapped_df = df.copy()

    # 문장 스왑
    swapped_df['sentence_1'], swapped_df['sentence_2'] = df['sentence_2'], df['sentence_1']

    # 원본 데이터와 합치기
    if concatFlag:
        result_df = pd.concat([df, swapped_df], ignore_index=True)
    else:
        result_df = swapped_df

    return result_df

def separate_data(df, num_splits=3) -> list:
    """
    증강된 데이터를 bagging 기법에 사용하기 위해서 지정된 수 만큼 분리합니다.
    
    Args:
    df (pd.DataFrame): 원본 데이터프레임
    num_splits (int) : 분리되고자 하는 수
    
    Returns:
    list : 분리된 원본 데이터프레임
    """

    indices = list(df.index)
    random.shuffle(indices)

    num_rows = len(indices)
    split_size = num_rows // 5
    remainder = num_rows % 5

    splits = []
    start = 0
    for i in range(5):
        end = start + split_size + (1 if i < remainder else 0)  
        splits.append(df.iloc[indices[start:end]])
        start = end

    return splits

if __name__ == "__main__":
    # CSV 파일에서 데이터 로드
    original_df = pd.read_csv('/data/ephemeral/home/data/train.csv')
    
    # 데이터 증강 수행
    result_df = swap_sentences(original_df)
    
    # 결과 확인
    print(f"Original dataset size: {len(original_df)}")
    print(f"Augmented dataset size: {len(result_df)}")
    
    # 증강된 데이터셋 저장
    result_df.to_csv('swapped_dataset.csv', index=False)
    print("Augmented dataset saved to 'swapped_dataset.csv'")