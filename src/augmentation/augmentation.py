import pandas as pd

def under_sampling(data_path: str) -> pd.DataFrame:
    """
    label 값이 0인 데이터를 under sampling하는 함수
        Args:
            data_path (str): 증강하고자 하는 데이터의 경로
        Returns:
            df_new (DataFrame): under sampling된 데이터
    """
    df = pd.read_csv(data_path)
    df_0 = df[df["label"] == 0][1000:2000].copy()  # 라벨 0의 일부 데이터 선택
    df_new = df[df["label"] != 0].copy()  # 라벨 0이 아닌 데이터 선택
    df_new = pd.concat([df_new, df_0])  # 라벨 0 데이터와 결합
    return df_new

def swap_sentence(data_path: str) -> pd.DataFrame:
    """
    sentence 1과 sentence 2의 위치를 바꾸어 증강하는 함수
        Args:
            data_path (str): 증강하고자 하는 데이터의 경로
        Returns:
            df_swapped (DataFrame): 증강된 데이터
    """
    df = pd.read_csv(data_path)
    df_swapped = df.copy()
    df_swapped["sentence_1"] = df["sentence_2"]
    df_swapped["sentence_2"] = df["sentence_1"]
    return df_swapped

def copy_sentence(data_path: str, index_min=250, index_max=750) -> pd.DataFrame:
    """
    라벨 0의 sentence 2를 sentence 1에 복사하여 라벨 5 데이터 생성
        Args:
            data_path (str): 증강하고자 하는 데이터의 경로
            index_min (int): 증강할 데이터에서 슬라이싱 시작  default = 250
            index_max (int): 증강할 데이터에서 슬라이싱 끝    default = 750
        Returns:
            df_copied (DataFrame): 증강된 데이터
    """
    df = pd.read_csv(data_path)
    df_copied = df[df["label"] == 0][index_min:index_max].copy()
    df_copied["sentence_1"] = df_copied["sentence_2"]  # sentence 2를 sentence 1으로 복사
    df_copied["label"] = 5.0  # 라벨 5로 설정
    return df_copied

def concat_data(data_path: str, *dataframes: pd.DataFrame):
    """
    데이터프레임을 합쳐서 csv 파일로 저장하는 함수
        Args:
            data_path (str): 증강하고자 하는 데이터의 경로
            dataframes (DataFrame): 합치려고 하는 데이터프레임
    """
    result = pd.concat(dataframes)
    result.to_csv(data_path, index=False)

def augment(source_data_path, dest_data_path):
    df_original = pd.read_csv(source_data_path)  # 원본 데이터 읽기
    df_under_sampled = under_sampling(source_data_path)
    df_swapped_sentence = swap_sentence(source_data_path)
    df_copied_sentence = copy_sentence(source_data_path)
    
    # 원본 데이터와 증강된 데이터 합치기
    concat_data(dest_data_path, df_original, df_under_sampled, df_swapped_sentence, df_copied_sentence)

if __name__ == "__main__":
    augment("./data/train.csv", "./data/aug_train.csv")
