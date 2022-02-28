import re

import pandas as pd
from soynlp.noun import NewsNounExtractor
from soynlp.tokenizer import LTokenizer


def preprocess_channel_subs(df: pd.DataFrame) -> pd.DataFrame:
    for i, row in df.iterrows():
        # 숫자만 남기기
        n = float(re.sub('[^0-9.]', '', row['channel_subs'])) 
        
        if '만' in row['channel_subs']:
            n = n * 10000
        elif '천' in row['channel_subs']:
            n = n * 1000
        df.loc[i, 'channel_subs'] = n

    return df

def preprocess_upload_date(df: pd.DataFrame) -> pd.DataFrame:
    df['upload_date'] = pd.to_datetime(df['upload_date'])
    
    # 연도 Column 추가
    df['year'] = df['upload_date'].apply(lambda x: x.year)
    
    # Date index 설정
    df.set_index('upload_date', inplace=True)
    df = df.sort_index()
    return df

def preprocess_hashtags(df: pd.DataFrame) -> pd.DataFrame:
    # hashtag 리스트화
    for i, row in df.iterrows():
        hashtag_lst = row['hashtags'].split(' ')
        
        # 결측치일 경우 빈 리스트
        if '' in hashtag_lst:
            hashtag_lst = []
        
        df.loc[i, 'hashtags'] = str(hashtag_lst)
    return df

def preprocess_all_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 결측치 처리
    df['hashtags'].fillna('', inplace=True)
    df['channel_subs'].fillna('0', inplace=True)
    
    # 최초 공개: 2020. 6. 17. 처리
    df['upload_date'] = df['upload_date'].apply(lambda x: re.sub('[^0-9.]', '', x))
    
    df = preprocess_channel_subs(df)
    df = preprocess_hashtags(df)
    df = preprocess_upload_date(df)
    return df

def tokenize_vid_name(df: pd.DataFrame, stopwords: list) -> pd.DataFrame:
    # 한글만 남기기
    df['vid_name'] = df['vid_name'].apply(lambda x: re.sub('[^가-힣]', ' ', x))
    
    # 명사 추출
    noun_extractor = NewsNounExtractor()
    nouns = noun_extractor.train_extract(list(df['vid_name']))
    scores = {word:score.score for word, score in nouns.items()}
    
    # tokenize
    t = LTokenizer(scores=scores)
    tokens = []

    for i, row in df.iterrows():
        result = t.tokenize(row['vid_name'])
        token = []
        # stopwords 제거
        for r in result:
            if r not in stopwords:
                token.append(r)

        tokens.append(token)

    df['tokens'] = tokens
    return df