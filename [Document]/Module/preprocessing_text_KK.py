import os
os.environ["TOKENIZERS_PARALLELISM"] = "True"
import numpy as np
import pandas as pd
import math
from itertools import product, permutations, combinations # iterative combinations
from collections import defaultdict, Counter # calculate frequency in values of list
from tqdm import tqdm, tqdm_pandas # execution time
tqdm.pandas()
import time
import datetime
import multiprocessing as mp
import ray
ray.init(num_cpus=mp.cpu_count()-1, ignore_reinit_error=True, log_to_driver=False)

from datasets import Features, Value, ClassLabel
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, BertTokenizer, BertTokenizerFast
from sklearn.model_selection import train_test_split, TimeSeriesSplit

import re
import string
from ast import literal_eval
import kss    # 문장분리
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, LdaModel, LdaMulticore, CoherenceModel
from keybert import KeyBERT
## 영어
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS as sw_eng
## 한국어
from konlpy.tag import Hannanum, Kkma, Komoran, Okt, Mecab
from kss import split_sentences
from kiwipiepy import Kiwi
from spacy.lang.ko.stop_words import STOP_WORDS as sw_kor
from soynlp.normalizer import *
from soynlp.word import WordExtractor
from soynlp.noun import LRNounExtractor, LRNounExtractor_v2, NewsNounExtractor
from soynlp.tokenizer import LTokenizer
from sentence_transformers import SentenceTransformer


### Date and Author: 20240301, Kyungwon Kim ###
### 1개의 문장에 대해서 불필요한 것들 제거하는 기초 전처리
def text_preprocessor(text, language='korean', del_number=False, del_bracket_content=False, stop_words=[]):
    # 한글 맞춤법과 띄어쓰기 체크 (PyKoSpacing, Py-Hanspell)
    # html 태그 제거하기
    text_new = re.sub(r'<[^>]+>', '', str(text))
    # 괄호와 내부문자 제거하기
    if del_bracket_content:
        text_new = re.sub(r'\([^)]*\)', '', text_new)
        text_new = re.sub(r'\[[^)]*\]', '', text_new)
        text_new = re.sub(r'\<[^)]*\>', '', text_new)
        text_new = re.sub(r'\{[^)]*\}', '', text_new)
    else:
        # 괄호 제거하기
        text_new = re.sub(r'\(*\)*', '', text_new)
        text_new = re.sub(r'\[*\]*', '', text_new)
        text_new = re.sub(r'\<*\>*', '', text_new)
        text_new = re.sub(r'\{*\}*', '', text_new)
    # 따옴표 제거하기
    text_new = text_new.replace('"', '')
    text_new = text_new.replace("'", '')
    # 영어(소문자화), 한글, 숫자만 남기고 제거하기
    text_new = re.sub('[^ A-Za-z0-9가-힣]', '', text_new.lower())
    # 한글 자음과 모음 제거하기
    text_new = re.sub('([ㄱ-ㅎㅏ-ㅣ]+)', '', text_new)
    # 숫자 제거하기
    if del_number:
        text_new = re.sub(r'\d+', '', text_new)
    # 숫자를 문자로 인식하기
    text_new = ' '.join([str(word) for word in text_new.split(' ')])
    # 양쪽공백 제거하기
    text_new = text_new.strip()
    # 문장구두점 제거하기
    translator = str.maketrans('', '', string.punctuation)
    text_new = text_new.translate(translator)
    # 2개 이상의 반복글자 줄이기
    text_new = ' '.join([emoticon_normalize(word, num_repeats=2) for word in text_new.split(' ')])
    text_new = ' '.join([repeat_normalize(word, num_repeats=2) for word in text_new.split(' ')])
    # 영어 및 한글 stopwords 제거하기
    stop_words_eng = set(stopwords.words('english'))
    stop_words_kor = ['아', '휴', '아이구', '아이쿠', '아이고', '어', '나', '우리', '저희', '따라', '의해', '을', '를', '에', '의', '가', '으로', 
 '로', '에게', '뿐이다', '의거하여', '근거하여', '입각하여', '기준으로', '예하면', '예를 들면', '예를 들자면', '저', '소인', 
 '소생', '저희', '지말고', '하지마', '하지마라', '다른', '물론', '또한', '그리고', '비길수 없다', '해서는 안된다', '뿐만 아니라', 
 '만이 아니다', '만은 아니다', '막론하고', '관계없이', '그치지 않다', '그러나', '그런데', '하지만', '든간에', '논하지 않다',
 '따지지 않다', '설사', '비록', '더라도', '아니면', '만 못하다', '하는 편이 낫다', '불문하고', '향하여', '향해서', '향하다',
 '쪽으로', '틈타', '이용하여', '타다', '오르다', '제외하고', '이 외에', '이 밖에', '하여야', '비로소', '한다면 몰라도', '외에도',
 '이곳', '여기', '부터', '기점으로', '따라서', '할 생각이다', '하려고하다', '이리하여', '그리하여', '그렇게 함으로써', '하지만',
 '일때', '할때', '앞에서', '중에서', '보는데서', '으로써', '로써', '까지', '해야한다', '일것이다', '반드시', '할줄알다',
 '할수있다', '할수있어', '임에 틀림없다', '한다면', '등', '등등', '제', '겨우', '단지', '다만', '할뿐', '딩동', '댕그', '대해서',
 '대하여', '대하면', '훨씬', '얼마나', '얼마만큼', '얼마큼', '남짓', '여', '얼마간', '약간', '다소', '좀', '조금', '다수', '몇',
 '얼마', '지만', '하물며', '또한', '그러나', '그렇지만', '하지만', '이외에도', '대해 말하자면', '뿐이다', '다음에', '반대로',
 '반대로 말하자면', '이와 반대로', '바꾸어서 말하면', '바꾸어서 한다면', '만약', '그렇지않으면', '까악', '툭', '딱', '삐걱거리다',
 '보드득', '비걱거리다', '꽈당', '응당', '해야한다', '에 가서', '각', '각각', '여러분', '각종', '각자', '제각기', '하도록하다',
 '와', '과', '그러므로', '그래서', '고로', '한 까닭에', '하기 때문에', '거니와', '이지만', '대하여', '관하여', '관한', '과연',
 '실로', '아니나다를가', '생각한대로', '진짜로', '한적이있다', '하곤하였다', '하', '하하', '허허', '아하', '거바', '와', '오',
 '왜', '어째서', '무엇때문에', '어찌', '하겠는가', '무슨', '어디', '어느곳', '더군다나', '하물며', '더욱이는', '어느때', '언제',
 '야', '이봐', '어이', '여보시오', '흐흐', '흥', '휴', '헉헉', '헐떡헐떡', '영차', '여차', '어기여차', '끙끙', '아야', '앗',
 '아야', '콸콸', '졸졸', '좍좍', '뚝뚝', '주룩주룩', '솨', '우르르', '그래도', '또', '그리고', '바꾸어말하면', '바꾸어말하자면',
 '혹은', '혹시', '답다', '및', '그에 따르는', '때가 되어', '즉', '지든지', '설령', '가령', '하더라도', '할지라도', '일지라도',
 '지든지', '몇', '거의', '하마터면', '인젠', '이젠', '된바에야', '된이상', '만큼어찌됏든', '그위에', '게다가', '점에서 보아',
 '비추어 보아', '고려하면', '하게될것이다', '일것이다', '비교적', '좀', '보다더', '비하면', '시키다', '하게하다', '할만하다',
 '의해서', '연이서', '이어서', '잇따라', '뒤따라', '뒤이어', '결국', '의지하여', '기대여', '통하여', '자마자', '더욱더',
 '불구하고', '얼마든지', '마음대로', '주저하지 않고', '곧', '즉시', '바로', '당장', '하자마자', '밖에 안된다', '하면된다',
 '그래', '그렇지', '요컨대', '다시 말하자면', '바꿔 말하면', '즉', '구체적으로', '말하자면', '시작하여', '시초에', '이상', '허',
 '헉', '허걱', '바와같이', '해도좋다', '해도된다', '게다가', '더구나', '하물며', '와르르', '팍', '퍽', '펄렁', '동안', '이래',
 '하고있었다', '이었다', '에서', '로부터', '까지', '예하면', '했어요', '해요', '함께', '같이', '더불어', '마저', '마저도',
 '양자', '모두', '습니다', '가까스로', '하려고하다', '즈음하여', '다른', '다른 방면으로', '해봐요', '습니까', '했어요',
 '말할것도 없고', '무릎쓰고', '개의치않고', '하는것만 못하다', '하는것이 낫다', '매', '매번', '들', '모', '어느것', '어느',
 '로써', '갖고말하자면', '어디', '어느쪽', '어느것', '어느해', '어느 년도', '라 해도', '언젠가', '어떤것', '어느것', '저기',
 '저쪽', '저것', '그때', '그럼', '그러면', '요만한걸', '그래', '그때', '저것만큼', '그저', '이르기까지', '할 줄 안다',
 '할 힘이 있다', '너', '너희', '당신', '어찌', '설마', '차라리', '할지언정', '할지라도', '할망정', '할지언정', '구토하다',
 '게우다', '토하다', '메쓰겁다', '옆사람', '퉤', '쳇', '의거하여', '근거하여', '의해', '따라', '힘입어', '그', '다음', '버금',
 '두번째로', '기타', '첫번째로', '나머지는', '그중에서', '견지에서', '형식으로 쓰여', '입장에서', '위해서', '단지', '의해되다',
 '하도록시키다', '뿐만아니라', '반대로', '전후', '전자', '앞의것', '잠시', '잠깐', '하면서', '그렇지만', '다음에', '그러한즉',
 '그런즉', '남들', '아무거나', '어찌하든지', '같다', '비슷하다', '예컨대', '이럴정도로', '어떻게', '만약', '만일',
 '위에서 서술한바와같이', '인 듯하다', '하지 않는다면', '만약에', '무엇', '무슨', '어느', '어떤', '아래윗', '조차', '한데',
 '그럼에도 불구하고', '여전히', '심지어', '까지도', '조차도', '하지 않도록', '않기 위하여', '때', '시각', '무렵', '시간',
 '동안', '어때', '어떠한', '하여금', '네', '예', '우선', '누구', '누가 알겠는가', '아무도', '줄은모른다', '줄은 몰랏다',
 '하는 김에', '겸사겸사', '하는바', '그런 까닭에', '한 이유는', '그러니', '그러니까', '때문에', '그', '너희', '그들', '너희들',
 '타인', '것', '것들', '너', '위하여', '공동으로', '동시에', '하기 위하여', '어찌하여', '무엇때문에', '붕붕', '윙윙', '나',
 '우리', '엉엉', '휘익', '윙윙', '오호', '아하', '어쨋든', '만 못하다하기보다는', '차라리', '하는 편이 낫다', '흐흐', '놀라다',
 '상대적으로 말하자면', '마치', '아니라면', '쉿', '그렇지 않으면', '그렇지 않다면', '안 그러면', '아니었다면', '하든지', '아니면',
 '이라면', '좋아', '알았어', '하는것도', '그만이다', '어쩔수 없다', '하나', '일', '일반적으로', '일단', '한켠으로는', '오자마자',
 '이렇게되면', '이와같다면', '전부', '한마디', '한항목', '근거로', '하기에', '아울러', '하지 않도록', '않기 위해서', '이르기까지',
 '이 되다', '로 인하여', '까닭으로', '이유만으로', '이로 인하여', '그래서', '이 때문에', '그러므로', '그런 까닭에', '알 수 있다',
 '결론을 낼 수 있다', '으로 인하여', '있다', '어떤것', '관계가 있다', '관련이 있다', '연관되다', '어떤것들', '에 대해', '이리하여',
 '그리하여', '여부', '하기보다는', '하느니', '하면 할수록', '운운', '이러이러하다', '하구나', '하도다', '다시말하면', '다음으로',
 '에 있다', '에 달려 있다', '우리', '우리들', '오히려', '하기는한데', '어떻게', '어떻해', '어찌됏어', '어때', '어째서', '본대로',
 '자', '이', '이쪽', '여기', '이것', '이번', '이렇게말하자면', '이런', '이러한', '이와 같은', '요만큼', '요만한 것',
 '얼마 안 되는 것', '이만큼', '이 정도의', '이렇게 많은 것', '이와 같다', '이때', '이렇구나', '것과 같이', '끼익', '삐걱', '따위',
 '와 같은 사람들', '부류의 사람들', '왜냐하면', '중의하나', '오직', '오로지', '에 한하다', '하기만 하면', '도착하다',
 '까지 미치다', '도달하다', '정도에 이르다', '할 지경이다', '결과에 이르다', '관해서는', '여러분', '하고 있다', '한 후', '혼자',
 '자기', '자기집', '자신', '우에 종합한것과같이', '총적으로 보면', '총적으로 말하면', '총적으로', '대로 하다', '으로서', '참',
 '그만이다', '할 따름이다', '쿵', '탕탕', '쾅쾅', '둥둥', '봐', '봐라', '아이야', '아니', '와아', '응', '아이', '참나', '년',
 '월', '일', '령', '영', '일', '이', '삼', '사', '오', '육', '륙', '칠', '팔', '구', '이천육', '이천칠', '이천팔', '이천구',
 '하나', '둘', '셋', '넷', '다섯', '여섯', '일곱', '여덟', '아홉', '령', '영']
    # 분석가 stopwords 추가
    if stop_words != []:
        stop_words_kor.extend(stop_words)
    text_new = ' '.join([word for word in text_new.split(' ') if word not in stop_words_eng])
    text_new = ' '.join([word for word in text_new.split(' ') if word not in stop_words_kor])
    text_new = ' '.join([word for word in text_new.split(' ') if word not in sw_eng])
    text_new = ' '.join([word for word in text_new.split(' ') if word not in sw_kor])
    # 한글과 영어 분리저장
    if language == 'korean':
        text_new = re.sub(r'[a-zA-Z]', '', text_new)
    elif language == 'english':
        text_new = re.sub(r'[ㄱ-ㅎㅏ-ㅣ가-힣]', '', text_new)
    
    return text_new


### Date and Author: 20250224, Kyungwon Kim ###
### series에 대해서 불필요한 것들 제거하는 기초 전처리
def preprocessing_basicNLP(df, colname, language='korean',
                           del_number=False, del_bracket_content=False, stop_words=[],
                           save_local=True, save_name='df_prepbasic.csv'):
    # 기초 전처리 진행
    df[colname+'_Prep'] = df[colname].progress_apply(lambda x: text_preprocessor(x, del_number=del_number, 
                                                                                 del_bracket_content=del_bracket_content,
                                                                                 stop_words=stop_words))

    # 정리
    ## 컬럼 정렬
    colnames = [col for col in df.columns if col not in [colname, colname+'_Prep']] + [colname, colname+'_Prep']
    df = df[colnames]
    ## na 제거
    df.dropna(subset=[colname+'_Prep'], inplace=True)
    df = df[df[colname+'_Prep'] != '']
    df = df[df[colname+'_Prep'] != 'nan']
    df = df[df[colname+'_Prep'] != np.nan]
    df = df[~df[colname+'_Prep'].isnull()]
    df = df.reset_index().iloc[:,1:]

    # 저장
    if save_local:
        print('Saving...:', datetime.datetime.now())
        folder_location = os.path.join(os.getcwd(), 'Data', '')
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)
        df.to_csv(os.path.join(folder_location, save_name), index=False, encoding='utf-8-sig')
    
    return df    


def preprocessing_nounextract(df_series):
    # 단어 추출기
    ## cohesion/branching entropy/accessor variety값이 큰 경우 하나의 단어일 가능성 높음
    word_extractor = WordExtractor()
    word_extractor.train(list(df_series.values))
    word_score = word_extractor.extract()
    ## cohesion_forward*right_branching_entropy 로 scoring
    words = {word:score.cohesion_forward * math.exp(score.right_branching_entropy) for word, score in word_score.items()
             if len(word) != 1}
    nouns_WE = sorted(words.items(), key=lambda x:x[1], reverse=True)

    # 단어추출기
    noun_extractor = LRNounExtractor_v2(verbose=False, extract_compound=True)
    noun_extractor.train(list(df_series.values))
    noun_score = noun_extractor.extract()
    ## unique 단어들로 필터
    unique_nouns = list(set([words for word_tuple in noun_extractor._compounds_components.values() for words in word_tuple]))
    nouns = {noun:noun_score[noun] for noun in unique_nouns if noun in noun_score.keys()}
    ## 빈도수*명사점수 로 scoring
    nouns = {noun:score.frequency * score.score for noun, score in nouns.items() if len(noun) != 1}
    nouns_LRE = sorted(nouns.items(), key=lambda x:x[1], reverse=True)
    
    # 단어추출기
    noun_extractor = NewsNounExtractor(verbose=False)
    noun_extractor.train(list(df_series.values))
    noun_score = noun_extractor.extract()
    ## 빈도수*명사점수 로 scoring
    nouns = {noun:score.frequency * score.score for noun, score in noun_score.items() if len(noun) != 1}
    nouns_NE = sorted(nouns.items(), key=lambda x:x[1], reverse=True)
    
    # 정리: 3가지 추출기에서 공통으로 추출된 단어들의 score들을 더하여 내림차순
    nouns_unique = list(set([word for word_result in [nouns_WE, nouns_LRE, nouns_NE] for word, _ in word_result]))
    nouns_intersection = []
    for noun in nouns_unique:
        if (noun in noun in dict(nouns_WE).keys()) and (noun in noun in dict(nouns_LRE).keys()) and (noun in noun in dict(nouns_NE).keys()):
            nouns_intersection.append((noun, int(dict(nouns_WE)[noun] + dict(nouns_LRE)[noun] + dict(nouns_NE)[noun])))
    df_wordfreq = sorted(dict(nouns_intersection).items(), key=lambda x:x[1], reverse=True)
    df_wordfreq = pd.DataFrame(df_wordfreq, columns=['word', 'score'])

    return df_wordfreq


def preprocessing_tfidf(df_series, max_features=1000, ngram_range=(1,1), del_lowfreq=False):
    # 빈도 학습
    tfidfier = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    tfidfier.fit(df_series.to_list())
#     ## 빈도 정리
#     df_wordfreq = pd.DataFrame.from_dict([tfidfier.vocabulary_]).T.reset_index()
#     df_wordfreq.columns = ['word', 'freq']
#     df_wordfreq = df_wordfreq.sort_values(by=[df_wordfreq.columns[-1]], ascending=False)
    ## TF-IDF 점수 정리
    df_wordscore = pd.DataFrame(tfidfier.transform(df_series.to_list()).sum(axis=0), 
                                columns=tfidfier.get_feature_names_out()).T.reset_index()
    df_wordscore.columns = ['word', 'score']
    df_wordscore = df_wordscore.sort_values(by=[df_wordscore.columns[-1]], ascending=False).reset_index().iloc[:,1:]
    df_wordscore['score'] = df_wordscore['score'].astype(int)
    ## 문장 벡터 정리
    df_sentvec = tfidfier.transform(df_series.to_list()).toarray()
    df_sentvec = pd.DataFrame(df_sentvec, index=['sentence' + str(i+1) for i in range(df_series.shape[0])], 
                              columns=tfidfier.get_feature_names_out())
    
    # 저빈도 삭제
    if del_lowfreq:
        del_criteria = df_sentvec.sum(axis=0).mean()
        del_columns = df_sentvec.columns[df_sentvec.sum(axis=0) < del_criteria]
        df_sentvec = df_sentvec[[col for col in df_sentvec.columns if col not in del_columns]]
#         df_wordfreq = df_wordfreq[df_wordfreq.word.apply(lambda x: False if x in del_columns else True)]
        df_wordscore = df_wordscore[df_wordscore.word.apply(lambda x: False if x in del_columns else True)]
          
    return df_wordscore, df_sentvec


def preprocessing_keybert(df_series, ngram_range=(1,1), doc_topn_kwd=5):
    # 키워드 추출
    keyword_extractor = KeyBERT('distilbert-base-nli-mean-tokens')
    scores = df_series.apply(lambda x: keyword_extractor.extract_keywords(x, keyphrase_ngram_range=ngram_range, top_n=doc_topn_kwd))
    scores = [score_1d for score in scores.tolist() for score_1d in score]
        
    # 정리
    df_wordscore = pd.DataFrame(scores, columns=['word', 'score'])
    df_wordscore = df_wordscore.groupby('word').agg('sum').sort_values('score', ascending=False).reset_index()
    df_wordscore['score'] = df_wordscore['score'].astype(int)
#     df_wordscore = df_wordscore[df_wordscore['score'] > 1].reset_index().iloc[:,1:]
    
    return df_wordscore    


### Date and Author: 20250219, Kyungwon Kim ###
### 키워드빈도결과에 따라 문장을 주요키워드의 인접 단어들로 방향/비방향 기준 edge list로 변환
### 메모리 효율성과 정보손실 우려로 sent2edgelist > preprocessing_sent2adjmat
def preprocessing_sent2edgelist(df_series, window_size=1, save_local=True):
    # 전체 어휘 추출 및 인덱싱
    print('sentence to cooccurent matrix...')
    ## categ 적용시 전처리
    if pd.Series(df_series).shape[0] != 1:
        sentences = df_series.fillna('').str.split()
    else:
        try:
            sentences = df_series.split()
        except:
            sentences = df_series.str.split()
    word_split = [word.strip() for sentence in sentences for word in sentence]
    word_unique = list(set(word_split))
    
    # 빈사전 만들기
    coword_direct = defaultdict(lambda: defaultdict(int))
    coword_pair = defaultdict(int)
    
    # 행렬 채우기
    for sentence in tqdm(sentences):
        for moving in range(len(sentence) - window_size):
            for idx, coword in enumerate(sentence[moving:moving+window_size+1]):
                coword = coword.strip()
                if idx == 0:
                    word_main = coword
                else:
                    ## 인접단어사전 계산
                    coword_direct[word_main][coword] += 1
                    key = tuple(sorted([word_main, coword]))
                    coword_pair[key] += 1

    # 정리
    df_coword_direct, df_coword_pair = [], []
    for source, targets in coword_direct.items():
        for target, count in targets.items():
            df_coword_direct.append({'source': source, 'target': target, 'weight': count})
    df_coword_direct = pd.DataFrame(df_coword_direct)
    for (source, target), count in coword_pair.items():
        df_coword_pair.append({'source': source, 'target': target, 'weight': count})
    df_coword_pair = pd.DataFrame(df_coword_pair)

    # 저장
    if save_local:
        print('Saving...:', datetime.datetime.now())
        folder_location = os.path.join(os.getcwd(), 'Result', 'WordFreq', '')
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)
        df_coword_direct.to_csv(os.path.join(folder_location, 'co_edgelist_direct.csv'), 
                                index=True, encoding='utf-8-sig')
        df_coword_pair.to_csv(os.path.join(folder_location, 'co_edgelist_undirect.csv'), 
                              index=True, encoding='utf-8-sig')

    return df_coword_direct, df_coword_pair


### Date and Author: 20250219, Kyungwon Kim ###
### 카테고리별 키워드빈도결과에 따라 문장을 주요키워드의 인접 단어들로 방향/비방향 기준 edge list로 변환
def preprocessing_sent2edgelist_bycateg(df, colname_target, colname_categ, window_size=1, save_local=True):
    # 전체 어휘 추출 및 인덱싱
    print('sentence to cooccurent matrix...')
    ## 데이터분리
    df_coword_direct, df_coword_pair = pd.DataFrame(), pd.DataFrame()
    for idx, categ in enumerate(df[colname_categ].unique()):
        print('Progress..: ', (idx+1)/len(df[colname_categ].unique())*100, '%')
        df_sub = df[df[colname_categ] == categ]
        df_series = df_sub[colname_target]

        ## sent2edgelist
        coword_direct, coword_pair = preprocessing_sent2edgelist(df_series, window_size=window_size, save_local=False)
        ## categ 추가 및 결합
        coword_direct[colname_categ] = categ
        coword_pair[colname_categ] = categ
        df_coword_direct = pd.concat([df_coword_direct, coword_direct], axis=0)
        df_coword_pair = pd.concat([df_coword_pair, coword_pair], axis=0)
    df_coword_direct = df_coword_direct.reset_index().iloc[:,1:]
    df_coword_pair = df_coword_pair.reset_index().iloc[:,1:]
    
    # 저장
    if save_local:
        print('Saving...:', datetime.datetime.now())
        folder_location = os.path.join(os.getcwd(), 'Result', 'WordFreq', '')
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)
        df_coword_direct.to_csv(os.path.join(folder_location, 'co_categ_edgelist_direct.csv'), 
                                index=True, encoding='utf-8-sig')
        df_coword_pair.to_csv(os.path.join(folder_location, 'co_categ_edgelist_undirect.csv'), 
                              index=True, encoding='utf-8-sig')

    return df_coword_direct, df_coword_pair
  

### Date and Author: 20250215, Kyungwon Kim ###
### 키워드빈도결과에 따라 문장을 주요키워드의 인접 단어들로 방향/비방향 기준 행렬로 변환
# df_coword_direct, df_coword_pair = preprocessing_sent2cooccurmat(df_freq[TARGET+'_Prep'], window_size=WINDOW_SIZE, 
#                                                                  word_freq=word_freq, num_showkeyword=NUM_SHOWKEYWORD, 
#                                                                  save_local=SAVE_LOCAL)
# df_coword_direct, df_coword_pair = preprocessing_sent2cooccurmat(df_freq['Token_'+TOKENIZER], window_size=WINDOW_SIZE,
#                                                                  word_freq=None, num_showkeyword=NUM_SHOWKEYWORD,
#                                                                  save_local=SAVE_LOCAL)
def preprocessing_sent2adjmat(df_series, window_size=1, 
                              word_freq=None, num_showkeyword=100, 
                              save_local=True):
    # 전체 어휘 추출 및 인덱싱
    print('sentence to cooccurent matrix...')
    ## word_freq 입력시 필터링
    if type(word_freq) == pd.DataFrame:
        df_series = preprocessing_sent2kwd(df_series.fillna(''), word_freq.iloc[:num_showkeyword,:], colname='Sent_Token')
        df_series = df_series.squeeze()
    sentences = df_series.fillna('').str.split()
    word_split = [word.strip() for sentence in sentences for word in sentence]
    word_unique = list(set(word_split))
    word_index = {word: idx for idx, word in enumerate(word_unique)}
    
    # 빈행렬 만들기
    df_coword_direct = np.zeros((len(word_unique), len(word_unique)), dtype=int)
    df_coword_pair = np.zeros((len(word_unique), len(word_unique)), dtype=int)
    
    # 행렬 채우기
    for sentence in tqdm(sentences):
        for moving in range(len(sentence) - window_size):
            for idx, coword in enumerate(sentence[moving:moving+window_size+1]):
                coword = coword.strip()
                if idx == 0:
                    word_main = coword
                else:
                    ## 인접단어행렬 계산
                    df_coword_direct[word_index[word_main]][word_index[coword]] += 1
                    df_coword_pair[word_index[word_main]][word_index[coword]] += 1
                    df_coword_pair[word_index[coword]][word_index[word_main]] += 1
    ## 대각행렬 무시 또는 자기자신 돌아오는 그래프 무시 필요시
    # df_coword_direct, df_coword_pair = np.fill_diagonal(df_coword_direct, 0), np.fill_diagonal(df_coword_pair, 0)
    df_coword_direct = pd.DataFrame(df_coword_direct, index=word_unique, columns=word_unique)
    df_coword_pair = pd.DataFrame(df_coword_pair, index=word_unique, columns=word_unique)

    # 정리
    ## 전체 데이터를 1차원으로 변환 후 상위100번째 값을 임계치로 지정 -> 임계치가 1개라도 있는 컬럼만 선택
    threshold = np.sort(df_coword_direct.values.flatten())[-num_showkeyword]
    colname_keep = df_coword_direct.columns[(df_coword_direct > threshold).any()]
    df_coword_direct = df_coword_direct.loc[colname_keep, colname_keep]
    threshold = np.sort(df_coword_pair.values.flatten())[-num_showkeyword]
    colname_keep = df_coword_pair.columns[(df_coword_pair > threshold).any()]
    df_coword_pair.loc[colname_keep, colname_keep]        

    # 저장
    print('Saving...:', datetime.datetime.now())
    if save_local:
        folder_location = os.path.join(os.getcwd(), 'Result', 'WordFreq', '')
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)
        df_coword_direct.to_csv(os.path.join(folder_location, 'co_adjmat_direct.csv'), 
                                index=True, encoding='utf-8-sig')
        df_coword_pair.to_csv(os.path.join(folder_location, 'co_adjmat_undirect.csv'), 
                              index=True, encoding='utf-8-sig')
    
    return df_coword_direct, df_coword_pair


### Date and Author: 20250218, Kyungwon Kim ###
### 동시출현단어 정방행렬을 쌍별 수치 edge list df로 변환
def preprocessing_adjmat2edgelist(df_adjmat, remove_diag=False):
    if remove_diag:
        df_edgelist = pd.DataFrame([(row, col, df_adjmat.loc[row, col]) 
                                  for row in list(df_adjmat.index) for col in list(df_adjmat.columns) 
                                  if (row != col) and (df_adjmat.loc[row, col] != 0)])
    else:
        df_edgelist = pd.DataFrame([(row, col, df_adjmat.loc[row, col]) 
                                  for row in list(df_adjmat.index) for col in list(df_adjmat.columns)
                                  if (df_adjmat.loc[row, col] != 0)])
    df_edgelist.columns = ['source', 'target', 'weight']
    
    return df_edgelist


### Date and Author: 20250217, Kyungwon Kim ###
### 주요 키워드 다음에 등장하는 인접 단어들로 방향/비방향 빈도 변환
### 바로 위 함수와 중복이라서 향후 수정 또는 제거 필요하나 일단 속도가 빨라서 내부 함수로 사용중
def preprocessing_sent2wordadjfreq(df_keyword, df_series, num_showkeyword=100):
    # 세팅
    keyword_list = list(df_keyword.iloc[:,0].values)
    
    # 각각의 keyword 별로 연산
    df_word_adjdirect, df_word_adjpair = pd.DataFrame(), pd.DataFrame()
    for keyword in keyword_list[:num_showkeyword]:
        # 인접 단어들 모아서 정렬
        sub_direct, sub_pair = [], []
        for row in df_series.values:
            if keyword in row.split(' '):
                for idx, val in enumerate(row.split(keyword)):
                    if idx == 0:
                        sub_pair.append(val.strip().split(' ')[-1])
                    elif idx == len(row.split(keyword))-1:
                        sub_direct.append(val.strip().split(' ')[0])
                        sub_pair.append(val.strip().split(' ')[0])
                    else:
                        sub_direct.append(val.strip().split(' ')[0])
                        sub_pair.append(val.strip().split(' ')[0])
                        if val.strip().split(' ')[0] != val.strip().split(' ')[-1]:
                            sub_direct.append(val.strip().split(' ')[-1])
                            sub_pair.append(val.strip().split(' ')[-1])
        sub_direct, sub_pair = [i for i in sub_direct if len(i) != 0], [i for i in sub_pair if len(i) != 0]
        sub_direct = dict([str(keyword)+' '+i, sub_direct.count(i)] for i in set(sub_direct))
        sub_pair = dict([str(keyword)+' '+i, sub_pair.count(i)] for i in set(sub_pair))

        # 정렬
        sub_direct = pd.DataFrame(sorted(sub_direct.items(), key=lambda x:x[1], reverse=True)[:num_showkeyword])
        sub_pair = pd.DataFrame(sorted(sub_pair.items(), key=lambda x:x[1], reverse=True)[:num_showkeyword])

        # 정리
        if sub_direct.shape[1] == 2:
            df_word_adjdirect = pd.concat([df_word_adjdirect, sub_direct], axis=0)
            df_word_adjpair = pd.concat([df_word_adjpair, sub_pair], axis=0)
    df_word_adjdirect.columns, df_word_adjpair.columns = ['word', 'score'], ['word', 'score']
    df_word_adjdirect, df_word_adjpair = df_word_adjdirect.reset_index().iloc[:,1:], df_word_adjpair.reset_index().iloc[:,1:]
    
    return df_word_adjdirect, df_word_adjpair


### Date and Author: 20250216, Kyungwon Kim ###
### 키워드빈도결과에 따라 문장을 주요키워드로 변환
def preprocessing_sent2kwd(df_series, word_freq, colname='Sent_Token'):
    # 필터링
    doc_token = df_series.str.split()
    word_freq_unique = [word for row in word_freq.word.str.split() for word in row]
    doc_token = doc_token.progress_apply(lambda x: [word if (len(x) != 0) and (word in word_freq_unique) else '' for word in x])
    doc_token = pd.DataFrame(' '.join(x) for x in doc_token)
    doc_token.columns = [colname]
    
    return doc_token

# 작동은하는데 10000개 샘플 기준 속도는 느림
@ray.remote
def preprocessing_sent2token(sentence, word_freq, colname='Sent_Token'):
    # 필터링
    sent_token = sentence.split()
    word_freq_unique = [word for row in word_freq.word.str.split() for word in row]
    sent_token = [word for word in sent_token if (len(sent_token) != 0) and (word in word_freq_unique)]
    return sent_token

def preprocessing_sent2kwdRay(df_series, word_freq, colname='Sent_Token'):
    ray.init(num_cpus=mp.cpu_count()-1, ignore_reinit_error=True, log_to_driver=False)
    task = [preprocessing_sent2token.remote(sentence, word_freq) for sentence in df_series.values]
    doc_token = ray.get(task)
    ray.shutdown()
    doc_token = pd.DataFrame(' '.join(x) for x in doc_token)
    doc_token.columns = [colname]
    return doc_token


### Date and Author: 20250217, Kyungwon Kim ###
### 3개의 패키지 사용 단어의 주요키워드 방향기준 빈도 계산 및 변환
def preprocessing_wordfreq_3library(df, colname_target, language='korean', 
                                    ngram_range=(1,1),
                                    tfidf_maxcol=1000, tfidf_dellowfreq=False, 
                                    keybert_topnkwd=5,
                                    num_showkeyword=10):
    # 빈도 계산
    word_freq_soynlp, wordadj_freq_soynlp = pd.DataFrame(), pd.DataFrame()
    word_freq_tfidf, wordadj_freq_tfidf = pd.DataFrame(), pd.DataFrame()
    word_freq_keybert, wordadj_freq_keybert = pd.DataFrame(), pd.DataFrame()
    if language == 'korean':
        try:
            print('Preprocessing...: SoyNLP', datetime.datetime.now())
            # 한글단어 요약
            word_freq_soynlp = preprocessing_nounextract(df[colname_target])
        except:
            pass
    try:
        print('Preprocessing...: TF-IDF', datetime.datetime.now())
        # TF-IDF 요약
        word_freq_tfidf, sent_mat = preprocessing_tfidf(df[colname_target], ngram_range=ngram_range,
                                                        max_features=tfidf_maxcol, del_lowfreq=tfidf_dellowfreq)
    except:
        pass
    try:
        print('Preprocessing...: KeyBERT', datetime.datetime.now())
        # keybert 요약
        word_freq_keybert = preprocessing_keybert(df[colname_target], ngram_range=ngram_range, doc_topn_kwd=keybert_topnkwd)
    except:
        pass

    return word_freq_soynlp, word_freq_tfidf, word_freq_keybert

### 작동안됨 향후작업
@ray.remote
def preprocessing_wordfreq_3lib(df, colname_target, language='kr', 
                                ngram_range=(1,1),
                                tfidf_maxcol=1000, tfidf_dellowfreq=False, 
                                keybert_topnkwd=5,
                                num_showkeyword=10):
    # 빈도 계산
    word_freq_soynlp, wordadj_freq_soynlp = pd.DataFrame(), pd.DataFrame()
    word_freq_tfidf, wordadj_freq_tfidf = pd.DataFrame(), pd.DataFrame()
    word_freq_keybert, wordadj_freq_keybert = pd.DataFrame(), pd.DataFrame()
    if language == 'kr':
        try:
            print('Preprocessing...: SoyNLP', datetime.datetime.now())
            # 한글단어 요약
            word_freq_soynlp = preprocessing_nounextract(df[colname_target])
            ## 인접어반영 요약
            _, wordadj_freq_soynlp = preprocessing_sent2wordadjfreq(word_freq_soynlp, 
                                                                   df[colname_target], num_showkeyword=num_showkeyword)
        except:
            pass
    try:
        print('Preprocessing...: TF-IDF', datetime.datetime.now())
        # TF-IDF 요약
        word_freq_tfidf, sent_mat = preprocessing_tfidf(df[colname_target], ngram_range=ngram_range,
                                                        max_features=tfidf_maxcol, del_lowfreq=tfidf_dellowfreq)
        ## 인접어반영 요약
        _, wordadj_freq_tfidf = preprocessing_sent2wordadjfreq(word_freq_tfidf, 
                                                               df[colname_target], num_showkeyword=num_showkeyword)
    except:
        pass
    try:
        print('Preprocessing...: KeyBERT', datetime.datetime.now())
        # keybert 요약
        word_freq_keybert = preprocessing_keybert(df[colname_target], ngram_range=ngram_range, doc_topn_kwd=keybert_topnkwd)
        ## 인접어반영 요약
        _, wordadj_freq_keybert = preprocessing_sent2wordadjfreq(word_freq_keybert, 
                                                                 df[colname_target], num_showkeyword=num_showkeyword)
    except:
        pass

    return word_freq_soynlp, wordadj_freq_soynlp, word_freq_tfidf, wordadj_freq_tfidf, word_freq_keybert, wordadj_freq_keybert

def preprocessing_wordfreq_3libRay(df, colname_target, colname_category, language='kr', 
                                   ngram_range=(1,1),
                                   tfidf_maxcol=1000, tfidf_dellowfreq=False, 
                                   keybert_topnkwd=5,
                                   num_showkeyword=10):
    ray.init(num_cpus=mp.cpu_count()-1, ignore_reinit_error=True, log_to_driver=False)
    df_subs = [df[df[colname_category] == category] for category in sorted(df[colname_category].unique())]
    task = [preprocessing_wordfreq_3lib.remote(df_sub, colname_target, colname_category) for df_sub in df_subs]
    result = ray.get(task)
    ray.shutdown()
    return result
    

### Date and Author: 20250217, Kyungwon Kim ###
### 3개의 패키지 사용 단어의 주요키워드와 인접키워드 방향기준 빈도 계산 및 변환
def preprocessing_wordfreq(df, colname_target, colname_category=None, language='kr', 
                           ngram_range=(1,1),
                           tfidf_maxcol=1000, tfidf_dellowfreq=False, 
                           keybert_topnkwd=5,
                           num_showkeyword=10, save_local=True, save_name='wf.csv'):
    # 빈도계산
    if colname_category == None:
        word_freq_soynlp, word_freq_tfidf, word_freq_keybert = preprocessing_wordfreq_3library(df=df, colname_target=colname_target, language=language,
                                                                                              ngram_range=ngram_range,
                                                                                              tfidf_maxcol=tfidf_maxcol, tfidf_dellowfreq=tfidf_dellowfreq,
                                                                                              keybert_topnkwd=keybert_topnkwd,
                                                                                              num_showkeyword=num_showkeyword)
    ## 카테고리 있는 경우
    elif type(colname_category) == str:
        word_freq_soynlp, word_freq_tfidf, word_freq_keybert = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for category in tqdm(sorted(df[colname_category].unique())):
            df_sub = df[df[colname_category] == category]
            word_freq_soy, word_freq_tf, word_freq_key = preprocessing_wordfreq_3library(df=df_sub, colname_target=colname_target, language=language,
                                                                                      ngram_range=ngram_range,
                                                                                      tfidf_maxcol=tfidf_maxcol, tfidf_dellowfreq=tfidf_dellowfreq,
                                                                                      keybert_topnkwd=keybert_topnkwd,
                                                                                      num_showkeyword=num_showkeyword)
            ## 병합 및 순서 정리
            word_freq_soy['category'] = str(category)
            word_freq_soy = word_freq_soy[['category']+list(word_freq_soy.columns[:-1])]
            word_freq_soynlp = pd.concat([word_freq_soynlp, word_freq_soy], axis=0, ignore_index=True)
            
            word_freq_tf['category'] = str(category)
            word_freq_tf = word_freq_tf[['category']+list(word_freq_tf.columns[:-1])]
            word_freq_tfidf = pd.concat([word_freq_tfidf, word_freq_tf], axis=0, ignore_index=True)
            
            word_freq_key['category'] = str(category)
            word_freq_key = word_freq_key[['category']+list(word_freq_key.columns[:-1])]
            word_freq_keybert = pd.concat([word_freq_keybert, word_freq_key], axis=0, ignore_index=True)

    # 문장 키워드화
    print('Sentence to Keywords...:', datetime.datetime.now())
    sent_kwd_soynlp, sent_kwd_tfidf, sent_kwd_keybert = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if word_freq_soynlp.shape[0] != 0:
        sent_kwd_soynlp = preprocessing_sent2kwd(df[colname_target], word_freq_soynlp, colname='Token_SoyNLP')
    if word_freq_tfidf.shape[0] != 0:
        sent_kwd_tfidf = preprocessing_sent2kwd(df[colname_target], word_freq_tfidf, colname='Token_TF-IDF')
    if word_freq_keybert.shape[0] != 0:
        sent_kwd_keybert = preprocessing_sent2kwd(df[colname_target], word_freq_keybert, colname='Token_KeyBERT')
    sent_keyword = pd.concat([sent2kwd for sent2kwd in [sent_kwd_soynlp, sent_kwd_tfidf, sent_kwd_keybert] if sent2kwd.shape[0] != 0], axis=1)
    df_freq = pd.concat([df, sent_keyword], axis=1)

    # 저장
    print('Saving...:', datetime.datetime.now())
    if save_local:
        folder_location = os.path.join(os.getcwd(), 'Result', 'WordFreq', '')
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)
        ## 이름 정리
        if type(colname_category) == str:
            save_name = save_name.split('.')[0] + '_categ'
            save_name_df = 'df_prepcateg2_'+str(ngram_range).replace(' ','')+'.csv'
        else:
            save_name = save_name.split('.')[0]
            save_name_df = 'df_prep2_'+str(ngram_range).replace(' ','')+'.csv'
        ## 저장 정리
        if word_freq_soynlp.shape[0] != 0:
            word_freq_soynlp.to_csv(os.path.join(folder_location, save_name+'_soynlp.csv'), 
                                    index=False, encoding='utf-8-sig')
        if word_freq_tfidf.shape[0] != 0:
            word_freq_tfidf.to_csv(os.path.join(folder_location, save_name+'_tfidf.csv'), 
                                   index=False, encoding='utf-8-sig')
        if word_freq_keybert.shape[0] != 0:
            word_freq_keybert.to_csv(os.path.join(folder_location, save_name+'_keybert.csv'), 
                                     index=False, encoding='utf-8-sig')
        df_freq.to_csv(os.path.join(os.getcwd(), 'Data', save_name_df), index=False, encoding='utf-8-sig')
          
    # 출력정리
    print('Results Concat!:', datetime.datetime.now())
    ## 빈도 통계량 출력
    result_stat = pd.DataFrame([word_freq_soynlp.shape[0], word_freq_tfidf.shape[0], word_freq_keybert.shape[0]])
    result_stat.columns = ['Length']
    result_stat.index = ['Token by SoyNLP', 'Token by TF-IDF', 'Token by KeyBERT']
    display(result_stat.T)
    display(pd.concat([word_freq_soynlp.head(10), word_freq_tfidf.head(10), word_freq_keybert.head(10)], axis=1))
    ## dictionary로 저장
    word_freq = dict()
    word_freq['SoyNLP'], word_freq['TF-IDF'], word_freq['KeyBERT'] = word_freq_soynlp, word_freq_tfidf, word_freq_keybert

    return word_freq, df_freq


### Date and Author: 20240812, Kyungwon Kim ###
### word2vec 변환
def preprocessing_word2vec(df_tokenized, embedding_dim=100, word_window=5):
    # 문장 별 단어 분리
    df_split = [row.split(' ') for row in df_tokenized]

    # 학습
    ## sentences: 문장 토큰화된 데이터
    ## vector_size: 임베딩 할 벡터의 차원
    ## window: 현재값과 예측값 사이의 최대 거리
    ## min_count: 최소 빈도수 제한
    ## worker: 학습을 위한 thread의 수
    ## sg: {0: CBOW, 1: skip-gram}
    model = Word2Vec(sentences=df_split, vector_size=embedding_dim, 
                     window=word_window, min_count=0, workers=8, sg=0, sample=1e-3)

    # 단어 벡터 추출
    df_wordvec = pd.DataFrame(pd.Series({word:vec for word, vec in zip(model.wv.index_to_key, model.wv.vectors)}), 
                              columns=['word vector'])
    
    # 단어 상관관계
    colnames = list(df_wordvec.index)
    wordcorr = np.corrcoef(df_wordvec['word vector'].to_list())
    df_wordcorrpair = pd.DataFrame([(colnames[i], colnames[j], wordcorr[i,j]) 
                                     for i in range(wordcorr.shape[1]) for j in range(wordcorr.shape[1]) if i != j])
    df_wordcorrpair.columns = ['source', 'target', 'correlation']
    df_wordcorr = pd.DataFrame(wordcorr, index=colnames, columns=colnames)

    # 문장 벡터 추출
    sentvec = []
    for sentence in df_split:
        wvec = []
        for word in sentence:
            wvec.append(list(model.wv[word]))
        sentvec.append(wvec)
    df_sentvec = pd.DataFrame(pd.Series(sentvec), columns=['sentence vector'])
    
    return df_wordvec, df_wordcorr, df_wordcorrpair, df_sentvec


### Date and Author: 20250211, Kyungwon Kim ###
### frequency 기반으로 vector를 만들고 시각화를 위한 matrix 생성
### 주의: word2vec과 다를수 있음
### 일부씩 떼어 다시 만들고 제거예정
def freq2vectorcorr_preprocessor(df_series, df_wordfreq, relation_type='corr', num_showkeyword=100):
    # wordfreq to dict
    df_wordfreq = df_wordfreq.sort_values(by=df_wordfreq.columns[-1], ascending=False)[:num_showkeyword]
    dict_wordfreq = {row[0]:row[1] for row in df_wordfreq.values}
    dict_wordfreq = dict(sorted(dict_wordfreq.items()))
    
    # 텍스트 벡터 생성 함수
    def word2vec_preprocessor(dict_wordfreq, text):
        text_new = []
        for key in list(dict_wordfreq.keys()):
            if key in text.split(' '):
                text_new.append(float(dict_wordfreq[key]))
            else:
                text_new.append(0)

        return text_new

    # series to dataframe vector
    df_tokenized = df_series.apply(lambda x: word2vec_preprocessor(dict_wordfreq, x))
    df_wordvec = pd.DataFrame([row for row in df_tokenized.values], columns=list(dict_wordfreq.keys()))
    df_wordvec = df_wordvec.loc[:,df_wordvec.columns.isin(df_wordfreq[df_wordfreq.columns[0]].tolist())]
    colnames = list(df_wordvec.columns)
    
    # word correlation
    if relation_type == 'corr':
        wordcorr = abs(np.corrcoef(df_wordvec.T.values))
    elif relation_type == 'dot':
        wordcorr = np.dot(df_wordvec.T, df_wordvec)
    wordcorr[np.diag_indices_from(wordcorr)] = 0
    df_wordcorrpair = pd.DataFrame([(colnames[i], colnames[j], wordcorr[i,j]) 
                                     for i in range(wordcorr.shape[1]) for j in range(wordcorr.shape[1]) if i != j])
    df_wordcorrpair.columns = ['source', 'target', 'score']
    df_wordcorr = pd.DataFrame(wordcorr, index=colnames, columns=colnames) 
    
    # 정리
    df_wordcorr.fillna(0, inplace=True)
    df_wordcorrpair.fillna(0, inplace=True)
    
    return df_wordvec, df_wordcorr, df_wordcorrpair


### Date and Author: 20250204, Kyungwon Kim ###
### df를 분리하고 Dataset 라이브러리 형식으로 변환
def preprocessing_df2Datasets(model_pretrained, df_labeltext, 
                              label_list=[1,0], val_size=0.2, 
                              random_seed=123, max_length=256):
    # remane
    df_labeltext.columns = ['label', 'text']
    
    # train & test 분리
    ds_train = df_labeltext[df_labeltext['label'].isin(label_list)]
    ds_test = df_labeltext[~df_labeltext['label'].isin(label_list)]
    
    # 변환
    ds_train, ds_validation = train_test_split(ds_train, test_size=val_size, random_state=random_seed)
    ds_train = Dataset.from_list(ds_train.to_dict(orient='records'))
    ds_validation = Dataset.from_list(ds_validation.to_dict(orient='records'))
    ds_test = Dataset.from_list(ds_test.to_dict(orient='records'))
    ds = DatasetDict({'train': ds_train, 'validation': ds_validation, 'test': ds_test})

    # tokenizing
    tokenizer = AutoTokenizer.from_pretrained(model_pretrained)

    def apply_tokenizer(x):
        return tokenizer(x[list(ds['train'].features)[-1]], 
                         padding='max_length', max_length=max_length, truncation=True)
    
    ds_tokenized = ds.map(apply_tokenizer, batched=True)
    
    return ds_tokenized


# gm = preprocessing_gephi()
# gm.wordfreq_to_gephiinput(word_corrpair.iloc[:,1:], '.\Data\word_corrpair.graphml')
class preprocessing_gephi:
    def wordfreq_to_gephiinput(self, pair_file, graphml_file):
        out = open(graphml_file, 'w', encoding = 'utf-8')
        entity = []
        e_dict = {}
        count = []
        for i in range(len(pair_file)):
            e1 = pair_file.iloc[i,0]
            e2 = pair_file.iloc[i,1]
            #frq = ((word_dict[e1], word_dict[e2]),  pair.split('\t')[2])
            frq = ((e1, e2), pair_file.iloc[i,2])
            if frq not in count: count.append(frq)   # ((a, b), frq)
            if e1 not in entity: entity.append(e1)
            if e2 not in entity: entity.append(e2)
        print('# terms: %s'% len(entity))
        #create e_dict {entity: id} from entity
        for i, w in enumerate(entity):
            e_dict[w] = i + 1 # {word: id}
        out.write(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlnshttp://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">" +
            "<key id=\"d1\" for=\"edge\" attr.name=\"weight\" attr.type=\"double\"/>" +
            "<key id=\"d0\" for=\"node\" attr.name=\"label\" attr.type=\"string\"/>" +
            "<graph id=\"Entity\" edgedefault=\"undirected\">" + "\n")
        # nodes
        for i in entity:
            out.write("<node id=\"" + str(e_dict[i]) +"\">" + "\n")
            out.write("<data key=\"d0\">" + i + "</data>" + "\n")
            out.write("</node>")
        # edges
        for y in range(len(count)):
            out.write("<edge source=\"" + str(e_dict[count[y][0][0]]) + "\" target=\"" + str(e_dict[count[y][0][1]]) + "\">" + "\n")
            out.write("<data key=\"d1\">" + str(count[y][1]) + "</data>" + "\n")
            #out.write("<edge source=\"" + str(count[y][0][0]) + "\" target=\"" + str(count[y][0][1]) +"\">"+"\n")
            #out.write("<data key=\"d1\">" + str(count[y][1]) +"</data>"+"\n")
            out.write("</edge>")
        out.write("</graph> </graphml>")
        print('now you can see %s' % graphml_file)
        #pairs.close()
        out.close()

###################################################################################       
def preprocessing_wordfreq_to_corr(df_wordfreq, df, colname_target, colname_category=None, num_showkeyword=100, 
                                   save_local=True, save_name='word_corrpair.csv'):
    if colname_category == None:
        # 단어 벡터화 및 상관관계
        _, _, word_corrpair_total = freq2vectorcorr_preprocessor(df_wordfreq.iloc[:,-2:], 
                                                                 df[colname_target], num_showkeyword=num_showkeyword)
    elif type(colname_category) == str:
        word_corrpair_total = pd.DataFrame()
        for category in tqdm(sorted(df_wordfreq[df_wordfreq.columns[0]].unique())):
            # 데이터 분리
            wf_sub = df_wordfreq[df_wordfreq[df_wordfreq.columns[0]] == category]
            df_sub = df[df[colname_category] == category]
            
            # 단어 벡터화 및 상관관계
            if wf_sub.shape[0] > 1:
                _, _, word_corrpair = freq2vectorcorr_preprocessor(wf_sub.iloc[:,-2:], 
                                                                   df_sub[colname_target], num_showkeyword=num_showkeyword)
            else:
                continue

            ## 카테고리 추가
            word_corrpair['category'] = str(category)
            word_corrpair = word_corrpair[['category']+list(word_corrpair.columns[:-1])]
            word_corrpair_total = pd.concat([word_corrpair_total, word_corrpair], axis=0, ignore_index=True)
            
    # 저장
    if save_local:
        folder_location = os.path.join(os.getcwd(), 'Data', 'WordCorr', '')
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)
        save_name = os.path.join(folder_location, save_name)
        word_corrpair_total.to_csv(save_name, index=False, encoding='utf-8-sig')

    return word_corrpair_total
###################################################################################        



# 문장에서 통계수치 전후문장 추출 함수
def statsentence_extractor(extranct_rule, sentences, window=2):
    kiwi = Kiwi()
    sent_list = [sent[0].strip() for sent in kiwi.split_into_sents(sentences)]
    sent_short = []
    for idx, sent in enumerate(sent_list):
        import re
        if re.search('[0-9]+\.?[0-9]+[%건명]', sent) != None:
            sent_short.extend(['...'])
            sent_short.extend(sent_list[idx-window:idx+window+1])
            sent_short.extend(['...'])
    sent_short = ' '.join(sent_short)
    
    return sent_short
     
    