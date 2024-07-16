# Ignore the warnings
import warnings
# warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_pandas # execution time
tqdm.pandas()
from holidayskr import year_holidays, is_holiday
from covid19dh import covid19



def get_data_from_ktx(date_max='2025-12-31'):
    # 데이터 로딩
    df_demand1 = pd.read_excel(os.path.join(os.getcwd(), 'Data', '(간선)수송-운행일-주운행(201501-202305).xlsx'), skiprows=5)
    df_demand2 = pd.read_excel(os.path.join(os.getcwd(), 'Data', '(간선)수송-운행일-주운행(202305-202403).xlsx'), skiprows=5)
    duplicate = list(set(df_demand1['운행년월']).intersection(set(df_demand2['운행년월'])))[0]
    df_demand1 = df_demand1[df_demand1['운행년월'] != duplicate]
    df_info1 = pd.read_excel(os.path.join(os.getcwd(), 'Data', '(간선)시종착역별 열차운행(201501-202305).xlsx'), skiprows=8)
    df_info2 = pd.read_excel(os.path.join(os.getcwd(), 'Data', '(간선)시종착역별 열차운행(202305-202403).xlsx'), skiprows=8)
    duplicate = list(set(df_info1['운행일자'].apply(lambda x:x[:9])).intersection(set(df_info2['운행일자'].apply(lambda x:x[:9]))))[0]
    df_info1 = df_info1[df_info1['운행일자'].apply(lambda x: x[:9] != duplicate)]
    df_demand = pd.concat([df_demand1, df_demand2], axis=0)
    df_info = pd.concat([df_info1, df_info2], axis=0)
                
    # 분석대상 필터
    ## 역무열차종: KTX
    ## 주운행선: '경부선', '경전선', '동해선', '전라선', '호남선'
    df_demand = df_demand[df_demand['역무열차종'].apply(lambda x: x[:3] == 'KTX')].reset_index().iloc[:,1:]
    df_info = df_info[df_info['역무열차종'].apply(lambda x: x[:3] == 'KTX')].reset_index().iloc[:,1:]
    df_demand = df_demand[df_demand['주운행선'].isin(['경부선', '경전선', '동해선', '전라선', '호남선'])].reset_index().iloc[:,1:]
    df_info = df_info[df_info['주운행선'].isin(['경부선', '경전선', '동해선', '전라선', '호남선'])].reset_index().iloc[:,1:]

    # 불필요 변수 삭제
    df_demand.drop(columns=['Unnamed: 1', '운행년도', '운행년월', '운행요일구분', '역무열차종', '메트릭'], inplace=True)
    df_info.drop(columns=['상행하행구분', '역무열차종', '운행요일구분', '메트릭'], inplace=True)
    df_demand = df_demand.reset_index().iloc[:,1:]
    df_info = df_info.reset_index().iloc[:,1:]
    
    # 일별 집계 및 변수생성
    df_demand = df_demand.groupby(['주운행선', '운행일자']).sum().reset_index()
    df_demand['1인당단가'] = df_demand['승차수입금액']/df_demand['승차인원수']
    df_demand['1인당거리'] = df_demand['승차연인거리']/df_demand['승차인원수']
    df_demand['1좌석당단가'] = df_demand['승차수입금액']/df_demand['공급좌석합계수']
    df_demand['좌석회전율'] = df_demand['승차인원수']/df_demand['공급좌석합계수']
    df_demand['1키로당단가'] = df_demand['승차수입금액']/df_demand['승차연인거리']
    df_demand['승차율'] = df_demand['승차연인거리']/df_demand['좌석거리']
    df_info['시발종착역'] = df_info['시발역']+df_info['종착역']
    df_info = pd.concat([df_info.groupby(['주운행선', '운행일자'])['열차속성'].value_counts().unstack().reset_index(),
                         df_info.groupby(['주운행선', '운행일자'])['열차구분'].value_counts().unstack().reset_index().iloc[:,-3:],
                         df_info.groupby(['주운행선', '운행일자'])['시발역'].nunique().reset_index().iloc[:,-1],
                         df_info.groupby(['주운행선', '운행일자'])['종착역'].nunique().reset_index().iloc[:,-1],
                         df_info.groupby(['주운행선', '운행일자'])['시발종착역'].nunique().reset_index().iloc[:,-1],
                         df_info.groupby(['주운행선', '운행일자'])[['공급좌석수', '열차운행횟수']].sum().reset_index().iloc[:,-2:]], axis=1)
    df_concat = pd.merge(df_demand, df_info, how='inner', on=['주운행선', '운행일자'])
    df_concat['1열차당승차인원'] = df_concat['승차인원수']/df_concat['열차운행횟수']
    del df_concat['공급좌석수']

    # 예측기간 확장
    ## 시간변수 정의
    df_concat['운행일자'] = pd.to_datetime(df_concat['운행일자'], format='%Y년 %m월 %d일')
    ## 예측 시계열 생성
    df_time = pd.DataFrame(pd.date_range(df_concat['운행일자'].min(), date_max, freq='D'))
    df_time.columns = ['운행일자']
    ## left 데이터 준비   
    df_temp = df_concat.groupby(['주운행선', '운행일자']).sum().reset_index()
    df_concat = pd.DataFrame()
    for line in df_temp['주운행선'].unique():
        df_sub = df_temp[df_temp['주운행선'] == line]
        ## 결합
        df_concat_temp = pd.merge(df_sub, df_time, left_on='운행일자', right_on='운행일자', how='outer')
        df_concat_temp['주운행선'].fillna(line, inplace=True)
        df_concat = pd.concat([df_concat, df_concat_temp], axis=0)
    
    # 시간변수 추출
    ## 월집계용 변수생성
    df_concat['운행년월'] = pd.to_datetime(df_concat['운행일자'].apply(lambda x: str(x)[:7]))
    ## 요일 추출
    df_concat['요일'] = df_concat['운행일자'].dt.weekday
    weekday_list = ['월', '화', '수', '목', '금', '토', '일']
    df_concat['요일'] = df_concat.apply(lambda x: weekday_list[x['요일']], axis=1)
    ## 주말/주중 추출
    df_concat['일수'] = 1
    df_concat['전체주중주말'] = df_concat['요일'].apply(lambda x: '주말' if x in ['금', '토', '일'] else '주중')
    df_concat['주말수'] = df_concat['요일'].isin(['금', '토', '일'])*1
    df_concat['주중수'] = df_concat['요일'].isin(['월', '화', '수', '목'])*1
    del df_concat['요일']
    ## 공휴일 추출
    df_concat['공휴일수'] = df_concat['운행일자'].apply(lambda x: is_holiday(str(x)[:10]))*1
    ## 명절 추출
    traditional_holidays = []
    for year in df_concat['운행일자'].dt.year.unique():
        for holiday, holiday_name in year_holidays(str(year)):
            if ('설날' in holiday_name) or ('추석' in holiday_name):
                traditional_holidays.append(holiday)
    traditional_holidays = pd.to_datetime(traditional_holidays, format='%Y년 %m월 %d일')
#     traditional_holidays = [t.strftime("%Y년 %m월 %d일") for t in traditional_holidays]
    df_concat['명절수'] = df_concat['운행일자'].apply(lambda x: 1 if x in traditional_holidays else 0)
    
    # Covid 데이터 결합
    ## Covid 데이터 전처리
    df_covid, src = covid19('KOR', verbose=False) 
    df_covid.date = pd.to_datetime(df_covid.date)
    time_covid = df_covid[~df_covid.confirmed.isnull()].date
    df_covid = df_covid[~df_covid.confirmed.isnull()]
    df_covid = df_covid[df_covid.columns[df_covid.dtypes == 'float64']].reset_index().iloc[:,1:]
    df_covid.dropna(axis=1, how='all', inplace=True)
    df_covid.fillna(0, inplace=True)
    ## 종속변수와의 관련도 높은 변수 필터
    feature_Yrelated = []
    df_Y = df_concat[df_concat['운행일자'].apply(lambda x: x in time_covid.values)]
    for line in df_concat['주운행선'].unique():
        Y = df_Y[df_Y['주운행선'] == line]['승차인원수'].reset_index().iloc[:,1:]
        corr = abs(pd.concat([Y, df_covid], axis=1).corr().iloc[:,[0]]).dropna()
        corr = corr.sort_values(by='승차인원수', ascending=False)
        feature_Yrelated.extend([i for i in corr[corr>0.5].dropna().index if i != corr.columns])
    Y_related_max = np.max([feature_Yrelated.count(x) for x in set(feature_Yrelated)])
    feature_Yrelated = [x for x in set(feature_Yrelated) if feature_Yrelated.count(x) == Y_related_max]
    df_covid = pd.concat([time_covid.reset_index().iloc[:,1:], df_covid[feature_Yrelated]], axis=1)
    ## 변수명변경
    df_covid.rename(columns={'stringency_index':'코로나진행정도', 'government_response_index':'정부대응정도',
                             'international_movement_restrictions':'국가이동제한정도', 'deaths':'사망자수',
                             'people_vaccinated':'접종시작자수', 'people_fully_vaccinated':'접종완료자수', 
                             'containment_health_index':'격리된자수','confirmed':'확진자수'}, inplace=True)
    ## 정리
    df_concat = pd.merge(df_concat, df_covid, left_on='운행일자', right_on='date', how='left')
    del df_concat['date']
      
    # 월별 집계
    year_month_day = df_concat['운행일자']
    del df_concat['운행일자']
    df_monthsum = df_concat.groupby(['주운행선', '운행년월']).sum()
    df_monthsum = df_monthsum[[col for col in df_monthsum.columns if col != '전체주중주말']].reset_index()
    df_monthsum['전체주중주말'] = '전체'
    df_temp = df_concat.groupby(['전체주중주말', '주운행선', '운행년월']).sum().reset_index()
    df_monthsum = df_monthsum[df_temp.columns]
    df_monthsum = pd.concat([df_monthsum, df_temp], axis=0).fillna(0)    
    ## 정리
    df_concat = pd.concat([year_month_day, df_concat], axis=1)
    df_monthsum = df_monthsum[['전체주중주말', '주운행선', '운행년월', '일수', '주말수', '주중수', '공휴일수', '명절수'] + [col for col in df_monthsum.columns if col not in ['전체주중주말', '주운행선', '운행년월', '일수', '주말수', '주중수', '공휴일수', '명절수']]]
    ## 저장
    folder_location = os.path.join(os.getcwd(), 'Data', '')
    if not os.path.exists(folder_location):
        os.makedirs(folder_location)
    save_name = os.path.join(folder_location, 'df_KTX_monthsum_KK.csv')
    df_monthsum.to_csv(save_name, encoding='utf-8-sig')
    
    # 일평균변환
    df_monthmean = []
    for each in df_monthsum.values:
        if each[0] == '전체':
            each_day = np.append(each[:8], (each[8:] / each[3]))
        elif each[0] == '주말':
            each_day = np.append(each[:8], (each[8:] / each[4]))
        elif each[0] == '주중':
            each_day = np.append(each[:8], (each[8:] / each[5]))
        df_monthmean.append(each_day)
    df_monthmean = pd.DataFrame(df_monthmean)
    df_monthmean.columns = df_monthsum.columns
    ## 저장
    folder_location = os.path.join(os.getcwd(), 'Data', '')
    if not os.path.exists(folder_location):
        os.makedirs(folder_location)
    save_name = os.path.join(folder_location, 'df_KTX_monthmean_KK.csv')
    df_monthmean.to_csv(save_name, encoding='utf-8-sig')    

    return df_monthsum, df_monthmean


def feature_lagging(df, colname, direction='downward', lag_length=1):
    # 변수명 생성
    colname_lag = [colname+'_lag'+str(i+1) for i in range(lag_length)]
    
    # lag 생성
    df_lag = pd.DataFrame()
    for i in range(lag_length):
        if direction == 'downward':
            df_lag = pd.concat([df_lag, df[colname].shift(i+1)], axis=1)
        elif direction == 'upward':
            df_lag = pd.concat([df_lag, df[colname].shift(-i-1)], axis=1)
    df_lag.columns = colname_lag
    
    # 결측치 처리
    df_lag.fillna(method='bfill', inplace=True)
    df_lag.fillna(method='ffill', inplace=True)
    
    # 결합 및 정리
    df_lag.index = df.index.copy()
    df = pd.concat([df, df_lag], axis=1)
    
    return df, colname_lag


def preprocessing_ktx(df_raw, Y_colname, X_delete=None,
                      lag_length=None, lag_direction='downward',
                      date_splits=None):
    # 변수삭제
    if X_delete != None:
        df = df_raw[[col for col in df_raw.columns if col not in X_delete]]
    else:
        df = df_raw.copy()
    
    # 시간인덱스
    df['운행년월'] = pd.to_datetime(df['운행년월'])
    df = df.set_index('운행년월')
    
    # 지연값
    if lag_length != None:
        df, _ = feature_lagging(df, colname=Y_colname, lag_length=lag_length, direction=lag_direction)
    
    # 변수구분
    df_validate = pd.DataFrame()
    if date_splits == None:
        df_train, df_test = df.iloc[:-int(df.shape[0] * 0.2),:], df.iloc[-int(df.shape[0] * 0.2):,:]
    elif len(date_splits) == 1:
        df_train = df[df.index <= date_splits[0]]
        df_test = df[(df.index > date_splits[0])]
    elif len(date_splits) == 2:
        df_train = df[df.index <= date_splits[0]]
        df_validate = df[(df.index > date_splits[0]) & (df.index <= date_splits[1])]
        df_test = df[(df.index > date_splits[1])]
        
    return df_train, df_validate, df_test, list(df_train.columns)