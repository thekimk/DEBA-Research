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
#     df_concat = df_concat.drop(['공급좌석수', '승차수입금액', '승차연인거리', '좌석거리', '열차운행횟수'], axis = 1)
    df_concat = df_concat.drop(['공급좌석수', '승차수입금액', '승차연인거리', '좌석거리'], axis = 1)

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
    df_concat = feature_timeinfo(df_concat, colname_time='운행일자', weekend_dow=['금', '토', '일'])
    
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
    df_concat = pd.merge(df_concat, df_covid, left_on='운행일자', right_on='date', how='left').reset_index().iloc[:,1:]
    df_concat = df_concat.sort_values(by=['주운행선', '운행일자'])
    del df_concat['date']
    
    # 데이터 추가
    # 24년 4~6월 승차인원수 추가
    df_y4 = pd.read_excel(os.path.join('.', 'Data', 'df_KTX_addition.xlsx'), sheet_name='승차인원수4월추출', header=2).dropna()
    df_y4['운행일자'] = pd.to_datetime(df_y4['운행년월일'], format='%Y년 %m월 %d일')
    del df_y4['운행년월일']
    df_y4 = df_y4[df_y4['운행일자'] >= '2024-04'].reset_index().iloc[:,1:]
    df_y56 = pd.read_excel(os.path.join('.', 'Data', 'df_KTX_addition.xlsx'), sheet_name='승차인원수5월6월추출').dropna()
    df_y56['운행일자'] = pd.to_datetime(df_y56['운행일자'], format='%Y년 %m월 %d일')
    df_y56 = df_y56.set_index('운행일자').stack().reset_index()
    df_y56.columns = ['운행일자', '주운행선', '합계 : 승차인원(합계)']
    df_y = pd.concat([df_y4, df_y56], axis=0).reset_index().iloc[:,1:]
    df_y.columns = ['주운행선', '승차인원수', '운행일자']
    df_y = df_y.sort_values(by=['주운행선', '운행일자'])
    ## 원본 데이터에 결합
    df_sub = df_concat[(df_concat['운행일자'] >= df_y['운행일자'].min()) & (df_concat['운행일자'] <= df_y['운행일자'].max())]
    df_y.index = df_sub.index
    df_update = df_concat.combine_first(df_y)
    df_concat = df_update[df_concat.columns]
    # 24년 4월 공급좌석합계수 추가
    df_supply4 = pd.read_excel(os.path.join('.', 'Data', 'df_KTX_addition.xlsx'), sheet_name='공급좌석수4월추출', header=2).dropna()
    df_supply4['운행일자'] = pd.to_datetime(df_supply4['운행년월일'], format='%Y년 %m월 %d일')
    del df_supply4['운행년월일']
    df_supply4 = df_supply4[df_supply4['운행일자'] >= '2024-04'].reset_index().iloc[:,1:]
    df_supply4.columns = ['주운행선', '공급좌석합계수', '운행일자']
    df_supply4 = df_supply4.sort_values(by=['주운행선', '운행일자'])
    ## 원본 데이터에 결합
    df_sub = df_concat[(df_concat['운행일자'] >= df_supply4['운행일자'].min()) & (df_concat['운행일자'] <= df_supply4['운행일자'].max())]
    df_supply4.index = df_sub.index
    df_update = df_concat.combine_first(df_supply4)
    df_concat = df_update[df_concat.columns]
    
    # 월별 집계
    year_month_day = df_concat['운행일자']
    del df_concat['운행일자']
    df_monthsum = df_concat.groupby(['주운행선', '운행년월']).sum()
    df_monthsum = df_monthsum[[col for col in df_monthsum.columns if col != '전체주중주말']].reset_index()
    df_monthsum['전체주중주말'] = '전체'
    df_temp = df_concat.groupby(['전체주중주말', '주운행선', '운행년월']).sum().reset_index()
    df_monthsum = df_monthsum[df_temp.columns]
    df_monthsum = pd.concat([df_monthsum, df_temp], axis=0).reset_index().iloc[:,1:].fillna(0)
    ## 정리
    df_concat = pd.concat([year_month_day, df_concat], axis=1)
    df_monthsum = df_monthsum[['전체주중주말', '주운행선', '운행년월', '일수', '주말수', '주중수', '공휴일수', '명절수'] + [col for col in df_monthsum.columns if col not in ['전체주중주말', '주운행선', '운행년월', '일수', '주말수', '주중수', '공휴일수', '명절수']]]
    
    # 도메인지식 첨부
    ## 공급좌석합계수
    ## 기존 24년 4월 데이터만 추출 하여 24년 5월 ~ 25년 12월까지 데이터를 덮어씀
    for wom in df_monthsum['전체주중주말'].unique():
        for line in df_monthsum['주운행선'].unique():
            df_sub = df_monthsum[(df_monthsum['전체주중주말'] == wom) & (df_monthsum['주운행선'] == line) & (df_monthsum['운행년월'] >= '2024-04-01')]
            numsit = df_sub[(df_sub['전체주중주말'] == wom) & (df_sub['주운행선'] == line)]['공급좌석합계수'].values[0]
            ## 5월부터 경부선 주말 4120석 증가, 주중 1030석 증가 및 호남 주중 1030석 증가
            if (line == '경부선') and (wom == '주말'):
                numsit = numsit + 4120
            elif (line == '경부선') and (wom == '주중'):
                numsit = numsit + 1030
            elif (line == '경부선') and (wom == '전체'):
                numsit = numsit + 4120 + 1030
            elif (line == '호남선') and (wom == '주중'):
                numsit = numsit + 1030
            elif (line == '호남선') and (wom == '전체'):
                numsit = numsit + 1030
            ## 덮어씀
            df_monthsum.loc[df_sub.index[1:], '공급좌석합계수'] = df_monthsum.loc[df_sub.index[1:], '공급좌석합계수'] + numsit
    ## 승차율
    ## 23년 3월까지와 24년 3월까지의 승차율 증감 테이블 생성
    df_compare = df_monthsum[(df_monthsum['운행년월'] >= '2023-01-01') & (df_monthsum['운행년월'] <= '2024-03-01')]
    df_compare = df_compare[(df_compare['운행년월'] <= '2023-03-01') | (df_compare['운행년월'] >= '2024-01-01')]
    df_compare['운행년월'] = df_compare['운행년월'].apply(lambda x: str(x)[:4])
    df_compare = df_compare.groupby(['전체주중주말', '주운행선', '운행년월'])['승차율'].mean().reset_index()
    df_target = df_compare[df_compare['운행년월'] == '2023']
    df_base = df_compare[df_compare['운행년월'] == '2024']
    df_compare['승차율증감'] = df_base['승차율'] - df_target['승차율'].values
    df_compare = df_compare.dropna()
    del df_compare['승차율']
    df_compare.sort_values(by=['전체주중주말', '주운행선'])
    ## 24년 4월~12월 승차율 = 23년 4월~12월의 승차율 + 증감
    df_target = df_monthsum[(df_monthsum['운행년월'] >= '2024-04-01') & (df_monthsum['운행년월'] <= '2024-12-01')]
    df_base = df_monthsum[(df_monthsum['운행년월'] >= '2023-04-01') & (df_monthsum['운행년월'] <= '2023-12-01')]
    for wom in df_target['전체주중주말'].unique():
        for line in df_target['주운행선'].unique():
            df_subt = df_target[(df_target['전체주중주말'] == wom) & (df_target['주운행선'] == line)]
            df_subb = df_base[(df_base['전체주중주말'] == wom) & (df_base['주운행선'] == line)]
            updown = df_compare[(df_compare['전체주중주말'] == wom) & (df_compare['주운행선'] == line)]['승차율증감'].values
            df_monthsum.loc[df_subt.index, '승차율'] = df_subb['승차율'].values + updown
    ## 25년 1월~12월 승차율 = 24년 1월~12월의 승차율
    df_target = df_monthsum[(df_monthsum['운행년월'] >= '2025-01-01') & (df_monthsum['운행년월'] <= '2025-12-01')]
    df_base = df_monthsum[(df_monthsum['운행년월'] >= '2024-01-01') & (df_monthsum['운행년월'] <= '2024-12-01')]
    for wom in df_target['전체주중주말'].unique():
        for line in df_target['주운행선'].unique():
            df_subt = df_target[(df_target['전체주중주말'] == wom) & (df_target['주운행선'] == line)]
            df_subb = df_base[(df_base['전체주중주말'] == wom) & (df_base['주운행선'] == line)]
            df_monthsum.loc[df_subt.index, '승차율'] = df_subb['승차율'].values
    ## 일반노선 승차인원수
    df_demand_normal = pd.read_excel(os.path.join('.', 'Data', 'df_KTX_addition.xlsx'), sheet_name='승차인원수일반')
    ## 노선필터
    df_demand_normal = df_demand_normal[df_demand_normal['주운행선'].isin(['경부선', '경전선', '동해선', '전라선', '호남선'])].reset_index().iloc[:,1:]
    ## 시간정보추출
    df_demand_normal['운행일자'] = pd.to_datetime(df_demand_normal['운행일자'], format='%Y년 %m월 %d일')
    df_demand_normal['운행년월'] = pd.to_datetime(df_demand_normal['운행일자'].apply(lambda x: str(x)[:7]))
    df_demand_normal['요일'] = df_demand_normal['운행일자'].dt.weekday
    weekday_list = ['월', '화', '수', '목', '금', '토', '일']
    df_demand_normal['요일'] = df_demand_normal.apply(lambda x: weekday_list[x['요일']], axis=1)
    df_demand_normal['전체주중주말'] = df_demand_normal['요일'].apply(lambda x: '주말' if x in ['금', '토', '일'] else '주중')
    df_demand_normal = df_demand_normal.drop(['운행일자', '요일'], axis = 1)
    df_demand_normal = df_demand_normal[['전체주중주말', '주운행선', '역무열차종', '운행년월', '승차인원수']]
    ## 전체주중주말 정리
    df_temp_all = df_demand_normal[['주운행선', '역무열차종', '운행년월', '승차인원수']].groupby(['주운행선', '역무열차종', '운행년월']).sum().reset_index()
    df_temp_all['전체주중주말'] = '전체'
    df_temp = df_demand_normal.groupby(['전체주중주말', '주운행선', '역무열차종', '운행년월']).sum().reset_index()
    df_demand_normal = pd.concat([df_temp, df_temp_all], axis=0).reset_index().iloc[:,1:]
    ## 일반노선별 정리
    df_demand_normal = df_demand_normal.set_index(['전체주중주말', '주운행선', '역무열차종', '운행년월']).unstack(level=2).reset_index().fillna(0)
    df_demand_normal.columns = ['전체주중주말', '주운행선', '운행년월'] + list(df_demand_normal.columns.droplevel(0)[3:])
    ## 예측기간 확장 및 승차인원수 확장
    df_temp = pd.DataFrame()
    for dow in df_demand_normal['전체주중주말'].unique():
        for line in df_demand_normal['주운행선'].unique():
            df_sub = df_demand_normal[(df_demand_normal['전체주중주말'] == dow) & (df_demand_normal['주운행선'] == line)]
            df_time = pd.DataFrame(pd.date_range(df_demand_normal['운행년월'].max(), '2025-12-01', freq='MS')).iloc[1:,:].reset_index().iloc[:,1:]
            df_time.columns = ['운행년월']
            df_time['주운행선'] = line
            df_time['전체주중주말'] = dow
            df_expand = pd.DataFrame([list(df_sub.iloc[-1,3:].values) for i in range(df_time.shape[0])])
            df_expand.columns = df_demand_normal.columns[3:]
            df_expand = pd.concat([df_time, df_expand], axis=1)
            df_sub = pd.concat([df_sub, df_expand], axis=0).reset_index().iloc[:,1:]
            df_temp = pd.concat([df_temp, df_sub], axis=0)
    df_demand_normal = df_temp.copy()
    df_demand_normal = df_demand_normal.reset_index().iloc[:,1:]
    ## 결합
    df_monthsum = pd.merge(df_monthsum, df_demand_normal, how='inner', on=['전체주중주말', '주운행선', '운행년월'])
    
    # 저장
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


def feature_timeinfo(df, colname_time, weekend_dow=['토', '일']):
    # 월집계용 변수생성
    df_time = df.copy()
    df_time['운행년월'] = pd.to_datetime(df_time[colname_time].apply(lambda x: str(x)[:7]))
    
    # 일수 반영
    df_time['일수'] = 1
    
    # 요일 추출
    df_time['요일'] = df_time[colname_time].dt.weekday
    weekday_list = ['월', '화', '수', '목', '금', '토', '일']
    df_time['요일'] = df_time.apply(lambda x: weekday_list[x['요일']], axis=1)
    
    # 공휴일 추출
    df_time['공휴일수'] = df_time[colname_time].apply(lambda x: is_holiday(str(x)[:10]))*1
    
    # 명절 추출
    traditional_holidays = []
    for year in df_time[colname_time].dt.year.unique():
        for holiday, holiday_name in year_holidays(str(year)):
            if ('설날' in holiday_name) or ('추석' in holiday_name):
                traditional_holidays.append(holiday)
    traditional_holidays = pd.to_datetime(traditional_holidays, format='%Y년 %m월 %d일')
#     traditional_holidays = [t.strftime("%Y년 %m월 %d일") for t in traditional_holidays]
    df_time['명절수'] = df_time[colname_time].apply(lambda x: 1 if x in traditional_holidays else 0)
    
    # 주말/주중 추출
    df_time['전체주중주말'] = df_time['요일'].apply(lambda x: '주말' if x in weekend_dow else '주중')
    weekday_dow = [each for each in ['월', '화', '수', '목', '금', '토', '일'] if each not in weekend_dow]
    df_time['주중수'] = df_time['요일'].isin(weekday_dow)*1
    df_time['주말수'] = df_time['요일'].isin(weekend_dow)*1
    df_time['주말수'] = df_time['주말수'] + np.array([1 if np.sum(each) == 2 else 0 for each in zip(df_time['주중수'], df_time['공휴일수'])])
    df_time['주중수'] = df_time['주중수'] + np.array([-1 if np.sum(each) == 2 else 0 for each in zip(df_time['주중수'], df_time['공휴일수'])])
    del df_time['요일']
    
    return df_time


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


def evaluation_ktx(target_line, target_dow, Y_real, Y_pred, year_prediction, year_comparison):
    # 실제와 예측평균 추출
    Y_pred = Y_pred[Y_pred.index >= year_prediction[0]]
    Y_pred = Y_pred[[col for col in Y_pred.columns[2:] if col.split('-')[-1].isalpha()]].mean(axis=0)
    Y_temp = []
    for year in year_comparison:
        Y_temp.append(Y_real[(Y_real.index >= year) & (Y_real != 0)].mean())
    # 통계량비교 정리
    Y_eval = pd.DataFrame([sum([Y_temp, [pred], list((pred / np.array(Y_temp) - 1)*100)], []) for pred in Y_pred])
    Y_eval.columns = [year_comparison[0]+'년', year_comparison[1]+'년', year_prediction[0]+'년', 
                      '증감율%('+str(year_comparison[0])+'-'+str(year_prediction[0])+')', 
                      '증감율%('+str(year_comparison[1])+'-'+str(year_prediction[0])+')']
    Y_eval['주운행선'] = target_line
    Y_eval['전체주중주말'] = target_dow
    Y_eval['알고리즘순위'] = list(Y_pred.index)
    Y_eval = Y_eval.set_index(['주운행선', '전체주중주말'])
    
    return Y_eval