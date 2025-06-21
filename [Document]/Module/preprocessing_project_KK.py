import os
import pandas as pd
import numpy as np
import datetime
import itertools
from tqdm import tqdm
import time
import datetime
import random
import matplotlib.pyplot as plt
import missingno as msno
plt.style.use('default')
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, CondensedNearestNeighbour, OneSidedSelection
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import KMeans

# from antropy import detrended_fluctuation, higuchi_fd, katz_fd, petrosian_fd
# from antropy import lziv_complexity, app_entropy, perm_entropy, sample_entropy, spectral_entropy, svd_entropy
from pyentrp.entropy import shannon_entropy, multiscale_entropy, permutation_entropy, multiscale_permutation_entropy, composite_multiscale_entropy
# from EntroPy import EntroPy

from preprocessing_KK import *
from description_KK import *
from visualization_KK import *


### Date and Author: 20240103, Kyungwon Kim ### 
def preprocessing_koweps(df_raw, year_min=None, year_max=None, 
                         age_min=None, age_max=None, gender='all',
                         X_reverse=None, X_delete=None,
                         Y_colname='IPV',
                         test_size=0.2, class_stat=True, 
                         sampling_method=None, sampling_strategy='auto',
                         scaler='minmax', random_state=123,
                         label_list = ['0', '1']):
    start = time.time()
    df = df_raw.copy()
    print('\nPreprocessing for Data Prepare...')
    print('Initial Shape: ', df.shape)
    
    # 데이터 필터
    ## 연도 필터
    if year_min == None: year_min = df.year.unique().min()
    if year_max == None: year_max = df.year.unique().max()
    df = df[df.year >= year_min]
    df = df[df.year <= year_max]
    print('After Year Filtering: ', df.shape)
    ## 나이 필터
    if age_min == None: age_min = (df.year - df.h_g4).unique().min()
    if age_max == None: age_max = (df.year - df.h_g4).unique().max()
    df = df[(df.year - df.h_g4) >= age_min]
    df = df[(df.year - df.h_g4) <= age_max]
    print('After Age Filtering: ', df.shape)
    ## 성별 필터
    if gender == 'male':
        df = df[df.h_g3 == 1]
        print('After Gender Filtering: ', df.shape)
    elif gender == 'female':
        df = df[df.h_g3 == 2]
        print('After Gender Filtering: ', df.shape)
        
    # 종속변수 후보군 처리
    print('\nPreprocessing of Y...')
    if Y_colname == 'IPV':
        df = df[df[Y_colname] >= 0].reset_index().iloc[:,1:]
        df[Y_colname] = df[Y_colname].apply(lambda x: 0 if x <= 1 else 1)
    elif Y_colname == 'p05_7aq1':    
        df[Y_colname].replace(9, 1, inplace=True)
        df[Y_colname].replace(2, 0, inplace=True)
        df[Y_colname].fillna(0, inplace=True)
#         df = df.dropna(axis=0)
        df[Y_colname] = df[Y_colname].astype(int)
        ## 자살관련변수 삭제
        col_delete = ['p05_6aq2','p05_6aq3','p05_6aq4','p05_6aq5','p05_6aq6','p05_6aq7','p05_6aq8','p05_6aq9','p05_7aq2','p05_7aq3']
        df = df[[col for col in df.columns if col not in col_delete]]
    elif Y_colname in ['p04_4', 'p04_5','p04_6']:
        if Y_colname == 'p04_4':
            ## 1 == 1, 2 == 0, 9 == 0, nan == drop
            df = df[df[Y_colname].notna()]
            df[Y_colname] = df[Y_colname].apply(lambda x: 1 if x == 1 else 0)
            ## 1 == 1, 2 == 0, 9 == 0, nan == 0
#             df[Y_colname] = df[Y_colname].apply(lambda x: 1 if x == 1 else 0)
        else:
            ## 1만원 또는 1회보다 크면 1, nan == 0, 0 == drop
            df['p04_5'].replace(np.nan, -1, inplace=True)
            df = df[df['p04_5'] != 0]
            df['p04_5'] = df['p04_5'].apply(lambda x: 1 if x>= 1 else 0)
            ## 고액기부자 예측시 특정값보다 크면1 아니면 0, 0 == drop, nan == drop
#             df = df[df['p04_5'] != 0]
#             df = df[df['p04_5'].notna()]
#             df['p04_5'] = df['p04_5'].apply(lambda x: 1 if x>= 1000 else 0)
        ## 기부관련변수 삭제
        col_delete = ['p04_4', 'p04_5','p04_6']
        df = df[[Y_colname]+[col for col in df.columns if col not in col_delete]]
    print('Ratio of Origin Y: ', df[[Y_colname]].value_counts())
    print('Shape of Data: ', df.shape)
#     display(df[df[Y_colname]==1]['Spousal relationship satisfaction'].value_counts())
      
    # 결측치 처리
    print('\nPreprocessing NAN...')
    df_X = null_filling(df[[i for i in df.columns if i != Y_colname]])
    df = pd.concat([df[Y_colname], df_X], axis=1)
    
    # 불필요 변수 삭제
    print('\nDeleting Non-meaning Variables...')
    col_delete = ['h_merkey', 'h_pid', 'wv', 'wv_num', 'first_wv', 'last_wv', 'p_wsl',  'p_wsc',  'p_wgl',  'p_wgc',
                  'p_wsl_all', 'p_wsc_all', 'p_wgl_all', 'p_wgc_all', 'p_wsl_n_all', 'p_wsc_n_all', 
                  'p_wgl_n_all', 'p_wgc_n_all', 'h_id', 'h_ind', 'h_sn', 'h_new', 'h_new1', 'h_pind', 'h_pid',
                  'h_flag', 'h_hc_all', 'h_hc_n_all', 'nh01_1', 'nh01_2', 'h_med1', 'h_eco1', 'h_g2', 'h_soc1', 'h_inc1',
                  'h_inc7_3', 'p_fnum', 'p_tq', 'p_cp', 'wc_fnum', 'wv_cp', 'h13_4aq15', 'h17_2', 'release_date']
    if X_delete != None:
        col_delete.extend(X_delete)
    df = df[[col for col in df.columns if col not in col_delete]]
    print('Shape of Data: ', df.shape)
    ## 전체 삭제변수 저장
    X_delete = [col for col in df_raw.columns if col not in list(df.columns)]
    X_rename = pd.read_excel(os.path.join(os.getcwd(), 'Result', '[Result_20240104].xlsx'), sheet_name='CodeBook')
    X_delete = pd.DataFrame.from_dict({val[0]:val[1] for val in X_rename[['이름', '레이블']].values if val[0] in X_delete},
                                      orient='index').reset_index()
    X_delete.columns = ['Deleted Feature', 'Description']
    X_delete.to_csv(os.path.join(os.getcwd(), 'Result', 'DeletedVariables.csv'), index=False, encoding='utf-8-sig')
    
    # 독립변수 문자열 처리
    colname_object = list(df.columns[df.dtypes == 'object'])
    print('\nObject X Features: ', colname_object)
    encoder = LabelEncoder()
    for col in colname_object:
        df[col] = encoder.fit_transform(df[col])
    print('Complete X Object Processing!')
    
    sec = time.time()-start
    print(str(datetime.timedelta(seconds=sec)).split(".")[0])
    
    start = time.time()
    print('\nPreprocessing for Learning...')
    ## reverse values setting
    if X_reverse != None:
        for col in X_reverse:
            if (col in df.columns) and (df[col].dtype != 'object'):
                df[col] = (-df[col]) + df[col].max() + 1
            
    # 변수명 변경
    df_rename_eng = pd.read_excel(os.path.join(os.getcwd(), 'Result', '[Result_20240104].xlsx'), sheet_name='DescriptiveStatistics_Rename')
    df_colnames = df_rename_eng.iloc[:,0].unique()
    df_rename_eng = df_rename_eng[df_rename_eng.iloc[:,1] != 0].reset_index().iloc[:,1:]
    df_rename_eng = df_rename_eng.dropna().reset_index().iloc[:,1:]
    df_rename_out = [i for i in df_colnames if i not in df_rename_eng.iloc[:,0].unique()]

    df_rename_kor = pd.read_excel(os.path.join(os.getcwd(), 'Result', '[Result_20240104].xlsx'), sheet_name='CodeBook')
    df_rename_kor = df_rename_kor.iloc[:,[0,4]]
    df_rename_kor = df_rename_kor[df_rename_kor.iloc[:,0].isin(df_rename_out)]

    df_rename_eng.columns = ['Feature', 'Descriptive']
    df_rename_kor.columns = ['Feature', 'Descriptive']
    df_rename = pd.concat([df_rename_eng, df_rename_kor], axis=0, ignore_index=True)
    df_rename = {pair[0]:pair[1] for pair in df_rename.values if pair[0] != Y_colname}
    ## 괄호안 내용 제거
    cleaned_data = {}
    seen_values = set()  # 이미 나타난 값을 저장할 set
    
    for key, value in df_rename.items():
        # 괄호 안의 내용을 제거
        cleaned_value = re.sub(r'\(.*?\)', '', value)
        # 동일한 값이 있으면 새로운 이름으로 변경
        counter = 1
        original_cleaned_value = cleaned_value
        while cleaned_value in seen_values:
            cleaned_value = f"{original_cleaned_value}_{counter}"
            counter += 1
        # 새로운 값은 set에 추가
        seen_values.add(cleaned_value)
        # 새로운 key-value 추가
        cleaned_data[key] = cleaned_value
    df.rename(columns=cleaned_data, inplace=True)   
    print('Renaming Columns!')
    
    # 종속변수 및 독립변수 설정
    X_colname = [x for x in df.columns if x != Y_colname]
    
    # Train, Test 분리
    X_train, X_test, Y_train, Y_test = train_test_split(df[X_colname], df[[Y_colname]],
                                                        test_size=test_size, random_state=random_state)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    print('Data Split!')
    
    # 기술통계
    if class_stat:
        plot_classfrequency(df, Y_colname, label_list=label_list)
        comparisonstat_origin = table_ratiobyclass(df, Y_colname, label_list=label_list, sorting=True)
        display(comparisonstat_origin)
        comparisonstat_origin.to_csv(os.path.join(os.getcwd(), 'Result', 'DescriptiveStatistics_Binary.csv'), 
                                     index=True, encoding='utf-8-sig')
    print('Comparison Statistics of X by Y class!')
        
    # 샘플링
    if sampling_method != None:
        if sampling_method in ['RandomUnderSampler', 'TomekLinks', 'CondensedNearestNeighbour', 'OneSidedSelection']:
            sampler, X_train, Y_train = undersampling(X_train, Y_train, 
                                                     method=sampling_method, strategy=sampling_strategy, 
                                                     random_state=random_state)  
            X_test, Y_test = sampler.fit_resample(X_test, Y_test) 
            print('Under Sampling!')
        elif sampling_method in ['SMOTE', 'SMOTETomek', 'BorderlineSMOTE', 'ADASYN']:
            sampler, X_train, Y_train = oversampling(X_train, Y_train, 
                                                     method=sampling_method, strategy=sampling_strategy, 
                                                     random_state=random_state)
            X_test, Y_test = sampler.fit_resample(X_test, Y_test)    # Over에선 Test는 건드리지 않는게 좀더 현실적
            print('Over Sampling!')
    
    # 스케일링
    print('\nPreprocessing of Scaling...')
    if scaler == 'minmax':
        scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print(X_train.shape, Y_train.shape, X_train.min(), X_train.max())
    print(X_test.shape, Y_test.shape, X_test.min(), X_test.max())    
    
    print('Complete!')
    sec = time.time()-start
    print(str(datetime.timedelta(seconds=sec)).split(".")[0])
    
    return X_train, X_test, X_colname, Y_train, Y_test


### Date and Author: 20240929, Kyungwon Kim ### 
def preprocessing_aispeaker(df_raw, 
                            Y_colname=['q44', 'q45', 'q46'], Y_criteria=None,
                            X_dummy=None, X_reverse=None, X_delete=None,
                            test_size=0.2, class_stat=True,
                            scaler='minmax', random_state=123,
                            label_list=['0', '1']):
    start = time.time()
    df = df_raw.copy()
    print('\nPreprocessing for Data Prepare...')
    print('Initial Shape: ', df.shape)
    
    # 종속변수 후보군 처리
    Y_coltemp = ['q41', 'q42', 'q43', 'q44', 'q45', 'q46']
    if (len(Y_colname) == 1) and (Y_colname[0] in ['q41', 'q42', 'q43']):
        df['Y'] = df[Y_colname]
    elif (len(Y_colname) == 1) and (Y_colname[0] in ['q44', 'q45', 'q46']):
        df['Y'] = df[Y_colname]
    elif (len(Y_colname) == 3) and (Y_colname == ['q41', 'q42', 'q43']):
        df['Y'] = df[Y_colname].mean(axis=1)
    elif (len(Y_colname) == 3) and (Y_colname == ['q44', 'q45', 'q46']):
        df['Y'] = df[Y_colname].mean(axis=1)
    df = feature_drop(df, col_target=Y_coltemp)   
    Y_colname = 'Y'
    if Y_criteria == None:
        df[Y_colname] = df[Y_colname].apply(lambda x: 1 if x >= df[Y_colname].mean() else 0)
    else:
        df[Y_colname] = df[Y_colname].apply(lambda x: 1 if x >= Y_criteria else 0)
    
    # 독립변수 처리
    ## 보유장비갯수변환
    df[['q4_1', 'q4_2', 'q4_3', 'q4_4']] = df[['q4_1', 'q4_2', 'q4_3', 'q4_4']].replace(2, 1).replace(3, 1).replace(4, 1).fillna(0)
    df['q4_sum'] = df[['q4_1', 'q4_2', 'q4_3', 'q4_4']].sum(axis=1)
    ## 더미변수 생성
    if X_dummy != None:
        df = feature_categ_DummyVariable(df, target=X_dummy)
    ## 더미합 생성
    df['q8_1_9'] = 0
    df_q8 = df[['q8_1_1', 'q8_1_2', 'q8_1_3', 'q8_1_4', 'q8_1_5', 'q8_1_6', 'q8_1_7', 'q8_1_8', 'q8_1_9', 'q8_1_10']].values
    df_q8 = df_q8 + df[['q8_2_1', 'q8_2_2', 'q8_2_3', 'q8_2_4', 'q8_2_5', 'q8_2_6', 'q8_2_7', 'q8_2_8', 'q8_2_9', 'q8_2_10']].values
    df_q8 = df_q8 + df[['q8_3_1', 'q8_3_2', 'q8_3_3', 'q8_3_4', 'q8_3_5', 'q8_3_6', 'q8_3_7', 'q8_3_8', 'q8_3_9', 'q8_3_10']].values
    df_q8 = pd.DataFrame(df_q8, columns=['q8_1', 'q8_2', 'q8_3', 'q8_4', 'q8_5', 'q8_6', 'q8_7', 'q8_8', 'q8_9', 'q8_10'])
    df_q8.index = df.index
    df = pd.concat([df, df_q8], axis=1)
    
    # 결측치 처리
    print('\nPreprocessing NAN...')
    df = null_filling(df, delcol_nullratio=0.75, delrow_nullratio=None, fill_method='newlabel')
    
    # 불필요 변수 삭제
    if X_delete != None:
        print('\nDeleting Non-meaning Variables...')
        df = feature_drop(df, col_target=X_delete)
        
    # 사용 변수명 변경
    df_rename = pd.read_excel(os.path.join(os.getcwd(), 'Data', '1.인공지능 스피커 인식조사.xlsx'), sheet_name='Rename')
    df_rename['신규변수명'] = df_rename['신규변수명'].apply(lambda x: x.replace('_x000D_', ''))
    df_rename = {pair[0]:pair[1] for pair in df_rename.values if pair[0] != Y_colname}
    df.rename(columns=df_rename, inplace=True)
    
    # 종속변수 및 독립변수 설정
    Y_colname = 'Y'
    X_colname = [x for x in df.columns if x != Y_colname]
    
    # Train, Test 분리
    X_train, X_test, Y_train, Y_test = train_test_split(df[X_colname], df[[Y_colname]],
                                                        test_size=test_size, random_state=random_state)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    print('Data Split!')
    
    # 기술통계
    if class_stat:
        plot_classfrequency(df, Y_colname, label_list=label_list)
        comparisonstat_origin = table_ratiobyclass(df, Y_colname, label_list=label_list, sorting=True)
        display(comparisonstat_origin)
        comparisonstat_origin.to_csv(os.path.join(os.getcwd(), 'Result', 'DescriptiveStatistics_Binary.csv'), 
                                     index=True, encoding='utf-8-sig')
    print('Comparison Statistics of X by Y class!')
    
    # 스케일링
    print('\nPreprocessing of Scaling...')
    if scaler == 'minmax':
        scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print(X_train.shape, Y_train.shape, X_train.min(), X_train.max())
    print(X_test.shape, Y_test.shape, X_test.min(), X_test.max())    
    
    print('Complete!')
    sec = time.time()-start
    print(str(datetime.timedelta(seconds=sec)).split(".")[0])
    
    return X_train, X_test, X_colname, Y_train, Y_test


### Date and Author: 20241205, Kyungwon Kim ### 
def preprocessing_mdis(df_raw, 
                       Y_colname='Y',
                       X_dummy=None, X_reverse=None, X_delete=None,
                       test_size=0.2, class_stat=True,
                       sampling_method=None, sampling_strategy='auto',
                       scaler='minmax', random_state=123,
                       label_list=['0', '1']):
    start = time.time()
    df = df_raw.copy()
    print('\nPreprocessing for Data Prepare...')
    print('Initial Shape: ', df.shape)
    
    # 결측치 처리
    print('\nPreprocessing NAN...')
    df = null_filling(df, delcol_nullratio=0.5, delrow_nullratio=None, fill_method='newlabel')
    print('Shape: ', df.shape)

    # 독립변수 처리
    ## 사용 변수명 변경
    df.rename(columns={'Key':'Year'}, inplace=True)
    ## 불필요 변수 삭제
    if X_delete != None:
        print('\nDeleting Non-meaning Variables...')
        df = feature_drop(df, col_target=X_delete)
        print('Shape: ', df.shape)
    ## 오름차순 정리
    if X_reverse != None:
        for col in X_reverse:
            if (col in df.columns) and (df[col].dtype != 'object'):
                df[col] = (-df[col]) + df[col].max() + 1
    ## 더미변수 생성
    if X_dummy != None:
        df = feature_categ_DummyVariable(df, target=X_dummy)
    print('Shape: ', df.shape)
    
    # 종속변수 및 독립변수 설정
    df[Y_colname] = df.apply(lambda x: 1 if (x[Y_colname]<=3) and (x['Year']==2015) 
                             else (1 if (x[Y_colname]<=1) and (x['Year']!=2015) else 0), axis=1)
    X_colname = [x for x in df.columns if x != Y_colname]
    
    # Train, Test 분리
    X_train, X_test, Y_train, Y_test = train_test_split(df[X_colname], df[[Y_colname]],
                                                        test_size=test_size, random_state=random_state)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    print('Data Split!')
    
    # 기술통계
    if class_stat:
        plot_classfrequency(df, Y_colname, label_list=label_list)
        comparisonstat_origin = table_ratiobyclass(df, Y_colname, label_list=label_list, sorting=True)
        print('Ratio of Origin Y:', df[Y_colname].value_counts())
        display(comparisonstat_origin)
        folder_location = os.path.join(os.getcwd(), 'Result')
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)
        comparisonstat_origin.to_csv(os.path.join(folder_location, 'DescriptiveStatistics_Binary.csv'), 
                                     index=True, encoding='utf-8-sig')
    print('Comparison Statistics of X by Y class!')
    
    # 샘플링
    if sampling_method != None:
        if sampling_method in ['RandomUnderSampler', 'TomekLinks', 'CondensedNearestNeighbour', 'OneSidedSelection']:
            sampler, X_train, Y_train = undersampling(X_train, Y_train, 
                                                     method=sampling_method, strategy=sampling_strategy, 
                                                     random_state=random_state)  
            X_test, Y_test = sampler.fit_resample(X_test, Y_test) 
            print('Under Sampling!')
        elif sampling_method in ['SMOTE', 'SMOTETomek', 'BorderlineSMOTE', 'ADASYN']:
            sampler, X_train, Y_train = oversampling(X_train, Y_train, 
                                                     method=sampling_method, strategy=sampling_strategy, 
                                                     random_state=random_state)
            X_test, Y_test = sampler.fit_resample(X_test, Y_test)    # Over에선 Test는 건드리지 않는게 좀더 현실적
            print('Over Sampling!')
    
    # 스케일링
    print('\nPreprocessing of Scaling...')
    if scaler == 'minmax':
        scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print(X_train.shape, Y_train.shape, X_train.min(), X_train.max())
    print(X_test.shape, Y_test.shape, X_test.min(), X_test.max())    
    
    print('Complete!')
    sec = time.time()-start
    print(str(datetime.timedelta(seconds=sec)).split(".")[0])
    
    return X_train, X_test, X_colname, Y_train, Y_test


### Date and Author: 20250528, Kyungwon Kim ### 
def preprocessing_edu(df_raw, Y_colname='Y', 
                      X_dummy=None, X_reverse=None, X_delete=None, X_custom=None,
                      test_size=0.2, class_stat=True,
                      sampling_method=None, sampling_strategy='auto',
                      scaler='minmax', random_state=123,
                      label_list=['0', '1']):
    df = df_raw.copy()
    print('\nPreprocessing for Data Prepare...')
    print('Initial Shape: ', df.shape)

    # 변수 처리
    ## 변수명 공백제거
    df.columns = [col.strip() for col in df.columns]
    ## 사용 변수명 변경
    df.rename(columns={'Key':'Year'}, inplace=True)
    ## 종속변수 제거
    if type(Y_colname) == dict:
        Y_col = [col for col in Y_colname.keys()][0]
    else:
        Y_col = Y_colname
    X_dummy = [name for name in X_dummy if name != Y_col]
    X_reverse = [name for name in X_reverse if name != Y_col]
    X_delete = [name for name in X_delete if name != Y_col]
    if Y_col in X_custom.keys():
        del X_custom[Y_col]     
    ## 임의변수 처리
    if X_custom != None:
        for col, mapping in X_custom.items():
            if col in df.columns:
                df[col] = df[col].replace(mapping)
            if col == '문C1) 앞으로 참여하길 희망하는 프로그램은 무엇입니까? (중복응답)1':
                df['문C1) 앞으로 참여하길 희망하는 프로그램은 무엇입니까? (1)학위취득을위한교육'] = ((df[col] >= 1) & (df[col] <= 10))*1
                df['문C1) 앞으로 참여하길 희망하는 프로그램은 무엇입니까? (2)성인기초및문해교육'] = (df[col] == 11)*1
                df['문C1) 앞으로 참여하길 희망하는 프로그램은 무엇입니까? (3)직업능력향상교육'] = ((df[col] >= 12) & (df[col] <= 17))*1
                df['문C1) 앞으로 참여하길 희망하는 프로그램은 무엇입니까? (4)인문교양교육'] = ((df[col] >= 18) & (df[col] <= 24))*1
                df['문C1) 앞으로 참여하길 희망하는 프로그램은 무엇입니까? (5)문화예술스포츠교육'] = ((df[col] >= 25) & (df[col] <= 27))*1
                df['문C1) 앞으로 참여하길 희망하는 프로그램은 무엇입니까? (5)문화예술스포츠교육'] = ((df[col] >= 28) & (df[col] <= 30))*1
    ## 그룹화
    X_group = {
        '학습 지향 평균': [
            '문F1-5) 다음 학습 관련 질문에 어느 정도 동의하시는지 동의 정도를 말씀해주시기 바랍니다.__교육훈련은 보다 나은 일상생활을 영위하는 데 도움을 준다.',
            '문F1-6) 다음 학습 관련 질문에 어느 정도 동의하시는지 동의 정도를 말씀해주시기 바랍니다.__새로운 것을 배우는 것은 즐겁다.',
            '문F1-7) 다음 학습 관련 질문에 어느 정도 동의하시는지 동의 정도를 말씀해주시기 바랍니다.__학습을 통해 자신감을 얻는다.',
            '문F1-8) 다음 학습 관련 질문에 어느 정도 동의하시는지 동의 정도를 말씀해주시기 바랍니다.__성인학습자는 자신의 학습을 위해 무언가를 지불할 각오를 해야 한다.'],
        '직업관련 목표지향 평균': [
            '문F1-1) 다음 학습 관련 질문에 어느 정도 동의하시는지 동의 정도를 말씀해주시기 바랍니다.__성인이 되어서도 지속적으로 학습을 하는 사람은 일자리를 잃을 가능성이 적다.',
            '문F1-2) 다음 학습 관련 질문에 어느 정도 동의하시는지 동의 정도를 말씀해주시기 바랍니다.__성공적인 직장생활을 위해서는 지식과 기술을 끊임없이 향상시켜야 한다.',
            '문F1-3) 다음 학습 관련 질문에 어느 정도 동의하시는지 동의 정도를 말씀해주시기 바랍니다.__고용주는 고용인들의 훈련을 책임져야 한다.']
    }
    if Y_colname == '학습효과성':
        X_group[Y_colname] = [
            '문H2-1) 평생학습 참여가 삶의 질 향상에 얼마나 도움^ 아직 경험이 없다면 삶의 질 향상에 얼마나 도움이 될 것인지 생각하십니까?__1) 정신적 건강(정서적 안정감)',
            '문H2-1) 평생학습 참여가 삶의 질 향상에 얼마나 도움^ 아직 경험이 없다면 삶의 질 향상에 얼마나 도움이 될 것인지 생각하십니까?__2) 육체적 건강',
            '문H2-1) 평생학습 참여가 삶의 질 향상에 얼마나 도움^ 아직 경험이 없다면 삶의 질 향상에 얼마나 도움이 될 것인지 생각하십니까?__3) 사회참여 만족도',
            '문H2-1) 평생학습 참여가 삶의 질 향상에 얼마나 도움^ 아직 경험이 없다면 삶의 질 향상에 얼마나 도움이 될 것인지 생각하십니까?__4) 경제적 안정감'
        ]
    X_del = []
    for group_name, columns in X_group.items():
        df[group_name] = df[columns].mean(axis=1, skipna=True)
        X_del.extend(columns)
    df.drop(columns=X_del, inplace=True, errors='ignore')
    X_group = {
        '무형식학습(의존형) 참여': [
            '문B1-1) 귀하께서는 작년에 다음과 같은 학습에 참여해 본 적이 있으십니까?__가족^ 친구 또는 직장동료^ 상사의 도움이나 조언을 통해 지식을 습득한 적이 있다', 
            '문B1-5) 귀하께서는 작년에 다음과 같은 학습에 참여해 본 적이 있으십니까?__학습을 목적으로 텔레비전^ 라디오 등을 활용해서 새로운 지식을 습득한 적이 있다', 
            '문B1-6) 귀하께서는 작년에 다음과 같은 학습에 참여해 본 적이 있으십니까?__책이나 전문잡지 등 인쇄매체를 활용해서 지식을 습득한 적이 있다'],
        '무형식학습(온라인) 참여': [
            '문B1-2) 귀하께서는 작년에 다음과 같은 학습에 참여해 본 적이 있으십니까?__트위터^ 페이스북^ 카페^ 블로그^ 밴드 등을 활용해서 새로운 정보나 기술을 습득한 적이 있다', 
            '문B1-3) 귀하께서는 작년에 다음과 같은 학습에 참여해 본 적이 있으십니까?__유튜브(Youtube) 등을 활용해서 새로운 정보나 기술을 습득한 적이 있다', 
            '문B1-4) 귀하께서는 작년에 다음과 같은 학습에 참여해 본 적이 있으십니까?__인터넷 뉴스^ E-book 등 온라인매체를 활용해서 새로운 정보나 기술을 습득한 적이 있다',],
        '무형식학습 (암묵적 학습) 참여': [
            '문B1-7) 귀하께서는 작년에 다음과 같은 학습에 참여해 본 적이 있으십니까?__역사적·자연적·산업적 장소를 방문해서 지식을 습득한 적이 있다',
            '문B1-8) 귀하께서는 작년에 다음과 같은 학습에 참여해 본 적이 있으십니까?__도서관 등을 방문해서 새로운 사실을 배운 적이 있다',
            '문B1-9) 귀하께서는 작년에 다음과 같은 학습에 참여해 본 적이 있으십니까?__축제^ 박람회^ 음악회 등에 참여해서 무언가를 새롭게 배우거나 깊이 있게 알게된 적이 있다',
            '문B1-10) 귀하께서는 작년에 다음과 같은 학습에 참여해 본 적이 있으십니까?__스포츠^ 등산 등 신체를 움직이는 활동에 참여해서 무언가를 새롭게 배우거나 깊이 있게 알게 된 적이 있다'],
        '사회참여': [
            '문G1-1) 지난 한 해 동안 자원봉사 또는 재능기부를 한 적이 있습니까? 있다면^ 몇 회 정도 참여하셨습니까?__참여경험',
            '문G1-2) 지난 한 해 동안 자선단체에 기부 또는 후원한 적이 있습니까? 있다면^ 몇 회 정도 참여하셨습니까?__참여경험',
            '문G1-3) 지난 한 해 동안 동아리에서 활동한 적이 있습니까? 있다면^ 몇 회 정도 참여하셨습니까?__참여경험',
            '문G1-4) 지난 한 해 동안 지역사회단체에 참여한 적이 있습니까? 있다면^ 몇 회 정도 참여하셨습니까?__참여경험']
    }
    X_del = []
    for group_name, columns in X_group.items():
        df[group_name] = (df[columns].applymap(lambda x: 1 if x == 1 else 0).any(axis=1).astype(int))  
        X_del.extend(columns)
    df.drop(columns=X_del, inplace=True, errors='ignore')
    ## 오름차순 정리
    if X_reverse != None:
        for col in X_reverse:
            if (col in df.columns) and (df[col].dtype != 'object'):
                df[col] = (-df[col]) + df[col].max() + 1
    ## 불필요 변수 삭제
    if X_delete != None:
        print('\nDeleting Non-meaning Variables...')
        df = feature_drop(df, col_target=X_delete)
        print('Shape: ', df.shape)
    ## 결측치 처리
    print('\nPreprocessing NAN...')
    df = null_filling(df, delcol_nullratio=0.5, delrow_nullratio=None, fill_method='newlabel')
    print('Shape: ', df.shape)
    ## 더미변수 생성 (생성과정은 결측치 처리 후!)
    if X_dummy != None:
        df = feature_categ_DummyVariable(df, target=X_dummy)
    print('Shape: ', df.shape)
    ## 문자열 처리
    colname_object = list(df.columns[df.dtypes == 'object'])
    print('\nObject X Features: ', colname_object)
    encoder = LabelEncoder()
    for col in colname_object:
        df[col] = encoder.fit_transform(df[col])
    print('Complete X Object Processing!')

    # 종속변수 처리
    if type(Y_colname) == dict:
        for col, mapping in Y_colname.items():
            df[col] = df[col].map(mapping)
        Y_colname = col
    if Y_colname == '학습효과성':
        df[Y_colname] = df[Y_colname].apply(lambda x: 1 if x>=4 else (0 if x<3 else None))
        df.dropna(subset=[Y_colname], inplace=True)
        df = df.reset_index().iloc[:,1:]

    # 독립변수 설정
    X_colname = [x for x in df.columns if x != Y_colname]

    # Train, Test 분리
    X_train, X_test, Y_train, Y_test = train_test_split(df[X_colname], df[[Y_colname]],
                                                        test_size=test_size, random_state=random_state)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    print('Data Split!')

    # 기술통계
    if class_stat:
        plot_classfrequency(df, Y_colname, label_list=label_list)
        comparisonstat_origin = table_ratiobyclass(df, Y_colname, label_list=label_list, sorting=True, save=True)
        display(comparisonstat_origin)
    print('Comparison Statistics of X by Y class!')

    # 샘플링
    if sampling_method != None:
        if df[Y_colname].value_counts().sort_values(ascending=False).index[0] == 0:
            print('Undersampling...')
            if sampling_method == 'auto':
                sampling_method = 'RandomUnderSampler'
            sampler, X_train, Y_train = undersampling(X_train, Y_train,
                                                        method=sampling_method, strategy=sampling_strategy,
                                                        random_state=random_state)    
        else:
            print('Oversampling...')
            if sampling_method == 'auto':
                sampling_method = 'ADASYN' # 'SMOTETomek' #'ADASYN'
            sampler, X_train, Y_train = oversampling(X_train, Y_train, 
                                                        method=sampling_method, strategy=sampling_strategy, 
                                                        random_state=random_state)

    # 스케일링
    print('\nPreprocessing of Scaling...')
    if scaler == 'minmax':
        scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print(X_train.shape, Y_train.shape, X_train.min(), X_train.max())
    print(X_test.shape, Y_test.shape, X_test.min(), X_test.max())    
    
    print('Complete!')

    return X_train, X_test, X_colname, Y_train, Y_test


### Date and Author: 20250423, Kyungwon Kim ### 
def preprocessing_puppeteer2df(df_json):
    # Define the required fields to extract
    fields = ['date', 'createdDate', 'updatedDate', 'press', 'category', 'journalist', 
              'title', 'summary', 'body', 
              'comment_nick', 'comment_date', 'comment_content', 
              'pressUrl', 'naverUrl']
    
    # Extract only the required fields from the JSON data
    filtered_data = []
    for item in df_json:
        ## 1차 필터 추출
        filtered_item = {field: item.get(field, None) for field in fields}
        ## 댓글 리스트화 
        if item.get('comments'):
            comment_nick = [comment.get('nick', '') for comment in item.get('comments', [])]
            comment_date = [comment.get('date', '') for comment in item.get('comments', [])]
            comment_content = [comment.get('content', '') for comment in item.get('comments', [])]
            filtered_item['comment_nick'] = comment_nick
            filtered_item['comment_date'] = comment_date
            filtered_item['comment_content'] = comment_content
        ## 저장
        if (filtered_item['date'] != ''):
            filtered_data.append(filtered_item)
    df = pd.DataFrame(filtered_data)

    ## rename
    df.rename(columns={'date':'Date', 'createdDate':'DateCreated', 'updatedDate':'DateUpdated',
                       'press':'Press', 'category':'Category', 'journalist':'Journalist',
                       'title':'Title', 'summary':'ContentSummary', 'body':'Content',
                       'comment_nick':'Nickname', 'comment_date':'DateComment', 'comment_content':'Comment',
                       'pressUrl':'Url', 'naverUrl':'NaverUrl'}, inplace=True)
    return df


### Date and Author: 20250501, Kyungwon Kim ### 
def preprocessing_bigkinds(df_raw):
    df = df_raw.copy()
    
    # 변수 삭제
    delete_colname = ['뉴스 식별자', '사건/사고 분류1', '사건/사고 분류2', '사건/사고 분류3', '분석제외 여부']
    df.drop(columns=delete_colname, inplace=True)

    # 전처리
    df['통합 분류1'] = df['통합 분류1'].apply(lambda x: str(x).split('>')[0])
    df['통합 분류1'].replace('nan', '', inplace=True)
    df['통합 분류2'] = df['통합 분류2'].apply(lambda x: str(x).split('>')[0])
    df['통합 분류2'].replace('nan', '', inplace=True)
    df['통합 분류3'] = df['통합 분류3'].apply(lambda x: str(x).split('>')[0])
    df['통합 분류3'].replace('nan', '', inplace=True)
    df['인물'].fillna('', inplace=True)
    df['위치'].fillna('', inplace=True)
    df['기관'].fillna('', inplace=True)
    df['키워드'].fillna('', inplace=True)
    df['본문'].fillna('', inplace=True)
    df.rename(columns={'특성추출(가중치순 상위 50개)':'키워드순위'}, inplace=True)

    # 정렬
    df = df[['검색어', '일자', '언론사', '기고자', '통합 분류1', '통합 분류2', '통합 분류3', '본문', '키워드', '키워드순위', '위치', '기관', 'URL']]
    
    return df







    