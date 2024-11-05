# ##### Usage of KK #####
# # Ignore the warnings
# import warnings
# # warnings.filterwarnings('always')
# warnings.filterwarnings('ignore')

# # System related and data input controls
# import os

# # Python path
# import sys
# base_folder = 'DataScience'
# location_base = os.path.join(os.getcwd().split(base_folder)[0], base_folder)
# location_module = [os.path.join(location_base, 'Module')] 
# for each in location_module:
#     if each not in sys.path:
#         sys.path.append(each)

# # Auto reload of library
# %reload_ext autoreload
# %autoreload 2

# # Import public library
# # from install_packages_KK import *
# from install_import_KK import *
# import data_KK
# from data_KK import *
# import description_KK
# from description_KK import *
# import preprocessing_KK
# from preprocessing_KK import *
# from algorithm_KK import *
# import algorithm_volatility_KK
# from algorithm_volatility_KK import *
# import algorithm_nonlinear_KK
# from algorithm_nonlinear_KK import *
# from algorithm_realoption_KK import *
# import algorithm_momentum_KK
# from algorithm_momentum_KK import *
# from algorithm_als_KK import *
# from algorithm_attribution_KK import *
# import evaluation_KK
# from evaluation_KK import *


##### data_KK.py + description_KK.py #####
# System related and data input controls
import sys
import sysconfig
import os
import platform
import io
from io import BytesIO
import mpu
from zipfile import ZipFile
import pickle as pk
import joblib
import glob
import inspect # inspect.getfile(module)
from urllib.request import urlopen

# Ignore the warnings
import warnings
warnings.filterwarnings('ignore') # 'always'

# JAVA 설치: https://www.oracle.com/java/technologies/downloads/#jdk17-windows
import os
if 'JAVA_HOME' not in os.environ:
    print('JAVA is in the system path?: ', 'JAVA_HOME' in os.environ)
    print('JAVA is in the system path?: ', 'Adding...')
    os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-17\bin\server'    # JAVA 설치경로 반영
else:
    print('JAVA is in the system path?:', 'JAVA_HOME' in os.environ)

# PC Status
print('Operation Machine: ', platform.processor())
print('Operation Platform: ', platform.architecture()[0])
print('OS Type: ', platform.system())
print('OS Version: ', platform.release())
print("Python Version: ", sys.version)

# CPU & GPU Setting
def DeviceStrategy_CPU():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
def DeviceStrategy_GPU():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.python.client import device_lib
    import tensorflow.python.platform.build_info as build
    
    # Status Check
    print('\nTensorflow Version: ', tf.__version__)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('Keras Version: ', keras.__version__)
    print('Num of Physical GPUs Available: ', len(gpus))
    print('Cuda is Ready? ', tf.test.is_built_with_cuda())
    print('Cuda Version: ', build.build_info['cuda_version'])
    print('Cudnn Version: ', build.build_info['cudnn_version'], '\n')
     
    # Use GPU Device  
    try:
        # 모든 GPU 사용 설정
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(len(gpus))])
        # GPU 메모리를 전부 사용하지 않고 천천히 사용할 만큼만 상승 시킨다.
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Batch Parellel Strategy
        # tf.distribute.HierarchicalCopyAllReduce() / tf.distribute.ReductionToOneDevice()
        device_names = ['/GPU:'+str(i) for i in range(len(gpus))]
        strategy = tf.distribute.MirroredStrategy(devices=device_names,
                                                  cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    except RuntimeError as e:
        print(e)
        
    # Torch Check
    import torch
    print('Torch Version: ', torch.__version__)
    print("Torch Cuda Version: {}".format(torch.version.cuda))
    print("Torch Cudnn Version:{}".format(torch.backends.cudnn.version()))
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU
        torch_device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        torch_device = torch.device("cpu")

    return strategy, torch_device, len(gpus)

# Distributed data handling
# from pyspark import SparkConf, SparkContext, SQLContext, ml
# from pyspark.sql import SparkSession
# from pyspark.sql import functions as fs
# from pyspark.sql.functions import udf
# from pyspark.sql.types import *
# from pyspark.ml.recommendation import ALS
# from pyspark.ml.evaluation import RegressionEvaluator

# Data manipulation and useful functions
import pandas as pd
import numpy as np
global precision
precision = 4
def global_precision(input_precision=6):
    global precision
    precision = input_precision
output_precision = '{:,.'+str(precision)+'f}'
pd.options.display.float_format = output_precision.format # output format
pd.options.display.max_rows = 100 # display row numbers
pd.options.display.max_columns = 40 # display column numbers
pd.set_option('display.max_colwidth', 100)
import math
from numpy.linalg import eig
import time
import datetime
from holidayskr import year_holidays, is_holiday
import patsy as pt
from patsy import dmatrix
import random
import itertools
from itertools import product, permutations, combinations # iterative combinations
from collections import Counter # calculate frequency in values of list
from tqdm import tqdm, tqdm_pandas # execution time
tqdm.pandas()
from ast import literal_eval
import multiprocessing as mp
import ray
ray.init(num_cpus=mp.cpu_count()-1, ignore_reinit_error=True, log_to_driver=False)
import logging
import dabl
import sweetviz as sv
from ydata_profiling import ProfileReport
import dtale

# Datasets
import pandas_datareader as pdr
import pandas_datareader.data as web
from statsmodels import datasets
from sklearn import datasets
try:
    from sklearn.datasets import load_boston
except:
    pass
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.datasets import fetch_california_housing, fetch_lfw_people, fetch_20newsgroups
from mglearn.datasets import make_signals
# from keras.datasets import cifar10, mnist, imdb
from covid19dh import covid19

# Sampling
from imblearn.under_sampling import TomekLinks, CondensedNearestNeighbour, OneSidedSelection
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek

# Preprocessing
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from gensim import corpora
# from keras.utils import np_utils, multi_gpu_model
# from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing import image
# from keras.applications.imagenet_utils import preprocess_input
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.manifold import TSNE

# Preprocessing Text
import re
import string
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
from spacy.lang.ko.stop_words import STOP_WORDS as sw_kor
from soynlp.normalizer import *
from soynlp.word import WordExtractor
from soynlp.noun import LRNounExtractor, LRNounExtractor_v2, NewsNounExtractor
from soynlp.tokenizer import LTokenizer
from sentence_transformers import SentenceTransformer

# General
import statsmodels
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence as sm_out
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import bds # brock dechert scheinkman (IID)
from scipy import stats
from scipy.stats import norm
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel, bartlett
from scipy.stats import wilcoxon, mannwhitneyu, kruskal, levene
# from scipy.stats import entropy
from pyGRNN import GRNN
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.get_logger().warning('test')
# import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Embedding, Reshape, RepeatVector, Permute, Multiply, Lambda
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU
# from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D, AveragePooling2D, UpSampling2D
# from keras.layers.convolutional import Conv3D, Convolution3D, MaxPooling3D, AveragePooling3D, UpSampling3D
# from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D
# from keras.layers import TimeDistributed, CuDDN, CuDNNLSTM
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras_tqdm import tqdm_callback, TQDMNotebookCallback
# import keras.backend as K
# from keras.optimizers import SGD
# import torch
# from torch.nn import CrossEntropyLoss
# import torch.nn.functional as F
# import torch.optim as optim
## Regression
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree, export_text, export_graphviz
from sklearn.ensemble import VotingRegressor, BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from xgboost import plot_importance as plot_importance_xgb
from lightgbm import plot_importance as plot_importance_lgbm
from catboost import Pool, CatBoostRegressor
from mlxtend.regressor import StackingRegressor, StackingCVRegressor
## Time series
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
# import pytimetk as tk
import pmdarima as pm
from pmdarima import AutoARIMA
# from statsforecast import StatsForecast
# from statsforecast.models import AutoARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
from statsmodels.tsa.vector_ar.var_model import VARProcess, VAR
from statsmodels.tsa.vector_ar.util import varsim
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import coint
import hmmlearn
from hmmlearn.hmm import BaseHMM, GaussianHMM, MultinomialHMM
# import arch
# from arch import arch_model
# from neuralprophet import NeuralProphet
## Classification
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree, export_text, export_graphviz
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from xgboost import plot_importance as plot_importance_xgb
from lightgbm import plot_importance as plot_importance_lgbm
from catboost import Pool, CatBoostClassifier
from mlxtend.classifier import StackingClassifier, StackingCVClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.neural_network import MLPClassifier
## Clustering
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
## Dimension Reduction
from sklearn.decomposition import PCA
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer import FactorAnalyzer
import pingouin as pg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.decomposition import NMF
## Text Mining
from umap import UMAP
from bertopic import BERTopic
from transformers import pipeline, AutoTokenizer, BertTokenizer, BertTokenizerFast
from transformers import AutoModel, BertModel, BertForSequenceClassification
from transformers import TFBertModel, TFBertForSequenceClassification
from transformers import BertConfig, AdamW
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup

# Model selection
import sklearn.model_selection  as skl_slct
from sklearn.model_selection import train_test_split, TimeSeriesSplit
## parameter selection
from sklearn.model_selection import GridSearchCV
## for one metric
from sklearn.model_selection import cross_val_score
## for more than tow metrics
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold

# Evaluation metrics
from sklearn import metrics
## for regression
from sklearn.metrics import r2_score, mean_squared_error,  mean_absolute_error, mean_absolute_percentage_error
from sktime.performance_metrics.forecasting import mean_squared_percentage_error, median_absolute_percentage_error, mean_relative_absolute_error, median_relative_absolute_error
## for classification
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
# from keras.metrics import top_k_categorical_accuracy
## for timeseries
# from fastdtw import fastdtw
# for clustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
## for distance
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import euclidean, correlation
from scipy import cluster
## for transformation
from sklearn.utils.extmath import softmax
from scipy.special import softmax


# Explainability
import shap
# from dominance_analysis import Dominance_Datasets
# from dominance_analysis import Dominance

# Visualization
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
plt.style.use('default')
## 한글 이슈
global FONT_NAME
global FONT_PATHS
from matplotlib import font_manager, rc
if platform.system() == 'Darwin': #맥
    plt.rc('font', family='AppleGothic') 
elif platform.system() == 'Windows': #윈도우
    FONT_NAME = 'Malgun Gothic'
    plt.rc('font', family=FONT_NAME) 
    plt.rcParams['font.family'] = FONT_NAME
    mpl.rc('font', family=FONT_NAME)
    sns.set(font=FONT_NAME) 
    sys_font = font_manager.findSystemFonts()
    FONT_PATHS = [path for path in sys_font if 'malgun' in path]
    rc('font', family=font_manager.FontProperties(fname=FONT_PATHS[0]).get_name())
elif platform.system() == 'Linux': #리눅스 (구글 콜랩)
    #!wget "https://www.wfonts.com/download/data/2016/06/13/malgun-gothic/malgun.ttf"
    #!mv malgun.ttf /usr/share/fonts/truetype/
    #import matplotlib.font_manager as fm 
    #fm._rebuild() 
    FONT_NAME = 'Malgun Gothic'
    plt.rc('font', family=FONT_NAME) 
## 마이너스 이슈
mpl.rc('axes', unicode_minus=False)
plt.rc('axes', unicode_minus=False)
sns.set(rc={"axes.unicode_minus":False}, style='white')
## Custom
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                                                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
                                                '#bcbd22', '#17becf'])
import matplotlib.dates as mdates
import matplotlib.mlab as mlab
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import FormatStrFormatter
import plotly.offline as pyo
import plotly.express as px
import plotly.graph_objects as go
pyo.init_notebook_mode()
import chart_studio
chart_studio.tools.set_credentials_file(username='thekimk', api_key='nEwucSU0B6SYPhhBIhxx')
import missingno as msno
# import mglearn
from PIL import Image
from sklearn.tree import export_graphviz
import pydot
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import networkx as nx
from networkx.drawing.nx_pydot import to_pydot
import pyLDAvis
import pyLDAvis.gensim_models
if platform.system() == 'Linux':
    import graph_tool.all as gt


##### algorithm_shapley_KK #####
# import shap


##### algorithm_volatility_KK.py #####


##### algorithm_nonlinear_KK.py #####
# Summary of Library: https://github.com/neuropsychology/NeuroKit/issues/40
# > API: Power-law Distribution Fitting (plfit)
# > Install & Use: https://github.com/keflavich/plfit
# import powerlaw

# > API: Multifractal DFA (MFDFA)
# > Install & Use: https://github.com/LRydin/MFDFA
# > Function: Multifractal DFA (Self-similarity and Variability): MFDFA
from MFDFA import MFDFA

# > API: NOnLinear measures for Dynamical Systems (nolds)
# > Install: https://github.com/CSchoel/nolds
# > Functions:
# [Fractal]
# - Detrended fluctuation analysis for non-stationary (Long-term Memory or Persistency): dfa
# - Hurst exponent for non-stationary (Long-term Memory or Persistency): hurst_rs
# - Lyapunov exponent (Chaos and Unpredictability): lyap_r, lyap_e
# - Correlation dimension (Fractal Dimension): corr_dim
from nolds import dfa, hurst_rs, lyap_r, lyap_e, corr_dim
# [Entropy: Non-linear Serial Dependence]
# - Sample Entropy (Complexity): sampen
from nolds import sampen

# > API: AntroPy (antropy)
# > Install & Use: https://raphaelvallat.com/entropy/build/html/index.html
# > Functions:
# [Fractal]
# - Detrended fluctuation analysis (DFA): detrended_fluctuation
# - Higuchi Fractal Dimension: higuchi_fd
# - Katz Fractal Dimension: katz_fd
# - Petrosian fractal dimension: petrosian_fd
# from antropy import detrended_fluctuation, higuchi_fd, katz_fd, petrosian_fd
# [Entropy: Non-linear Serial Dependence]
# - Lempel-Ziv complexity of (binary) sequence (1976): lziv_complexity
# - Approximate Entropy (1991, Regularity/Irregularity): app_entropy
# - Permutation Entropy (1992): perm_entropy
# - Sample Entropy (2000): sample_entropy
# - Spectral Entropy (2004): spectral_entropy
# - Singular Value Decomposition entropy (2014): svd_entropy
# from antropy import lziv_complexity, app_entropy, perm_entropy, sample_entropy, spectral_entropy, svd_entropy

# > API: pyEntropy (pyentropy)
# > Install & Use: https://github.com/nikdon/pyEntropy
# > Functions:
# [Entropy: Non-linear Serial Dependence]
# - Shannon Entropy (1949): shannon_entropy
# - Sample Entropy (2000): sample_entropy
# - Multiscale Entropy (2002): multiscale_entropy
# - Permutation Entropy (2002): permutation_entropy
# - Multiscale Permutation Entropy (2005): multiscale_permutation_entropy
# - Composite Multiscale Entropy (2013): composite_multiscale_entropy
from pyentrp.entropy import shannon_entropy, multiscale_entropy, permutation_entropy, multiscale_permutation_entropy, composite_multiscale_entropy

# API: EntroPy (EntroPy-Package)
# > Install & Use: https://pypi.org/project/EntroPy-Package/
# > Functions:
# - Shannon Entropy (1949): EntroPy.entropy
# - Shannon Entropy (1949): EntroPy.shannon_entropy
# - Shannon Entropy (1949): EntroPy.shannon_entropy_from_sequence
# - Sample (2000) + Fuzzy (2007) + FuzzyM (2013) + Updates: EntroPy.sample_entropy
# - Multiscale (2002) + Refined Multiscale (2009) + Composite Multiscale (2013) + Updates: EntroPy.multiscale_entropy
# - Conditional Entropy: conditional_entropy_from_sequence
# - Mutual Information: mutual_information_from_sequence
# - Variation Information: variation_of_information_from_sequence
# from EntroPy import EntroPy

# > API: Information Theory (dit)
# > Install & Use: https://github.com/dit/dit
# > Functions: Entropies / Divergences / Mutual and Informations / Common Information / Partial Information Decomposition / Other Measures
# - Shannon entropy (1949): dit.shannon.entropy
# - Renyi entropy (1961, generalizes the Hartley, Shannon, collision and min-entropy): dit.other.renyi_entropy
# - Mutual information: dit.shannon.mutual_information
# from dit.inference import binned, dist_from_timeseries
# from dit import shannon, other

# > API: Information Theoretic (pyinform)
# > Install & Use: https://elife-asu.github.io/PyInform/starting.html
# > Functions: Empirical Distributions / Shannon Information / ...
# [Entropy: Non-linear Serial Dependence]
from pyinform.dist import Dist
from pyinform.shannon import entropy, relative_entropy
# - Shannon entropy (1949): pyinform.shannon.entropy
# - Relative entropy: pyinform.shannon.relative_entropy
# - Conditional entropy: pyinform.shannon.conditional_entropy
# - Mutual information: pyinform.shannon.mutual_info
# - Conditional mutual information: pyinform.shannon.conditional_mutual_info
from pyinform.shannon import mutual_info, conditional_mutual_info

# > API: Customized
# [Entropy: Non-linear Serial Dependence]
# - Shannon entropy (average uncertainty of x as the number of bits): https://onestopdataanalysis.com/shannon-entropy/
def entropy_shannon(dna_sequence):
    bases = Counter([tmp_base for tmp_base in dna_sequence])
    # define distribution
    dist = [x/sum(bases.values()) for x in bases.values()]
 
    # use scipy to calculate entropy
    entropy_value = stats.entropy(dist, base=2)
 
    return entropy_value

# > API: EntropyHub (EntropyHub) - Install: https://pypi.org/project/EntropyHub/
import EntropyHub as EH

# > Using R
# https://cran.r-project.org/web/packages/RTransferEntropy/index.html
# https://cran.r-project.org/web/views/TimeSeries.html
# https://stackoverflow.com/questions/49776568/calling-functions-from-within-r-packages-in-python-using-importr
# https://www.rdocumentation.org/packages/tseries/versions/0.10-47/topics/terasvirta.test
# https://pkg.robjhyndman.com/tsfeatures/articles/tsfeatures.html


##### algorithm_realoption_KK.py #####


##### algorithm_momentum_KK.py #####
import ruptures as rpt
from ruptures.metrics import hausdorff, randindex, precision_recall


##### algorithm_als_KK.py #####


##### algorithm_attribution_KK.py #####

