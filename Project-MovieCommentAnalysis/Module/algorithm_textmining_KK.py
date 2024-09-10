### KK ###
from import_KK import *


@ray.remote
def modeling_LDA_OptNTopic(tokenized_sentences, dictionary, corpus, num_topics):
    model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, random_state=123)
    model_coherence = CoherenceModel(model=model, texts=tokenized_sentences, dictionary=dictionary, coherence='c_v')
    coherence = model_coherence.get_coherence()
    return [num_topics, coherence, model.log_perplexity(corpus)]


### Date and Author: 20231027, Kyungwon Kim ###
### 토픽 모델링
### LDA 학습할 토큰 입력 후 적용할 문서 입력
def modeling_LDA(df_tokens, df_series, 
                 num_topics='auto', num_topicwords=10, num_topicsamples=10, 
                 save_local=True, save_name='Topics_byLDA.xlsx'):
    # 토큰으로 분리된 문장들을 토큰으로 분리 후 (word_id, word_frequency) 변환
    tokenized_sentences = df_tokens.apply(lambda x: [i for i in str(x).split(' ') if len(i) > 1])
    dictionary = corpora.Dictionary(tokenized_sentences)
    corpus = [dictionary.doc2bow(text) for text in tokenized_sentences]
    
    # Optimal Topic Number
    if num_topics == 'auto':
        ## 병렬처리
        num_list = list(range(2, 10))
        ray.init(num_cpus=mp.cpu_count()-1, ignore_reinit_error=True, log_to_driver=False)
        task = [modeling_LDA_OptNTopic.remote(tokenized_sentences, dictionary, corpus, i) for i in num_list]
        metric_topicnum = ray.get(task)
        ray.shutdown()
        ## Perplexity: 낮을수록 잘 학습 및 정확한 예측 vs Coherence: 높을수록 의미론적 일관성 높음
        ## https://bab2min.tistory.com/587
        metric_topicnum = pd.DataFrame(metric_topicnum, columns=['TopicNumber', 'Coherence', 'Perplexity'])    
        metric_topicnum = metric_topicnum.set_index('TopicNumber')
        num_topics = metric_topicnum.Coherence.idxmax()
        print('Optimal Topic Number: ', num_topics)
        ## 시각화
        plt.figure(figsize=(6, 4))
        sns.lineplot(data=metric_topicnum, x=metric_topicnum.index, y=metric_topicnum.columns[0], 
                     color='red', linestyle='dashed', label='Coherence')
        ax2 = plt.twinx()
        sns.lineplot(data=metric_topicnum, x=metric_topicnum.index, y=metric_topicnum.columns[1], 
                     color='blue', linestyle='dashed', label='Perplexity', ax=ax2)
        plt.legend(loc='best', framealpha=0.3, fancybox=True, fontsize=12)
        ax2.legend(loc='best', framealpha=0.3, fancybox=True, fontsize=12)
        plt.xticks(np.arange(min(metric_topicnum.index), max(metric_topicnum.index)+1, 1))
        plt.tight_layout()
        if save_local:
            folder_location = os.path.join(os.getcwd(), 'Result', 'TopicExtraction', '')
            if not os.path.exists(folder_location):
                os.makedirs(folder_location)
            save_name_opt = os.path.join(folder_location, save_name)
            plt.savefig(save_name_opt.split('.xlsx')[0]+'.png', dpi=600, bbox_inches='tight')
        else:
            plt.show()
     
    # LDA
    model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, random_state=123)
    ## 토픽별 결과 정리
    topic_keyword = []
    for topic, keywords in model.show_topics(num_words=num_topicwords):
        ## 키워드와 비중 정리 결합
        kwd = [keyword.split('*')[1].strip()[1:-1] for keyword in keywords.split('+')]
        kwdweight = [float(keyword.split('*')[0]) for keyword in keywords.split('+')]
        kwdweight = [f"({weight/np.sum(kwdweight)*100:.1f}%)" for weight in kwdweight]
        topic_keyword.append([k+w for k, w in zip(kwd, kwdweight)])
    topic_keyword = pd.DataFrame(pd.Series(topic_keyword))
    topic_keyword.index = ['Topic '+str(i+1) for i in list(topic_keyword.index)]
    topic_keyword.columns = ['Related Keywords']
    topic_keyword['Related Keywords'] = topic_keyword['Related Keywords'].apply(lambda x: str(x)[1:-1])

    ## 문서별 토빅 비중
    doc_topic = []
    for i, topics in enumerate(model[corpus]):
        ## 각 문장에서 비중이 높은 토픽순으로 정렬
        doc = topics[0] if model.per_word_topics else topics        
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        ## 모든 문장에서 각각 수행
        for j, (topic_num, prop_topic) in enumerate(doc): #  몇 번 토픽인지와 비중을 나눠서 저장한다.
                if j == 0:  # 가장 비중이 높은 토픽과 비중, 그리고 전체 토픽의 비중을 저장
                    doc_topic.append([int(topic_num+1), round(prop_topic,4), topics])
                else:
                    break

    doc_topic = pd.DataFrame(doc_topic, index=df_series.index).reset_index()
    doc_topic.columns = ['Document Index', 'Related Topic', 'Weight', 'Each Topic Weights']
    
    ## 토픽별 대표 문서
    topic_doc = pd.DataFrame()
    for topic in sorted(doc_topic['Related Topic'].unique()):
        group = doc_topic.groupby(['Related Topic']).get_group(topic).sort_values(by=['Weight'], ascending=False)
        doc_temp = pd.DataFrame(df_series[list(group[0:num_topicsamples].index)])
        doc_temp['Topic Number'] = 'Topic ' + str(topic)
        doc_temp = pd.concat([doc_temp[['Topic Number']], group[0:num_topicsamples]['Weight'], doc_temp[[doc_temp.columns[0]]]], axis=1)
        topic_doc = pd.concat([topic_doc, doc_temp], axis=0)
    
    ## 정리
    num_sentences = pd.DataFrame(doc_topic['Related Topic'].value_counts())
    num_sentences.index = ['Topic '+str(i) for i in num_sentences.index]
    num_sentences.columns = ['Number of Sentences']
    num_sentences['Percentage'] = num_sentences/np.sum(num_sentences)
    topic_keyword = pd.concat([topic_keyword, num_sentences], axis=1)
    doc_topic = pd.merge(df_series.reset_index(), doc_topic, how='inner', left_on='index', right_on='Document Index')
    doc_topic = doc_topic[[col for col in doc_topic.columns if col not in ['index', 'Document Index']]]
    
    # 저장
    if save_local:
        ## 위치설정
        folder_location = os.path.join(os.getcwd(), 'Result', 'TopicExtraction', '')
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)
        save_name = os.path.join(folder_location, save_name)
        ## 저장
        with pd.ExcelWriter(save_name, engine='xlsxwriter') as writer:
            topic_keyword.to_excel(writer, sheet_name='TopicKeyword')
            topic_doc.to_excel(writer, sheet_name='TopicSentence')
            doc_topic.to_excel(writer, sheet_name='DocumentTopic')
        ## 시각화
        pyLDAvis.enable_notebook(local=True)
        vis = pyLDAvis.gensim_models.prepare(model, corpus, dictionary)
        pyLDAvis.save_html(vis, save_name.split('.xlsx')[0] + '.html')

    return topic_keyword, topic_doc, doc_topic, model


### Date and Author: 20231028, Kyungwon Kim ###
### 토픽 모델링
### UMAP, HDBSACN, embedding_model 추가 참고: https://mz-moonzoo.tistory.com/23
### 시각화설명: https://bongholee.com/bertopic/
### 참고: https://maartengr.github.io/BERTopic/getting_started/quickstart/quickstart.html
def modeling_BERTopic(df_series, vectorizer_type='count', ngram_range=(1,1),
                      tfidf_maxcol=1000, 
                      umap_metric='euclidean', umap_randomseed=123,
                      num_topics='auto', num_topicwords=10, num_topicsamples=10,
                      save_local=True, save_name='Topics_byBERTopic.xlsx'):
    # 설정
    if vectorizer_type == 'count':
        vectorizer = CountVectorizer(max_features=tfidf_maxcol)    # 영어일때 추천
    elif vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=tfidf_maxcol)    # 한국어시 기본
    elif vectorizer_type == 'mecab':    # 한국어시 추천하지만 window 미지원해서 colab 설치 필요
        class CustomTokenizer:
            def __init__(self, tagger):
                self.tagger = tagger
            def __call__(self, sent):
                sent = sent[:1000000]
                word_tokens = self.tagger.morphs(sent)
                result = [word for word in word_tokens if len(word) > 1]
                return result
        vectorizer = CountVectorizer(tokenizer=CustomTokenizer(Mecab()), max_features=tfidf_maxcol)    # 한국어시 추천

    # BERTopic
    umap_model = UMAP(metric=umap_metric, random_state=umap_randomseed)
    print('Valid metrics for dimension reduction of embeddings...:')
    print('''
    * euclidean
    * manhattan
    * chebyshev
    * minkowski
    * canberra
    * braycurtis
    * mahalanobis
    * wminkowski
    * seuclidean
    * cosine
    * correlation
    * haversine
    * hamming
    * jaccard
    * dice
    * russelrao
    * kulsinski
    * ll_dirichlet
    * hellinger
    * rogerstanimoto
    * sokalmichener
    * sokalsneath
    * yule''')
    if num_topics == 'auto':
        model = BERTopic(embedding_model="sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens", 
                         vectorizer_model=vectorizer,
                         n_gram_range=ngram_range,
                         nr_topics=num_topics,
                         top_n_words=num_topicwords,
                         umap_model=umap_model,
                         calculate_probabilities=True)
    else:
        model = BERTopic(embedding_model="sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens", 
                         vectorizer_model=vectorizer,
                         n_gram_range=ngram_range,
                         nr_topics=num_topics,
                         top_n_words=num_topicwords,
                         umap_model=umap_model,
                         calculate_probabilities=True)
    topics, probs = model.fit_transform(df_series.to_list())

    # 정리
    ## 토픽 요약 통계
    topic_keyword = []
    for kwdweight in model.get_topics().values():
        total_weight = np.sum([weight[1] for weight in kwdweight])
        topic_keyword.append([weight[0]+str(f"({weight[1]/total_weight*100:.1f}%)") for weight in kwdweight])
    topic_keyword = pd.DataFrame(pd.Series(topic_keyword))
    topic_keyword.columns = ['Related Keywords']
    topic_keyword['Related Keywords'] = topic_keyword['Related Keywords'].apply(lambda x: str(x)[1:-1])
    topic_keyword = pd.concat([topic_keyword, model.get_topic_info()['Count']], axis=1, ignore_index=False)
    topic_keyword.rename(columns={'Count':'Number of Sentences'}, inplace=True)
    topic_keyword['Percentage'] = topic_keyword.iloc[:,-1]/np.sum(topic_keyword.iloc[:,-1])
    topic_keyword = pd.concat([topic_keyword, model.get_topic_info()['Representative_Docs']], axis=1, ignore_index=False)
    topic_keyword.index = ['Topic '+str(i+2) for i in list(model.get_topics().keys())]
    ## 문서 토픽화
    doc_topic = pd.DataFrame(df_series)
    doc_topic['Related Topic'] = model.get_document_info(0)['Topic']+2
    doc_topic['Weight'] = model.get_document_info(0)['Probability']
    ## 토픽별 대표 문서
    topic_doc = pd.DataFrame()
    for topic in sorted(doc_topic['Related Topic'].unique()):
        group = doc_topic.groupby(['Related Topic']).get_group(topic).sort_values(by=['Weight'], ascending=False)
        doc_temp = pd.DataFrame(df_series[list(group[0:num_topicsamples].index)])
        doc_temp['Topic Number'] = 'Topic ' + str(topic)
        doc_temp = pd.concat([doc_temp[['Topic Number']], group[0:num_topicsamples]['Weight'], doc_temp[[doc_temp.columns[0]]]], axis=1)
        topic_doc = pd.concat([topic_doc, doc_temp], axis=0)
        
    # 저장
    if save_local:
        ## 위치설정
        folder_location = os.path.join(os.getcwd(), 'Result', 'TopicExtraction', '')
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)
        save_name = os.path.join(folder_location, save_name)
        ## 저장
        with pd.ExcelWriter(save_name, engine='xlsxwriter') as writer:
            topic_keyword.to_excel(writer, sheet_name='TopicKeyword')
            topic_doc.to_excel(writer, sheet_name='TopicSentence')
            doc_topic.to_excel(writer, sheet_name='DocumentTopic')
#         ## 시각화
#         pyLDAvis.enable_notebook(local=True)
#         vis = pyLDAvis.gensim_models.prepare(model, corpus, dictionary)
#         pyLDAvis.save_html(vis, save_name.split('.xlsx')[0] + '.html')
            
    return topic_keyword, topic_doc, doc_topic, model


### Date and Author: 20231027, Kyungwon Kim ###
### 긍부정 추론 엔진
### 1000회 이상 호출시 유료 과금
### df_news_sentiment = modeling_CLOVA_SentimentAnalysis(df_news['제목'].to_list())
def modeling_CLOVA_SentimentAnalysis(df_list):
    # Naver Sentiment Analysis API 호출을 위한 설정
    client_id = 'hzq5oum3b9'
    client_secret = 'WcBMOlb0eO4aQHIzSE5zVIen0gp53S1VHlaRTxPm'
    url = 'https://naveropenapi.apigw.ntruss.com/sentiment-analysis/v1/analyze'
    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret,
        "Content-Type": "application/json"
    }

    # 감성분석
    sentiment_total = []
    for sentence in tqdm(df_list):
        data = {'content': sentence}
        
        # API 호출
        response = requests.post(url, data=json.dumps(data, sort_keys=False), headers=headers)
        rescode = response.status_code
        
        # 응답 처리
        if rescode == 200:
            result = json.loads(response.text)['sentences'][0]
            sentiment = [result['content'], 
                         result['sentiment'],
                         result['confidence']['negative'], 
                         result['confidence']['neutral'], 
                         result['confidence']['positive']]
            sentiment_total.append(sentiment)
        else:
            print('Error: '+response.text)
            
    # 정리
    sentiment_total = pd.DataFrame(sentiment_total, 
                                   columns=['sentence', 'sentiment', 'prob_negative', 'prob_neutral', 'prob_positive'])
    sentiment_total.sentiment = sentiment_total.sentiment.apply(lambda x: -1 if x=='negative' else 
                                                                (0 if x=='neutral' else 'positive'))
    
    return sentiment_total
