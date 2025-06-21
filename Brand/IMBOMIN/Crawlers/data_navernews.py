# Ignore the warnings
import warnings
# warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import os
import pickle as pk
import pandas as pd
import numpy as np
import random
import time as t # t로 별칭, 아마 밑에 time이 있어서 겹쳐서 작동되지 않는 것 같음.
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm

#크롤링
from bs4 import BeautifulSoup
import requests
import re
import sys
import csv, json


### Date and Author: 20230731, Kyungwon Kim ###
### 네이버뉴스를 크롤링할 url 생성 및 뉴스 추출 함수 만들기
headers = {
    'authority': 'apis.naver.com',
    'accept': '*/*',
    'accept-language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
    # Requests sorts cookies= alphabetically
    # 'cookie': 'ab.storage.userId.7af503ae-0c84-478f-98b0-ecfff5d67750=%7B%22g%22%3A%22browser-1625985144309-6%22%2C%22c%22%3A1626101500089%2C%22l%22%3A1626101500089%7D; ab.storage.deviceId.7af503ae-0c84-478f-98b0-ecfff5d67750=%7B%22g%22%3A%224cbe130c-6edd-d4aa-a78d-290b003c3592%22%2C%22c%22%3A1626101500094%2C%22l%22%3A1626101500094%7D; ASID=7992e0220000017aaa36664e0000004e; _ga=GA1.2.612969395.1626832328; ab.storage.sessionId.7af503ae-0c84-478f-98b0-ecfff5d67750=%7B%22g%22%3A%2228148006-e01d-7623-b7d1-b4fff0f59b4e%22%2C%22e%22%3A1627919390179%2C%22c%22%3A1627908091281%2C%22l%22%3A1627917590179%7D; MM_NEW=1; NFS=2; NNB=RDIIILNX6JCWE; nx_ssl=2; nid_inf=1665554565; NID_AUT=tP3V5ox533EjyAgkJ1JaqWEnPOhXs2hr3teD39pK972fuXqDWQZXoIOMzICJpa1A; NID_JKL=d393brIzilbjw+7TVvG0OW6Eo22+WIhQAfihItUdgbY=; _naver_usersession_=SPdJTrlTMrn8Udkyn58eo6HL; NID_SES=AAABwJaKJ5FjUAETXL8SAX2HKMUSTt3l8pPu49OSzbGzgKEEMN/ckpP4DbQVHQwTV1hVPWtbpP7Nomg0CbD8TtCpyOYbeq8+OpHb5eWbDsXXCeLHO4epgthLtbQHiBE8spXqEtx/h0D6MzxsIlN4pa8gz51jV+oWzQQNnpQCeaKKLaxcpMfhGXnZv4BK1Rg+TAgUFE9RtExcKjteTL2hB9tKT41C7antdQdhLfVXWUbsJ/q5b62iDZnnZUAANXHnWp/9RI2YyKSn70SVu4Bag+fxA/23OqjCHSbK5RMiNOQKV+Bs7uugaAsMKkH6lGBBIbNDkTXGZ4n1+KbqFwe1kV9oCaPJ+siwXESEqvY0jaLVNAqUATQZjnIMFIYwARw41FTuduxW1IOF7MdP7R3EqOvnqNir2lXW1UfRlHlOtMC4w/tXk8xqJR/HVlZrnltKkMZB5zfyDNvnt02jbOKJcORjmOeVvL+xoCdSXwZclfJzRkC31l43+9jSu4X8RPUfuJILRMHf2e1A0NU7Mwds7h+S//5AD0yUJlPtFFzLvriuD1SMTRXiSwN4pNWBi6UIsPzScRpyLMc8hUE8Bi8jJtGk4e0=; NDARK=N; page_uid=hrKUflprvN8ssNc4Muwssssss3R-382317; BMR=',
    'referer': 'https://n.news.naver.com/article/028/0002595736',
    'sec-ch-ua': '"Whale";v="3", " Not;A Brand";v="99", "Chromium";v="102"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'script',
    'sec-fetch-mode': 'no-cors',
    'sec-fetch-site': 'same-site',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
}
# 여럿 user-agent 목록
# Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.101 Safari/537.36
# Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0
# Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:46.0) Gecko/20100101 Firefox/46.0
# Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.101 Safari/537.36
### 페이지 설정에 따른 URL 추출
def get_urls_from_navernews_bypage(search_query, start_pg, end_pg, sort=0):
    start = datetime.datetime.now()
    if start_pg == end_pg:
        start_page = str((start_pg-1) * 10 + 1)
        url = f"https://search.naver.com/search.naver?where=news&ie=utf8&sm=tab_pge&query={search_query}&sort={sort}&start={start_page}"
        end = datetime.datetime.now()
        print('URL Extracting Time: ', end-start)
        return url
    else:
        urls = []
        for i in range(start_pg, end_pg + 1):
            page = str((i-1) * 10 + 1)
            url = f"https://search.naver.com/search.naver?where=news&ie=utf8&sm=tab_pge&query={search_query}&sort={sort}&start={page}"
            urls.append(url)
        end = datetime.datetime.now()
        print('URL Extracting Time: ', end-start)
        return urls   
    
### 날짜 설정에 따른 URL 추출
def get_urls_from_navernews_bydate(search_query, start_date, end_date, sort=0, maxpage=1000):
    start = datetime.datetime.now()
    i, urls = 1, []
    while True:
        page = str((i-1) * 10 + 1)
        url = f"https://search.naver.com/search.naver?where=news&ie=utf8&sm=nws_hty&query={search_query}&sort={sort}&nso=so%3Ar%2Cp%3Afrom{start_date}to{end_date}&start={page}"
        urls.append(url)
        if i == maxpage+1:
            end = datetime.datetime.now()
            print('URL Extracting Time: ', end-start)
            return urls
        else:
            i = i + 1

### URL에 담긴 댓글 추출
def get_comments_from_navernews(url):
    # setting
    url = url.split('?')[0]
    oid_1, oid_2 = url.split('/')[-1], url.split('/')[-2]

    i, comments = 1, [] #모든 댓글을 담는 리스트
    while True:
        params = {
            'ticket': 'news',
            'templateId': 'default_society',
            'pool': 'cbox5',
            'lang': 'ko',
            'country': 'KR',
            'objectId': f'news{oid_2},{oid_1}',
            'pageSize': '100',
            'indexSize': '10',
            'page': str(i),
            'currentPage': '0',
            'moreParam.direction': 'next',
            'moreParam.prev': '10000o90000op06guicil48ars',
            'moreParam.next': '1000050000305guog893h1re',
            'followSize': '100',
            'includeAllStatus': 'true',
        }

        response = requests.get('https://apis.naver.com/commentBox/cbox/web_naver_list_jsonp.json', params=params, headers=headers)
        response.encoding = "UTF-8-sig"
        res = response.text.replace("_callback(","")[:-2]
        temp=json.loads(res)
        try :
            comment = list(pd.DataFrame(temp['result']['commentList'])['contents'])
            for j in range(len(comment)):
                comments.append(comment[j])
            else:
                i+=1
        except :
            break

    return comments

### 시간표시 24H 기준으로 형식 변환
def get_update_timeformat(time_origin):
    try:
        # 영문화
        if '오전' in time_origin:
            time_origin = time_origin.replace('오전', 'AM')
        elif '오후' in time_origin:
            time_origin = time_origin.replace('오후', 'PM')
        # 형식 변환
        time_final = datetime.datetime.strptime(time_origin, '%Y.%m.%d. %p %I:%M')
        time_final = time_final.strftime('%Y-%m-%d %H:%M:%S')
        
        return time_final
    
    except ValueError:
        
        return time_origin  # 날짜 형식이 아닌 경우 그대로 반환

### URL에 담긴 댓글을 포함한 뉴스정보 추출
def get_data_from_navernews(url):
    start = datetime.datetime.now()
    time_articles, press_articles, category_articles, title_articles, content_articles, comment_articles = [], [], [], [], [], []
    url_articles, url_articles_naver = [], []
    for pg in tqdm(url):
        response = requests.get(pg, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        t.sleep(random.uniform(2, 4)) # 요청 사이 무작위로 시간 간격 두게 하기 (2~4초), 랜덤으로 하면 동일한 시간 간격으로 들어와 비정상적인 접근이라고 생각하는 걸 방지해줌.
        
        # 페이지 내 기사시간, 언론사, 제목, 기사URL 불러오기
        news_elements = soup.select('div.news_wrap.api_ani_send')
        for element in news_elements:
            time = element.select_one('span.info').text.strip()
            time_articles.append(time)
            press = element.select_one('a.info.press').text.strip()
            press_articles.append(press)
            title = element.select_one('a.news_tit').text.strip()
            title = re.sub(pattern='<[^>]*>', repl='', string=str(title))
            title_articles.append(title)
            each_url = element.select_one('a.news_tit')['href']
            url_articles.append(each_url)     
            
        # 기사URL 중 네이버뉴스 주소로 반영된 것은 업데이트
        article_elements = soup.select("div.group_news > ul.list_news > li div.news_area > div.news_info > div.info_group > a.info")
        for article in article_elements:
            ## 네이버 URL이 없는 경우는 빈칸 저장
            if "news.naver.com" not in article.attrs['href']:
                url_articles_naver.append(article.attrs['href'])
                category_articles.append([])
                content_articles.append([])
                comment_articles.append([])
            ## 네이버 URL이 있는 경우 내용 저장
            else:   
                # 링킹
                url_articles_naver[-1] = article.attrs['href'] 
                article_response = requests.get(url_articles_naver[-1], headers=headers, verify=False)
                article_soup = BeautifulSoup(article_response.text, 'html.parser')
                # 카테고리 불러오기
                category = article_soup.select_one('#_LNB > ul > li.Nlist_item._LNB_ITEM.is_active > a > span')
                if category != None:
                    category_articles[-1].append(str(category).split('menu">')[1].split('</span>')[0])
                else:
                    category_articles[-1].append([])
                # 본문 불러오기
                content = article_soup.select("article#dic_area")    # "div#dic_area"에서 "article#dic_area"로 변경
                if content == []:
                    content = article_soup.select("#articeBody")
                ## 본문 전처리 정리
                content = ''.join(str(content))
                content = re.sub(pattern='<[^>]*>', repl='', string=content)
                content = content.replace("""[\n\n\n\n\n// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback() {}""", '')
                content = content.replace('\n', ' ').replace('\t', ' ')
                content_articles[-1] = content
                # 댓글 불러오기
                comment = get_comments_from_navernews(url_articles_naver[-1])
                comment_articles[-1] = comment         
                # 기사시간 업데이트
                try:
                    time_html = article_soup.select_one("div#ct> div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp > div > span")
                    time = time_html.attrs['data-date-time']
                    time_articles[len(comment_articles)-1] = get_update_timeformat(time)
                except AttributeError:
                    time = article_soup.select_one("#content > div.end_ct > div > div.article_info > span > em")
                    time = re.sub(pattern='<[^>]*>',repl='',string=str(time))
                    time_articles[len(comment_articles)-1] = get_update_timeformat(time)
            
        # 마지막 페이지면 종료
        if len(news_elements) < 10: 
            break
                  
    # 정리
    df_news = pd.DataFrame({'Date':time_articles,
                            'Press':press_articles,
                            'Category':category_articles,
                            'Title':title_articles,
                            'Content':content_articles,
                            'Comment':comment_articles,
                            'URL_Origin':url_articles,
                            'URL_Naver':url_articles_naver})
    end = datetime.datetime.now()
    print('News Info Extracting Time: ', end-start)
    print('Size of News Data: ', df_news.shape[0])
    
    return df_news
