/**
 * Puppeteer 네이버 뉴스 크롤러
 * 
 * 키워드와 기간을 기반으로 네이버 뉴스 검색 결과를 수집하고
 * 기사 본문, 발행일(최종 수정일), 기자 정보, 카테고리, 댓글 등을 저장합니다.
 * 
 * 데이터 추출량에 따라 서버 딜레이 발생 가능성이 있음으로 timeout 조정이 필요합니다.
 * csv 파일 활용 시 댓글 수 많은 기사에 대한 줄바꿈 문제 발생 가능성으로 json 파일 활용을 추천합니다.
 */

// ===============================
// 모듈 및 라이브러리 임포트
// ===============================
const puppeteer = require('puppeteer');
const fs = require('fs');
const { Parser } = require('json2csv');
const cliProgress = require('cli-progress');


// ===============================
// 기본 설정 (키워드, 기간)
// ===============================
/**
 * 기존 입력 방식
 */
// const keywords = ['AI 디지털교과서'];
// const START_DATE = '2022.12.01';
// const END_DATE = '2022.12.31';

/**
 * config.json 파일 내 입력 방식
 * keywords 리스트 내 추가 가능 -> 여러 개 키워드 순차적 크롤링
 * 예: "keywords": ["AI 디지털교과서", "사회공헌", "머신러닝"]
 */
const config = JSON.parse(fs.readFileSync('config.json', 'utf-8'));

const keywords = config.keywords;
const START_DATE = config.startDate;
const END_DATE = config.endDate;

/**
 * 네이버 뉴스 검색 링크 nso 파라미터 부분에 . 없는 날짜형식 필요 + 파일명 날짜 부분
 */
const DS_NO_DOT = START_DATE.replace(/\./g, '');
const DE_NO_DOT = END_DATE.replace(/\./g, '');


// ===============================
// 메인 실행 함수
// ===============================
/**
 * @function main
 * 메인 실행:
 * 크롤링 후 전체 기사와 네이버 뉴스 기사 별도 저장
 */
async function main() {
  const startTime = Date.now();

  for (const keyword of keywords) {
    console.log(`Crawling articles for keyword: "${keyword}"`);
    try {
      const articles = await crawlArticlesByKeyword(keyword);
      console.log(`Total articles collected: ${articles.length}`);
      const naverCount = articles.filter(article => article.type === 'naver').length;
      console.log(`Naver news articles: ${naverCount}`);
        
      saveArticlesToFile(keyword, START_DATE, END_DATE, articles);
      saveNaverArticlesToFile(keyword, START_DATE, END_DATE, articles);

    } catch (err) {
      console.error(`Error occurred while crawling "${keyword}":`, err);
    }
  }

  const endTime = Date.now();
  const elapsedMs = endTime - startTime;
  const seconds = Math.floor((elapsedMs / 1000) % 60);
  const minutes = Math.floor((elapsedMs / (1000 * 60)) % 60);
  const hours = Math.floor(elapsedMs / (1000 * 60 * 60));

  console.log(`Total time: ${hours}h ${minutes}m ${seconds}s (${elapsedMs} ms)`);
  console.log('Program finished.');
}

main().catch(console.error);

// ===============================
// 네이버 뉴스 검색 결과 크롤링
// ===============================
/**
 * @function crawlArticlesByKeyword
 * 키워드 별 네이버 뉴스 검색 결과 크롤링, 네이버 뉴스 기사 상세 처리
 */
async function crawlArticlesByKeyword(keyword) {
  const browser = await puppeteer.launch({ headless: false });
  const page = await browser.newPage();

  // 네이버 뉴스 검색 (sort=0:관련도순, pd=3:날짜 직접 지정 -> 변경 가능)
  const query = encodeURIComponent(keyword);
  const searchUrl = `https://search.naver.com/search.naver?where=news&query=${query}&sm=tab_opt&sort=0&pd=3&ds=${START_DATE}&de=${END_DATE}&nso=so%3Ar%2Cp%3Afrom${DS_NO_DOT}to${DE_NO_DOT}`;
  
  await page.goto(searchUrl, { waitUntil: 'networkidle2' });
  await scrollPageToBottom(page);

  // 기사 추출 (메인+관련)
  const articles = await extractArticlesFromSearchPage(page);
  await page.close();

  console.log(`Processing detailed crawling for ${articles.length} articles...`);
  const progressBar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
  progressBar.start(articles.length, 0);

  // 네이버 뉴스 상세 크롤링 처리
  for (let i = 0; i < articles.length; i++) {
    const art = articles[i];
    try {
      if (art.type === 'naver') {
        const detailLink  = art.link && art.link.includes('news.naver.com')
                          ? art.link
                          : art.naverNewsLink;
        const detail = await crawlArticleDetail(detailLink, browser);
        articles[i] = { ...art, ...detail };
      } else {
        console.log(`Skipping non-Naver article: ${art.link}`);
      }
    } catch (err) {
      console.warn(`Error crawling article (${art.link}): ${err.message}`);
    }

    progressBar.update(i + 1);
  }
    
  progressBar.stop();

  await browser.close();
  return articles;
}


// ===============================
// 검색 결과 페이지 스크롤 및 추출
// ===============================
/**
 * @function scrollPageToBottom 
 * - 페이지 끝까지 스크롤
 */
async function scrollPageToBottom(page) {
  let lastHeight = await page.evaluate(() => document.body.scrollHeight);
//   console.log("Initial height:", lastHeight);
  while (true) {
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await new Promise(resolve => setTimeout(resolve, 300)); 
    const newHeight = await page.evaluate(() => document.body.scrollHeight);
//     console.log("After scroll - lastHeight:", lastHeight, "newHeight:", newHeight);      
    if (newHeight === lastHeight) break;
    lastHeight = newHeight;
  }
}

/**
 * @function extractArticlesFromSearchPage
 * - 메인기사 li.bx & 하위 관련기사 div.news_cluster > li.sub_bx
 */
async function extractArticlesFromSearchPage(page) {
  const articles = await page.$$eval('li.bx', items => {
    const results = [];
    items.forEach(item => {
      // 메인
      const pressEl = item.querySelector('div.news_info > div.info_group > a.info.press');
      const press = pressEl ? pressEl.innerText.trim() : '';
      const titleEl = item.querySelector('div.news_contents > a.news_tit');
      const title = titleEl ? titleEl.innerText.trim() : '';        
      const dateEl = item.querySelector('div.news_info > div.info_group > span.info');
      const date = dateEl ? dateEl.innerText.trim() : '';
      const excerptEl = item.querySelector('div.news_contents > div > div > a');
      const excerpt = excerptEl ? excerptEl.innerText.trim() : '';        
      const link = titleEl ? titleEl.href : '';      
      const navLinkEl = item.querySelector('div.news_info > div.info_group > a:nth-child(3)');
      const naverNewsLink = navLinkEl ? navLinkEl.href : '';
      const type = (naverNewsLink.includes('news.naver.com')) ? 'naver' : 'external';
      
      results.push({
        isSubArticle: false,
        parentTitle: null,
        press,
        date,          
        title,
        excerpt,
        link,
        naverNewsLink,
        type
      });

      // 관련(하위 목록)
      const clusterEl = item.querySelector('div.news_cluster');
      if (clusterEl) {
        const subs = clusterEl.querySelectorAll('li.sub_bx');
        subs.forEach(sub => {
          const subTitleEl = sub.querySelector('a.sub_tit');
          const subTitle = subTitleEl ? subTitleEl.innerText.trim() : '';   
          const subLink = subTitleEl ? subTitleEl.href : '';
          const subPressEl = sub.querySelector('cite.sub_txt.press');
          const subPress = subPressEl ? subPressEl.innerText.trim() : '';
          const subDateEl = sub.querySelector('div.sub_area > span.sub_txt');
          const subDate = subDateEl ? subDateEl.innerText.trim() : '';     
          const subNaverEl = sub.querySelector('div.sub_area > a.sub_txt');
          const subNaverLink = subNaverEl ? subNaverEl.href : '';
          const subType = (subNaverLink && (subNaverLink.includes('news.naver.com') || subNaverLink.includes('n.news.naver.com')))
            ? 'naver'
            : 'external';

          results.push({
            isSubArticle: true,
            parentTitle: title,
            press: subPress,
            date: subDate,
            title: subTitle,             
            excerpt: '',
            link: subLink,
            naverNewsLink: subNaverLink,
            type: subType
          });
        });
      }
    });
    return results;
  });
  return articles;
}


// ===============================
// 네이버 뉴스 상세 페이지 크롤링
// ===============================
/**
 * @function crawlArticleDetail
 *  - 기사 페이지 로드
 *  - 기사 정보 추출
 */
async function crawlArticleDetail(newsUrl, browser) {
  const page = await browser.newPage();
  await page.goto(newsUrl, { waitUntil: 'networkidle2' });
  
  // 기자
  let journalist = '';
  try {
    const journalistList = await page.$$eval(
      '#ct .media_end_head_journalist .media_end_head_journalist_name',
      nodes => nodes.map(el => el.innerText.trim()).filter(Boolean)
    );
    journalist = journalistList .join(', ');
  } catch {
    journalist = '';
  }

  // 날짜 (수정일(=최종발행일) > 없으면 입력일)
  let naverDate = '';
  try {
    const hasModifyDate  = await page.$('span._ARTICLE_MODIFY_DATE_TIME');
    if (hasModifyDate ) {
      naverDate = await page.$eval('span._ARTICLE_MODIFY_DATE_TIME', el => el.getAttribute('data-modify-date-time'));
    } else {
      naverDate = await page.$eval('span._ARTICLE_DATE_TIME', el => el.getAttribute('data-date-time'));
    }
  } catch {
    naverDate = '';
  }
  
  // 본문
  let fullContent = '';
  try {
    fullContent = await page.$eval('#dic_area', el => el.innerText.trim());
  } catch (e) {
    try {
      fullContent = await page.$eval('#newsct_article', el => el.innerText.trim());
    } catch (e2) {
      fullContent = '';
    }
  }

  // 카테고리
  let category = '';
  try {
    category = await page.$eval('#contents div.media_end_categorize a em', el => el.innerText.trim());
  } catch (e) {
    category = '';
  }

  // 댓글
  let comments = [];
  try {
    // 댓글 수 가져오기  
    const commentCountText = await page.$eval('#comment_count', el => el.innerText.trim());
    const commentCount  = parseInt(commentCountText.replace(/,/g, ''), 10) || 0;
    // 5개 초과 시 댓글 전용 페이지로 이동하여 수집      
    if (commentCount  > 5) {
      const commentUrl  = newsUrl.replace('/mnews/article/', '/mnews/article/comment/');
      await page.goto(commentUrl , { waitUntil: 'networkidle2' });
      await loadAllComments(page);
      comments = await extractAllComments(page);
    } else {
      comments = await extractAllComments(page);
    }
  } catch (e) { comments = [];
    console.warn('Error during comment processing:', e);
    comments = [];
  }
  
  await page.close();
  return {
    journalist,
    naverDate,
    fullContent,
    category,
    commentCount: comments.length,
    comments
  };
}


// ===============================
// 댓글 전용 로직
// ===============================
/**
 * @function loadAllComments
 * - 댓글 전용 페이지에서 '더보기' 무한 클릭
 */
async function loadAllComments(page) {
  const MORE_BTN_SELECTOR = '#cbox_module > div.u_cbox_wrap.u_cbox_ko.u_cbox_type_sort_new > div.u_cbox_paginate > a > span > span > span.u_cbox_page_more';
    
  while (true) {
    try {
      const moreBtn = await page.$(MORE_BTN_SELECTOR);
      if (!moreBtn) break;
      await page.waitForSelector(MORE_BTN_SELECTOR, { visible: true, timeout: 1000 });
      await page.evaluate(selector => {
        const btn = document.querySelector(selector);
        if (btn) {
          btn.scrollIntoView({ behavior: 'smooth', block: 'center' });
          btn.click();
        }
      }, MORE_BTN_SELECTOR);
      await new Promise(resolve => setTimeout(resolve, 500));
    } catch (error) {
      console.log('No more "더보기" button. All comments loaded.:');
      break;
    }
  }
}


/**
 * @function extractAllComments
 * - 댓글의 모든 정보 추출
 */
async function extractAllComments(page) {
  return await page.$$eval('#cbox_module_wai_u_cbox_content_wrap_tabpanel > ul > li.u_cbox_comment', items => {
    return items.map(item => {
      const nickEl = item.querySelector('.u_cbox_nick');
      const nick = nickEl ? nickEl.innerText.trim() : '';
      const dateEl = item.querySelector('.u_cbox_date');
      const date = dateEl ? dateEl.innerText.trim() : '';
      const contentEl = item.querySelector('.u_cbox_text_wrap > span.u_cbox_contents');
      const content = contentEl ? contentEl.innerText.trim() : '';
      const recommEl = item.querySelector('.u_cbox_btn_recomm > em');
      const recomm = recommEl ? recommEl.innerText.trim() : '0';
      const unrecommEl = item.querySelector('.u_cbox_btn_unrecomm > em');
      const unrecomm = unrecommEl ? unrecommEl.innerText.trim() : '0';
      return { nick, date, content, recomm, unrecomm };
    });
  });
}


// ===============================
// 파일 저장 함수
// ===============================
/**
 * @function cleanFileName
 * 파일명에 포함될 수 없는 특수문자 제거
 */
function cleanFileName(filename) {
  return filename.replace(/[\/\\:*\?"<>\|]/g, '-');
}

/**
 * @function saveArticlesToFile
 * 전체 기사 JSON/CSV로 저장
 */
function saveArticlesToFile(keyword, startDate, endDate, articles) {
  const period = `${startDate.replace(/\./g, '')}-${endDate.replace(/\./g, '')}`;
  const baseFileName = `${cleanFileName(keyword)}_${period}`;

  // JSON 저장
  const jsonFileName = `${baseFileName}.json`;
  fs.writeFileSync(jsonFileName, JSON.stringify(articles, null, 2), 'utf-8');
  console.log(`All articles saved to ${jsonFileName}`);

  // CSV 저장
  if (articles.length > 0) {
    const csvFileName = `${baseFileName}.csv`;
    const parser = new Parser({
      fields: [
        'isSubArticle',
        'parentTitle',
        'press',
        'date',
        'title',
        'excerpt',
        'link',
        'naverNewsLink',
        'type',
        'journalist',
        'naverDate',
        'fullContent',
        'category',
        'commentCount'
      ]
    });
   
    const csv = parser.parse(articles);
    const utf8BOM = '\uFEFF';
    fs.writeFileSync(csvFileName, utf8BOM + csv, 'utf-8');
    console.log(`All articles saved to ${csvFileName}`);
  } else {
    console.log(`No articles to save for keyword ${keyword}`);
  }
}

/**
 * @function saveNaverArticlesToFile
 * 네이버 뉴스만 별도 저장
 */
function saveNaverArticlesToFile(keyword, startDate, endDate, articles) {
  const naverArticles = articles.filter(article => article.type === 'naver');
  if (naverArticles.length === 0) {
    console.log(`No Naver articles found for keyword: ${keyword}`);
    return;
  }

  const period = `${startDate.replace(/\./g, '')}-${endDate.replace(/\./g, '')}`;
  const baseFileName = `${cleanFileName(keyword)}_naver_${period}`;

  // 댓글 텍스트(commentTexts) 생성
  naverArticles.forEach(article => {
//     if (article.comments && Array.isArray(article.comments)) {
//       article.commentTexts = article.comments.map(c => c.content);
//     } else {
//       article.commentTexts = [];
//     }
//   });
    if (Array.isArray(article.comments)) {
      article.commentTexts = article.comments.map(c => c.content);
    } else {
      article.commentTexts = [];
    }
  });
    
  // JSON 저장
  const jsonFileName = `${baseFileName}.json`;
  fs.writeFileSync(jsonFileName, JSON.stringify(naverArticles, null, 2), 'utf-8');
  console.log(`Naver articles saved to ${jsonFileName}`);

  // CSV 저장
  const csvFileName = `${baseFileName}.csv`;
  const parser = new Parser({
    fields: [
      'isSubArticle',
      'parentTitle',
      'press',
      'date',
      'title',
      'excerpt',
      'link',
      'naverNewsLink',
      'type',
      'journalist',
      'naverDate',
      'fullContent',
      'category',
      'commentCount',
      'comments',
      'commentTexts'
    ],
   });
            
  const csv = parser.parse(naverArticles);
  const utf8BOM = '\uFEFF';
  fs.writeFileSync(csvFileName, utf8BOM + csv, 'utf-8');
  console.log(`Naver articles saved to ${csvFileName}`);
}


