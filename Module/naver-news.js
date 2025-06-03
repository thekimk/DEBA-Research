/**
 * Puppeteer 네이버 뉴스 크롤러
 *
 * 키워드와 기간을 기반으로 네이버 뉴스 검색 결과를 수집하고
 * 상위 및 하위 뉴스 그룹을 처리 및 각 고유 네이버 뉴스 링크 접속으로
 * 기사 본문, 발행일(최종 수정일), 기자 정보, 카테고리, 댓글 정보 등을 추출 후 저장합니다.
 *
 */
// ===============================
// 모듈 및 라이브러리 임포트
// =============================== 
const puppeteer = require('puppeteer');
const fs = require('fs');
const { Parser } = require('json2csv');
const cliProgress = require('cli-progress');
const pLimit = require('p-limit');
const limit = pLimit(5);  // 병렬 처리 수 제한(각 네이버 링크들에 대한 동시 접속 작업에서 작동)

// ===============================
// 기본 설정 (키워드, 기간)
// ===============================
// // 기존 입력 방식
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
// 셀렉터 정의(element&selector)
// ===============================
/**
 * 개발자 도구에서 각 요소의 selector를 copy해서 고유한 부분 추출+조합해서 구성
 * 업데이트에 따라 각 요소 태그(클래스명)가 다양
 * 추출 결과를 통해 빈칸일 경우 보완하는 방향으로(해당 기간, 기사 링크 특성에 따라 다른 태그 추가)
 */

const SELECTORS = {
  newsGroup: 'div.sds-comps-vertical-layout.sds-comps-full-layout.UC1lc0LnKWszTX7CYO7K', 
  relatedLink: 'a[href*="related=1"]',
  subNewsOuter: 'div.sds-comps-base-layout.sds-comps-full-layout.JT4g6KsELnSY85CYAym9',
  subNewsItem: 'div.sds-comps-base-layout.sds-comps-full-layout.Eb67Vg8smoO6HeVy39Y9',
  top: {
    profile: 'div[data-sds-comp="Profile"]',
    content: 'div.sds-comps-base-layout.sds-comps-full-layout.HyuoyN_3xv7CrtOc6W9S',
    press: '.sds-comps-profile-info-title span', 
    date: ['span.sds-comps-profile-info-subtext > span.sds-comps-text-type-body2',
           'span.sds-comps-text.sds-comps-text-type-body2.sds-comps-text-weight-sm'].join(', '),
    title: 'span.sds-comps-text-type-headline1.sds-comps-text-ellipsis-1',
    summary: 'span.sds-comps-text-ellipsis-3.sds-comps-text-type-body1',
    pressUrl: 'a[href^="http"]:not([href*="n.news.naver.com"])',
    naverUrl: '.sds-comps-profile-info-subtext a[href*="n.news.naver.com"]'
  },
  sub: {
    press: ':scope .sds-comps-profile-info-title > span.sds-comps-text-type-body2',
    date: '.sds-comps-profile-info-subtext span',
    title: 'span.sds-comps-text-type-body2.sds-comps-text-ellipsis-1',
    pressUrl: 'a[href^="http"]:not([href*="naver"])',
    naverUrl: 'a[href*="n.news.naver.com"]'
  },
  article: {
    body: '#newsct_article, #dic_area, #comp_news_article',
    journalist: '.media_end_head_journalist_name, span.byline_s',
    category: '.media_end_categorize a em',
    dateStamp: '.media_end_head_info_datestamp_time',
    commentCount: '#comment_count',
    commentItems: '#cbox_module_wai_u_cbox_content_wrap_tabpanel > ul > li.u_cbox_comment',
    articleTitle: '#title_area > span'
  }
};

// ===============================
// 메인 실행 함수
// ===============================
// utility functions
// Url 정규화
function normalizeUrl(u) {
  try {
    const url = new URL(u);
    url.search = '';
    url.pathname = url.pathname.replace(/\/$/, '');
    return url.toString();
  } catch {
    return u;
  }
}

// url 중복 검사 + 반영
function addIfNew(url, item, targetArray, seenSet) {
  const norm = normalizeUrl(url);
  if (!seenSet.has(norm)) {
    seenSet.add(norm);
    targetArray.push(item);
  }
}

/**
 * 크롤링 후 전체 기사와 네이버 뉴스 기사 별도 저장 + 병렬 처리 및 키워드 검색 후 나타나는 페이지 구조로 변경
 */
(async () => {
  const crawlStart = Date.now();     
  const browser = await puppeteer.launch({headless: false});
  const page = await browser.newPage();
  // URL 구성 및 이동
  for (const KEYWORD of keywords) {
    const searchUrl = ['https://search.naver.com/search.naver?where=news',
                        `&query=${encodeURIComponent(KEYWORD)}`,
                        `&pd=3&ds=${START_DATE}&de=${END_DATE}`,
                        `&nso=so%3Ar%2Cp%3Afrom${DS_NO_DOT}to${DE_NO_DOT}`,
                        '&sm=tab_opt&sort=0'].join('');
      await page.goto(searchUrl, {waitUntil: 'networkidle2'});
      await scrollPageToBottom(page);            
      const allNews = [];
      const naverNews = [];
      const seenUrls = new Set();
      const groups = await page.$$(SELECTORS.newsGroup);
      const groupBar = new cliProgress.SingleBar({format: 'News groups {bar} {value}/{total}'}, cliProgress.Presets.shades_classic);
      groupBar.start(groups.length, 0);
      /**
      * @function processGroup
      * 상위 및 하위 뉴스 그룹 처리 + 관련 뉴스 페이지 링크 있는 그룹 처리
      */
      async function processGroup(el) {
        const top = await extractTopNews(el, browser);
        if (top) {
          allNews.push(top);
          if (top.naverUrl) {
            addIfNew(top.naverUrl, top, naverNews, seenUrls);
          }
        }
        for (const subOuter of await el.$$(SELECTORS.subNewsOuter)) {
          const subItems = await subOuter.$$(SELECTORS.subNewsItem);
          for (const subEl of subItems) {
            const sub = await extractSubNews(subEl, browser);
            if (sub) {
              allNews.push(sub);
              if (sub.naverUrl) {
                addIfNew(sub.naverUrl, sub, naverNews, seenUrls);
              }
            }
          }
        }
      }    
      for (const group of groups) {
        const relatedEl = await group.$(SELECTORS.relatedLink);
        if (relatedEl) {
          const url  = await relatedEl.evaluate(el => el.href);
          const relatedPage = await browser.newPage();
          await relatedPage.goto(url, {waitUntil:'networkidle2'});
          await scrollPageToBottom(relatedPage);
          for (const relatedGroups of await relatedPage.$$(SELECTORS.newsGroup)) {
            await processGroup(relatedGroups);
          }
          await relatedPage.close();
        } else {
          await processGroup(group);
        }
        groupBar.increment();
      }
      groupBar.stop();
      // 5개 병렬 탭으로 네이버뉴스 상세 크롤링 
      const detailBar = new cliProgress.SingleBar({format: 'Naver news {bar} {value}/{total}'},
        cliProgress.Presets.shades_classic);
        detailBar.start(naverNews.length, 0);
        const groupTasks = naverNews.map((item, idx) => limit(async () => {
          const groupDetails = await extractArticleDetails(browser, item.naverUrl);  
          if (groupDetails.title && groupDetails.title.length > 10) { // 네이버 뉴스 페이지에서 제목 대체
            item.title = groupDetails.title; // 네이버 뉴스 링크가 있는 경우 그 안에서의 제목으로 대체(검색 페이지에서 짤리는 경우 존재해서)
          }
          Object.assign(naverNews[idx], groupDetails); // 원본 배열에 정보 병합 
          detailBar.increment();
        })
      );
      await Promise.all(groupTasks);
      detailBar.stop();
      // 날짜 형식 통일
      for (const item of allNews) {
        if (item.date) item.date = normalizeDate(item.date);
        if (item.createdDate) item.createdDate = normalizeDate(item.createdDate);
        if (item.updatedDate) item.updatedDate = normalizeDate(item.updatedDate);
        if (item.comments) {
          item.comments = item.comments.map(c => ({...c, date: normalizeCommentDate(c.date)}));
        }
      }
      for (const item of naverNews) {
        if (item.date) item.date = normalizeDate(item.date);
        if (item.createdDate) item.createdDate = normalizeDate(item.createdDate);
        if (item.updatedDate) item.updatedDate = normalizeDate(item.updatedDate);
        if (item.comments) {
          item.comments = item.comments.map(c => ({...c, date: normalizeCommentDate(c.date)}));
        }
      }
      // 파일 저장
      const basename = `${KEYWORD}_${DS_NO_DOT}-${DE_NO_DOT}`;
      fs.writeFileSync(`./${basename}_all.json`, JSON.stringify(allNews,   null, 2), 'utf-8');
      fs.writeFileSync(`./${basename}_all.csv`, '\uFEFF' + new Parser().parse(allNews), 'utf-8');
      fs.writeFileSync(`./${basename}_naver.json`, JSON.stringify(naverNews, null, 2), 'utf-8');
      fs.writeFileSync(`./${basename}_naver.csv`, '\uFEFF' + new Parser().parse(naverNews), 'utf-8');
    }
    // 크롤링 소요 시간 계산 & 출력
    await browser.close();
    const crawlEnd = Date.now();           
    const elapsed = crawlEnd - crawlStart;
    const seconds = Math.floor((elapsed / 1000) % 60);
    const minutes = Math.floor((elapsed / (1000 * 60)) % 60);
    const hours = Math.floor(elapsed / (1000 * 60 * 60));
    console.log('Program finished.');
    console.log(`Total time: ${hours}h ${minutes}m ${seconds}s (${elapsed}ms)`);
  }
)();

// ===============================
// 뉴스 추출
// ===============================
/**
 * @function extractTopNews
 * 상위 뉴스 그룹에서 정보 추출
 */
async function extractTopNews(group, browser) {
  try {
    const profile = await group.$(SELECTORS.top.profile); // PROFILE block 부분만 상위 뉴스로(하위 뉴스와 혼합 방지)
    if (!profile) return null;
    const pressEl = await profile.$(SELECTORS.top.press);
    const press = pressEl ? clean(await pressEl.evaluate(e => e.innerText)): '';
    const dateEls = await profile.$$(SELECTORS.top.date);
    let date = '';
    for (const el of dateEls) {
      const txt = (await el.evaluate(e => e.innerText)).trim();
      if (/^\d{4}\.\d{2}\.\d{2}/.test(txt)) {
        date = txt;
        break;
      }
      if (/\d+\s*(일|주|시간|분) 전/.test(txt)) {
        date = formatYMD(relativeDate(txt));
        break;
      }
    }
    const naverEl = await profile.$(SELECTORS.top.naverUrl);
    const naverUrl = naverEl ? await naverEl.evaluate(e => e.href): '';
    const content = await group.$(SELECTORS.top.content);
    if (!content) return null;
    const titleEl = await content.$(SELECTORS.top.title);
    const title = titleEl ? clean(await titleEl.evaluate(e => e.innerText)): '';
    const summaryEl = await content.$(SELECTORS.top.summary);
    const summary = summaryEl ? clean(await summaryEl.evaluate(e => e.innerText)): '';
    const pressUrlEl = await content.$(SELECTORS.top.pressUrl);
    const pressUrl = pressUrlEl ? await pressUrlEl.evaluate(e => e.href): '';
    return {press, date, title, summary, pressUrl, naverUrl};
  } catch {
    return null;
  }
}

/**
 * @function extractSubNews
 * 하위 뉴스 그룹에서 정보 추출
 */
async function extractSubNews(subGroup, browser) {
  try {
    const pressEl = await subGroup.$(SELECTORS.sub.press);   
    const press = pressEl ? clean(await pressEl.evaluate(e => e.innerText)): '';
    const dateEls = await subGroup.$$(SELECTORS.sub.date);
    let date = '';
    for (const el of dateEls) {
      const txt = (await el.evaluate(e => e.innerText)).trim();
      if (/^\d{4}\.\d{2}\.\d{2}/.test(txt)) {
        date = txt;
        break;
      }
      if (/\d+\s*(일|주|시간|분) 전/.test(txt)) {
        date = formatYMD(relativeDate(txt));
        break;
      }
    } 
    const naverUrlEl = await subGroup.$(SELECTORS.sub.naverUrl);
    const naverUrl = naverUrlEl ? await naverUrlEl.evaluate(e => e.href): '';       
    const titleEl = await subGroup.$(SELECTORS.sub.title);
    const title = titleEl ? clean(await titleEl.evaluate(e => e.innerText)): '';
    const pressUrlEl = await subGroup.$(SELECTORS.sub.pressUrl);
    const pressUrl = pressUrlEl ? await pressUrlEl.evaluate(e => e.href): '';
    return {press, date, title, summary: '', pressUrl, naverUrl};
  } catch {
    return null;
  }
}

/**
 * @function extractArticleDetails
 * 네이버 뉴스 페이지에서 상세 정보(기자, 날짜, 본문, 카테고리, 댓글) 추출
 * 수정일이 없을 경우 createdDate을 그대로 updatedDate에 반영시켰기 때문에 분석에 활용시 updatedDate 컬럼을 활용하는 것을 권장
 */
async function extractArticleDetails(browser, url) {
  const page = await browser.newPage();
  await page.goto(url, {waitUntil:'networkidle2'});
  let journalist = '';
  let createdDate = '';
  let updatedDate = '';
  let body = '';
  let category = '';
  let commentCount = 0;
  let comments = [];
  let title = '';
  // 기자
  try {
    const js = await page.$$(SELECTORS.article.journalist);
    const names = [];
    for (const j of js) {
      const name = await j.evaluate(e => e.innerText.trim());
      if (name && !names.includes(name)) {
        names.push(name);
        break;
      }
    }
    journalist = names.join(', ')
  } catch {}
  // 날짜 (created&updated)
  try {
    const stamps = await page.$$(SELECTORS.article.dateStamp);
    if (stamps[0]) createdDate = await stamps[0].evaluate(e => e.getAttribute('data-date-time') || e.innerText.trim());
    if (stamps[1]) updatedDate = await stamps[1].evaluate(e => e.getAttribute('data-modify-date-time') || e.innerText.trim());
    else updatedDate = createdDate;
  } catch {}
  // 제목(overwrite)
  try {
    const titleEl = await page.$(SELECTORS.article.articleTitle);
    title = titleEl ? await titleEl.evaluate(el => el.innerText.trim()): '';
  } catch {}
  // 본문
  try {
    const el = await page.$(SELECTORS.article.body);
    body = el ? clean(await el.evaluate(e => e.innerText)): '';
  } catch {}
  // 카테고리
  try {
    const el = await page.$(SELECTORS.article.category);
    category = el ? clean(await el.evaluate(e => e.innerText)): '';
  } catch {}
  // 댓글
  try {
    const countEl = await page.$(SELECTORS.article.commentCount);
    const cntText = countEl ? await page.evaluate(el => el.innerText.replace(/,/g, '').trim(), countEl): '0';
    commentCount = parseInt(cntText, 10) || 0;
    if (commentCount > 5) {
      const comUrl = url.replace('/mnews/article/','/mnews/article/comment/');
      await page.goto(comUrl, {waitUntil:'networkidle2'});
      await loadAllComments(page);
    }
    comments = await extractAllComments(page);
  } catch (e) {}
  await page.close();
  return {journalist, createdDate, updatedDate, body, category, commentCount, comments, title}
}

// ===============================
// 검색 및 댓글 페이지 스크롤 & 더보기 클릭 & 텍스트 정제 & 날짜 변환
// ===============================
/**
 * 기존 @function scrollPageToBottom 
 * - 페이지 끝까지 스크롤
 */
// async function scrollPageToBottom(page) {
//   let lastHeight = await page.evaluate(() => document.body.scrollHeight);
// //   console.log("Initial height:", lastHeight);
//   while (true) {
//     await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
//     await new Promise(resolve => setTimeout(resolve, 300)); 
//     const newHeight = await page.evaluate(() => document.body.scrollHeight);
// //     console.log("After scroll - lastHeight:", lastHeight, "newHeight:", newHeight);      
//     if (newHeight === lastHeight) break;
//     lastHeight = newHeight;
//   }
// }
/**
 * updated ver.
 * @function scrollPageToBottom 
 * - 페이지 끝까지 스크롤
 */
async function scrollPageToBottom(page) {
  let lastHeight = await page.evaluate(() => document.body.scrollHeight);
  let retryCount = 0;
  const maxRetry = 3; // newHeight == lastHeight 이어도 n번 더 스크롤 내려 안정화
  while (true) {
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await new Promise(resolve => setTimeout(resolve, 300));
    const newHeight = await page.evaluate(() => document.body.scrollHeight);
    if (newHeight === lastHeight) {
      retryCount++;
      if (retryCount >= maxRetry) {
        break;
      }
    } else {
      retryCount = 0; // 높이가 바뀌면 retryCount 초기화
      lastHeight = newHeight;
    }
  }
}

/**
 * @function clean
 * 텍스트 내 줄바꿈, 이스케이프 문자 제거 및 공백 통일
 */
function clean(text = '') {
  return text
  .replace(/[\n\r]/g, ' ')           
  .replace(/\\/g, '')
  .replace(/\s+/g, ' ')
  .trim();
}

/**
 * @function loadAllComments
 * 댓글 페이지에서 '더보기' 버튼 클릭하여 모든 댓글 로드
 */
async function loadAllComments(page) {
  const MORE_BTN_SELECTOR = 'a.u_cbox_btn_more';
  while (true) {
    try {
      const moreBtn = await page.$(MORE_BTN_SELECTOR);
      if (!moreBtn) break;
      await page.waitForSelector(MORE_BTN_SELECTOR, {visible: true, timeout: 1000});
      await page.evaluate(selector => {
        const btn = document.querySelector(selector);
        if (btn) {
          btn.scrollIntoView({
            behavior: 'smooth', block: 'center'
          }); btn.click();
        }
      }, MORE_BTN_SELECTOR);
      await new Promise(r => setTimeout(r, 300));
    } catch {
      break;
    }
  }
}

/**
 * @function extractAllComments
 * 댓글 요소에서 닉네임, 작성일, 댓글내용, 추천/비추천 정보 추출
 */
async function extractAllComments(page) {
  return await page.$$eval(SELECTORS.article.commentItems, items => items.map(item => ({
    nick: item.querySelector('.u_cbox_nick')?.innerText.trim() || '',
    date: item.querySelector('.u_cbox_date')?.innerText.trim() || '',
    content: item.querySelector('.u_cbox_text_wrap > span.u_cbox_contents')?.innerText.trim() || '',
    recomm: item.querySelector('.u_cbox_btn_recomm > em')?.innerText.trim() || '0',
    unrecomm: item.querySelector('.u_cbox_btn_unrecomm > em')?.innerText.trim() || '0'
  })));
}

/**
 * @function relativeDate
 * 날짜 처리 (datetime 형식으로 + '주 전' 및 '일 전'은 년-월 까지만)
 */
// 특정 날짜 정보 datetime 형식에 맞게 변환
function relativeDate(text) {
  text = typeof text === 'string' ? text: String(text);
  const date = new Date();
  if (text.includes('일 전')) {
    const days = parseInt(text.replace(/[^0-9]/g, ''), 10);
    date.setDate(date.getDate() - days);
  } else if (text.includes('주 전')) {
    const weeks = parseInt(text.replace(/[^0-9]/g, ''), 10);
    date.setDate(date.getDate() - weeks * 7);
  } else if (text.includes('시간 전')) {
    const hours = parseInt(text.replace(/[^0-9]/g, ''), 10);
    date.setHours(date.getHours() - hours);
  } else if (text.includes('분 전')) {
    const mins = parseInt(text.replace(/[^0-9]/g, ''), 10);
    date.setMinutes(date.getMinutes() - mins);
  }
  return date;
}

// 날짜, 시간 정보 형식 통일
function formatYMD(date) {
  const Y = date.getFullYear();
  const M = String(date.getMonth() + 1).padStart(2, '0');
  const D = String(date.getDate()).padStart(2, '0');
  return `${Y}-${M}-${D}`;
}

function formatYMDhms(date) {
  const hh = String(date.getHours()).padStart(2, '0');
  const mm = String(date.getMinutes()).padStart(2, '0');
  const ss = String(date.getSeconds()).padStart(2, '0');
  return `${formatYMD(date)} ${hh}:${mm}:${ss}`;
}

function normalizeDate(raw) {
  if (/^\d{4}\.\d{2}\.\d{2}/.test(raw)) {
    const [Y, M, D] = raw.split('.');return `${Y}-${M}-${D}`;
  }
  if (/^\d{4}-\d{2}-\d{2}/.test(raw)) {return raw;}
  const dt = relativeDate(raw);
  return formatYMD(dt);
}

// 댓글용
function normalizeCommentDate(raw) {
  const dt = raw.includes('전') ? relativeDate(raw): new Date(raw);
  const hh = String(dt.getHours()).padStart(2, '0');
  const mm = String(dt.getMinutes()).padStart(2, '0');
  return `${formatYMD(dt)} ${hh}:${mm}`;
}