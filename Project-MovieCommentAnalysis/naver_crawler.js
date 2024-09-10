const puppeteer = require('puppeteer');
const fs = require('fs');
const { Parser } = require('json2csv');
const xlsx = require('xlsx');

// 전체 페이지를 스크롤하는 함수(사용 보류)
async function scrollPageToBottom(page) {
    let lastHeight = await page.evaluate('document.body.scrollHeight');

    while (true) {
        await page.evaluate('window.scrollBy(0, window.innerHeight * 3)'); // 전체 페이지의 n배 만큼 스크롤
        await new Promise(resolve => setTimeout(resolve, 500)); // 0.5초 대기
        let newHeight = await page.evaluate('document.body.scrollHeight');
        if (newHeight === lastHeight) {
            break;
        }
        lastHeight = newHeight;
    }
}

// 스크롤 가능한 요소 내부에서 스크롤하는 함수(펼쳐보기 후 스크롤)
async function scrollSectionToBottom(page, selector) {
    const scrollableElement = await page.waitForSelector(selector);
    let lastHeight = await page.evaluate(el => el.scrollHeight, scrollableElement);

    while (true) {
        await page.evaluate(el => el.scrollBy(0, el.clientHeight * 50), scrollableElement);
        await new Promise(resolve => setTimeout(resolve, 500)); // 0.5초 대기
        let newHeight = await page.evaluate(el => el.scrollHeight, scrollableElement);
        if (newHeight === lastHeight) {
            break;
        }
        lastHeight = newHeight;
    }
}

// 요소가 보일 때까지 대기한 후 클릭하는 함수
async function waitForSelectorAndClick(page, selector) {
    try {
        await page.waitForSelector(selector, { visible: true, timeout: 2000 });
        const element = await page.$(selector);
        if (element) {
            await page.evaluate(el => el.scrollIntoView({ behavior: 'smooth', block: 'center' }), element);
            await new Promise(resolve => setTimeout(resolve, 500)); // 0.5초 대기
            await element.click();
            await new Promise(resolve => setTimeout(resolve, 1000)); // 1초 대기
        } else {
            console.error(`Element not found: ${selector}`);
        }
    } catch (error) {
        console.error(`Error in waitForSelectorAndClick: ${error.message}`);
    }
}

// 스포일러 토글 버튼 클릭 함수
async function toggleSpoiler(page, selectorList) {
    for (const selector of selectorList) {
        try {
            const element = await page.$(selector);
            if (element) {
                await waitForSelectorAndClick(page, selector);
                return;
            }
        } catch (error) {
            console.error(`Error in toggleSpoiler: ${error.message}`);
        }
    }
    console.error('No valid spoiler toggle button found.');
}

// 펼쳐보기 버튼 클릭 함수
async function clickMoreButton(page, selectorList) {
    for (const selector of selectorList) {
        try {
            const element = await page.$(selector);
            if (element) {
                await waitForSelectorAndClick(page, selector);
                return;
            }
        } catch (error) {
            console.error(`Error in clickMoreButton: ${error.message}`);
        }
    }
    console.error('No valid "More" button found.');
}

// 리뷰 크롤링 함수
async function crawlReviews(movieTitle, tabSelector, spoilerSelectors, moreButtonSelectors, reviewListSelector) {
    const browser = await puppeteer.launch({ headless: false });
    const page = await browser.newPage();

    const query = encodeURIComponent(`영화 ${movieTitle} 관람평`);
    const url = `https://m.search.naver.com/search.naver?sm=mtp_hty.top&where=m&query=${query}#movie-rating`;

    await page.goto(url, { waitUntil: 'networkidle2' });

    await waitForSelectorAndClick(page, tabSelector);
    await toggleSpoiler(page, spoilerSelectors); // 스포일러 토글 버튼 클릭
    await clickMoreButton(page, moreButtonSelectors); // 펼쳐보기 버튼 클릭
    await scrollSectionToBottom(page, 'div.panel_content._slide_board_scroller');

    const reviews = await page.evaluate((movieTitle, reviewListSelector) => {
        const reviewItems = Array.from(document.querySelectorAll(reviewListSelector));
        if (reviewItems.length === 0) {
            console.warn('No reviews found for:', movieTitle);
            return [];
        }
        return reviewItems.map(item => {
            const ratingText = item.querySelector('div.lego_movie_pure_star .area_text_box')?.textContent.trim() || '정보 없음';
            const ratingMatch = ratingText.match(/(\d+)\s*$/);
            const rating = ratingMatch ? ratingMatch[1] : '정보 없음';

            const reviewText = item.querySelector('div.area_review_content span.desc')?.textContent.trim() || '정보 없음';
            const writerId = item.querySelector('dl.cm_upload_info dd.this_text_stress')?.textContent.trim() || '정보 없음';
            const dateStr = item.querySelector('dl.cm_upload_info dd.this_text_normal')?.textContent.trim() || '정보 없음';
            const upvoteCount = item.querySelector('button._btn_upvote ._count_num')?.textContent.trim() || '정보 없음';
            const downvoteCount = item.querySelector('button._btn_downvote ._count_num')?.textContent.trim() || '정보 없음';

            return {
                "영화제목": movieTitle,
                "작성자": writerId,
                "작성일": dateStr,
                "평점": rating,
                "리뷰": reviewText,
                "공감수": upvoteCount,
                "비공감수": downvoteCount
            };
        });
    }, movieTitle, reviewListSelector);

    await browser.close();
    return reviews;
}

// 영화 목록 읽어오는 함수
function readMovieListFromExcel(filePath) {
    const workbook = xlsx.readFile(filePath);
    const sheetName = workbook.SheetNames[0];
    const worksheet = workbook.Sheets[sheetName];
    const movieList = xlsx.utils.sheet_to_json(worksheet, { header: 1 });
    // 첫 번째 행(헤더)제외하고 영화 목록만 반환
    return movieList.slice(1).map(row => ({
        year: row[1],
        title: row[2],
        summary: row[3],
    }));
}

// 리뷰 데이터 저장 함수
function saveReviewsToFile(movieTitle, reviews, reviewType) {
    const baseFileName = sanitizeFileName(movieTitle) + `(${reviewType})`;

    // JSON 파일 저장
    const jsonFileName = `${baseFileName}.json`;
    fs.writeFileSync(jsonFileName, JSON.stringify(reviews, null, 2), 'utf-8');
    console.log(`The reviews have been saved to '${jsonFileName}'.`);

    // CSV 파일 저장
    if (reviews.length > 0) {
        const csvFileName = `${baseFileName}.csv`;
        const json2csvParser = new Parser();
        const csv = json2csvParser.parse(reviews);

        const utf8BOM = '\uFEFF';

        fs.writeFileSync(csvFileName, utf8BOM + csv, 'utf-8');
        console.log(`The reviews have been saved to '${csvFileName}'`);
    } else {
        console.log(`No reviews to save for ${movieTitle} (${reviewType}).`);
    }
}

// 파일 이름에서 불필요한 문자 제거 함수
function sanitizeFileName(fileName) {
    return fileName.replace(/[\/\\:*\?"<>\|]/g, '-');
}

// 메인 실행 함수
async function main() {
    const movieList = readMovieListFromExcel('movies.xlsx'); // 엑셀 파일 경로를 입력

    for (const movie of movieList) {
        const movieTitle = movie.title;
        
// 실관람객, 네티즌 그리고 네티즌만 있는 영화 모두 요소가 다름
        // 실관람객 탭 선택
        const realAudienceReviews = await crawlReviews(movieTitle, 
            'div.cm_content_wrap._cm_content_wrap_rating > div > div > div.cm_tap_area._tab_wrap > div > div > ul > li:nth-child(1) > a > span', // 실관람객 있는 부분
            [
                'div.cm_content_wrap._cm_content_wrap_rating > div > div > div:nth-child(4) > div.cm_tap_area.type_inline._sorting_wrap > div.lego_toggle_sort._spoiler_switch > button' //실관람객 스포일러 버튼
            ],
            [
                'div.cm_content_wrap._cm_content_wrap_rating > div > div > div:nth-child(4) > div.more_wrap > a' //실관람객 펼쳐보기
            ],
            'div.panel_content._slide_board_scroller > div > div.lego_review_list._audience_review_list > ul > li.area_card._item'
        ); //실관람객 리뷰 아이템
        saveReviewsToFile(movieTitle, realAudienceReviews, "실관람객");

        // 네티즌 탭 선택
        const netizenReviews = await crawlReviews(movieTitle, 
            'div.cm_content_wrap._cm_content_wrap_rating > div > div > div.cm_tap_area._tab_wrap > div > div > ul > li:nth-child(2) > a > span', //네티즌 부분
            'div.cm_content_wrap._cm_content_wrap_rating > div > div > div.cm_tap_area._tab_wrap > div > div > ul > li:nth-child(1) > a > span'// 네티즌 (only) 부분
            [
                'div.cm_content_wrap._cm_content_wrap_rating > div > div > div:nth-child(5) > div.cm_tap_area.type_inline._sorting_wrap > div.lego_toggle_sort._spoiler_switch > button', // 네티즌 스포일러 버튼
                'div.cm_content_wrap._cm_content_wrap_rating > div > div > div:nth-child(3) > div.cm_tap_area.type_inline._sorting_wrap > div.lego_toggle_sort._spoiler_switch > button' //보완 부분이나 제거하여도 무방(실관람객 없는 영화의 네티즌)
            ],
            [
                'div.cm_content_wrap._cm_content_wrap_rating > div > div > div:nth-child(5) > div.more_wrap > a', //네티즌 펼쳐보기
                'div.cm_content_wrap._cm_content_wrap_rating > div > div > div:nth-child(3) > div.more_wrap > a'
            ],
            'div.panel_content._slide_board_scroller > div > div.lego_review_list._netizen_review_list > ul > li.area_card._item'
        ); // 네티즌(only) 리뷰 아이템
        saveReviewsToFile(movieTitle, netizenReviews, "네티즌");
        const onlyNetizenReviews = await crawlReviews(movieTitle,  'div.cm_content_wrap._cm_content_wrap_rating > div > div > div.cm_tap_area._tab_wrap > div > div > ul > li:nth-child(1) > a > span', //실관람객이 존재하지 않는(네티즌(only) 영화의 네티즌
                    [
                        'div.cm_content_wrap._cm_content_wrap_rating > div > div > div:nth-child(3) > div.cm_tap_area.type_inline._sorting_wrap > div.lego_toggle_sort._spoiler_switch > button' // 네티즌(only) 스포일러
                    ],
                    [
                        'div.cm_content_wrap._cm_content_wrap_rating > div > div > div:nth-child(3) > div.more_wrap > a'
                    ], //네티즌(only) 펼쳐보기
                    'div.panel_content._slide_board_scroller > div > div.lego_review_list._netizen_review_list > ul > li.area_card._item' // 네티즌(only) 리뷰 아이템
                );
        saveReviewsToFile(movieTitle, onlyNetizenReviews, "네티즌");        
    }
}

// 메인 실행
main().catch(console.error);

//향후 업데이트 시도
//1. 실관람객 있는 것/없는 것에 따른 전체 요인(평론가 포함) 크롤링.
//2. 301행 고정 관련(마지막 리뷰까지 비교하여 보았을 때는 동일) *추가 확인: 직접 스크롤로 작동하는 코드 생성 해서 비교.