const puppeteer = require('puppeteer');
const fs = require('fs');
const { parse } = require('json2csv');

// 무한 스크롤 함수
async function autoScroll(page) {
    await page.evaluate(async () => {
        const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));
        let lastHeight = document.body.scrollHeight;

        while (true) {
            window.scrollTo(0, document.body.scrollHeight);
            await delay(1000); // 페이지 로딩 대기 시간

            let newHeight = document.body.scrollHeight;
            if (newHeight === lastHeight) break;
            lastHeight = newHeight;
        }
    });
}

// 스포일러 포함 리뷰 처리 함수
async function toggleSpoiler(page, selector) {
    try {
        // 스크롤 + 스포일러 버튼
        await page.evaluate(async (selector) => {
            const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));
            let lastHeight = document.body.scrollHeight;
            
            while (true) {
                // 스포일러 버튼이 있는지 확인
                const spoilerButtons = Array.from(document.querySelectorAll(selector));
                if (spoilerButtons.length > 0) {
                    // 스포일러 버튼 클릭
                    for (const button of spoilerButtons) {
                        button.click();
                        await delay(1000); // 클릭 후 페이지 로딩 대기
                    }
                }

                window.scrollTo(0, document.body.scrollHeight);
                await delay(1000); // 페이지 로딩 대기 시간

                let newHeight = document.body.scrollHeight;
                if (newHeight === lastHeight) break;
                lastHeight = newHeight;
            }
        }, selector);
    } catch (error) {
        console.error(`error in spoiler button: ${error.message}`);
    }
}

// 리뷰 크롤링 함수
async function crawlReviews(page) {
    const reviews = await page.evaluate(() => {
        const reviewElements = document.querySelectorAll('.A5k28.GFvWN'); // 리뷰 부분 선택
        const data = [];

        reviewElements.forEach((review) => {
            const reviewer = review.querySelector('.YIGt3.eWROw')?.textContent.trim() || '작성자 없음';
            const rating = review.querySelector('.aytsx.Zcvzh span')?.textContent.trim() || '평점 없음';
            const likes = review.querySelector('.TmaB2.Zcvzh em')?.textContent.trim() || '0';
            const comments = review.querySelector('.TmaB2.Zcvzh em:nth-of-type(2)')?.textContent.trim() || '0';
            const reviewText = review.querySelector('.NrA8Q.NAkgh.qz3Mr .eqSew')?.textContent.trim() || '리뷰 없음';

            data.push({
                작성자: reviewer,
                평점: rating,
                좋아요: likes,
                댓글수: comments,
                리뷰: reviewText,
            });
        });

        return data;
    });

    return reviews;
}

// 메인 실행 함수
// async function main() {
//     const browser = await puppeteer.launch({ headless: false });
async function main() {
    const browser = await puppeteer.launch({ 
        headless: false,
        protocolTimeout: 600000 // 프로토콜 타임아웃(10분) 설정 (영화 리뷰가 많을 경우 대비)
    });
    const page = await browser.newPage();
    
    const url = 'https://pedia.watcha.com/ko-KR/contents/mW48wPW/comments'; // 영화 URL
    await page.goto(url, { waitUntil: 'networkidle2' });

    // 무한 스크롤 처리
    await autoScroll(page);

    // 스포일러가 있는 경우 버튼 클릭 (보기 버튼)
    await toggleSpoiler(page, '.DuZ3S.ugaiG .HwyMg'); // '보기' 버튼 셀렉터

    // 페이지 제목에서 영화 제목 추출
    const title = await page.evaluate(() => {
        return document.title.split(':')[0].trim();
    });

    // 모든 리뷰 크롤링
    const reviews = await crawlReviews(page);

    // 리뷰 JSON 저장
    const jsonFileName = `${title}.json`;
    fs.writeFileSync(jsonFileName, JSON.stringify(reviews, null, 2), { encoding: 'utf8' });
    console.log(`The reviews have been saved to ${jsonFileName}.`);

    // CSV 파일 저장
    if (reviews.length > 0) {
        const csvFileName = `${title}.csv`;
        const csv = parse(reviews);

        // UTF-8 BOM 추가
        const utf8BOM = '\uFEFF';
        fs.writeFileSync(csvFileName, utf8BOM + csv, 'utf-8');
        console.log(`The reviews have been saved to ${csvFileName}.`);
    } else {
        console.log(`No reviews to save for ${title}.`);
    }

    await browser.close();
}

// 실행
main().catch(console.error);