## 네이버 영화 및 왓챠피디아 영화 리뷰 크롤러
### 1. Node.js 설치
- 아래 링크를 통한 최신 버전 설치
https://nodejs.org/
- 설치 후 cmd(명령프롬프트)에서 버전 확인
  ```bash
  node -v
  npm -v

### 2. Puppeteer 설치
- 프로젝트 디렉토리 생성(예시)
mkdir puppeteer-crawler  
cd puppeteer-crawler
- npm 초기화(package.json 파일 생성 후 기본적인 설정 입력)
npm init -y
- puppeteer 설치
npm install puppeteer

### 3. 실행 코드
- 프로젝트 디렉토리로 이동(cmd)
cd C:\Users\사용자\puppeteer-crawler
- Node.js 실행
예) node naver_crawler.js  
예) node watcha_crawler.js
