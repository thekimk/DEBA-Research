## 🎬네이버 영화 및 왓챠피디아 영화 리뷰 크롤러
### 1. Node.js 설치
- 아래 링크를 통한 최신 버전 설치하되 추가 체크할 건 없이 change -> next만 쭉 진행하면 됨  
https://nodejs.org/  

### 2. Puppeteer 설치(cmd)
- cmd(명령프롬프트) 진입 및 설치 확인을 위한 버전 확인
  ```bash
  node -v
  npm -v
- 프로젝트 디렉토리 생성 및 이동
  ```bash
  mkdir puppeteer-crawler  
  cd C:\Users\사용자\puppeteer-crawler
- package.json 파일 생성(프로젝트 정보 및 라이브러리 기록파일)
  ```bash
  npm init -y
- puppeteer 설치
  ```bash
  npm install puppeteer
- 모듈 설치
  ```bash
  npm install json2csv
  npm install cli-progress
  npm install p-limit@2

### 3. 파일 준비(파일탐색기)
- `naver-news.js & config.json` 2개의 파일을 `C:\Users\사용자\puppeteer-crawler`에 붙여넣기

### 4. 실행 방법(cmd)
- Node.js를 사용하여 크롤러 스크립트 실행
  ```bash
  node naver-news.js

### 속도 최적화 팁
- headless 모드 활성화(기본세팅: false)
  ```bash
  headless: true
