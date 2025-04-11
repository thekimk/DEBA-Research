## 🎬네이버 영화 및 왓챠피디아 영화 리뷰 크롤러
### 1. Node.js 설치
- 아래 링크를 통한 최신 버전 설치  
https://nodejs.org/  
  
- 설치 후 cmd(명령프롬프트)에서 버전 확인
  ```bash
  node -v
  npm -v

### 2. Puppeteer 설치
- 프로젝트 디렉토리 생성(예시)
  ```bash
  mkdir puppeteer-crawler  
  cd puppeteer-crawler
- package.json 파일 생성(프로젝트 정보 및 라이브러리 기록파일)
  npm init -y
- puppeteer 설치
  ```bash
  npm install puppeteer
- 모듈 설치
  ```bash
  npm install json2csv cli-progress

### 3. 실행 방법
- 프로젝트 디렉토리로 이동(cmd)
  ```bash
  cd C:\Users\사용자\puppeteer-crawler
- Node.js를 사용하여 크롤러 스크립트 실행
  ```bash
  node 파일명.js

### 속도 최적화 팁
- headless 모드 활성화(기본세팅: false)
  ```bash
  headless: true
