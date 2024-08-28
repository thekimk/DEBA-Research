## :office: 연구실 소개
- 김경원 교수 소개 (<a href="https://sites.google.com/view/thekimk" target="_blank"><img src="https://img.shields.io/badge/Homepage-4285F4?style=flat-square&logo=Google&logoColor=white"/></a> <a href="https://scholar.google.com/citations?hl=ko&user=nHPe-4UAAAAJ&view_op=list_works&sortby=pubdate" target="_blank"><img src="https://img.shields.io/badge/Google Scholar-4285F4?style=flat-square&logo=Google Scholar&logoColor=white"/></a> <a href="https://www.youtube.com/channel/UCEYxJNI5dhnn_CdC9BEWTuA" target="_blank"><img src="https://img.shields.io/badge/YouTube-FF0000?style=flat-square&logo=YouTube&logoColor=white"/></a> <a href="https://github.com/thekimk" target="_blank"><img src="https://img.shields.io/badge/Github-181717?style=flat-square&logo=Github&logoColor=white"/></a>)
- DEBA 연구실은 디지털경제의 표준이 되어가고 있는 `빅데이터와 머신/딥러닝이란 인공지능 이론을 활용`하여,     
`경제산업 분야의 문제`를 `데이터`를 기반으로 `정량적으로 분석하고 의사결정`함으로써 `전략적으로 문제를 해결`하기 위한 방법을 연구합니다.

---

### KTX 승차수요 단기예측 분석
- **분석내용:** `예측, 성능, 설명력`
- **데이터기간:** `2015년1월~2024년3월`
> - **Training:** `2015년1월~2023년3월`
> - **Validation:** `2023년4월~2024년3월`
> - **Test:** `2024년4월~2025년12월`
- **사용데이터:** 전체 6개 파일 중 `4개의 일별 데이터를 월별로 집계`한 것 (나머지는 이미 집계된 월별데이터라 미사용)
- **파생변수:** `시간정보/주말갯수/주중갯수/공유일/경제환경/이벤트 등` 활용

---

### 예측대상: 
- **KTX 5개노선:** `경부선, 경전선, 동해선, 전라선, 호남선`
- **대상 3종:** `월별 전체 > 월별 주말 > 월별 주중`
- **주중 및 주말기준:** (주중)`월화수목`, (주말)`금토일`
- **케이스 15종:** 월별 KTX 전체/주말/주중 15개 수요
 
| **시간빈도** | **노선** | **대상** | **총 15개 월별 KTX 수송수요** |
|:---:|:---:|:---:|:---:|
| 월별 | 경부선 | 전체 | 월별 경부선 전체 수요 |
|  |  | 주말 | 월별 경부선 주말 수요 |
|  |  | 주중 | 월별 경부선 주중 수요 |
|  | 경전선 | 전체 | 월별 경전선 전체 수요 |
|  |  | 주말 | 월별 경전선 주말 수요 |
|  |  | 주중 | 월별 경전선 주중 수요 |
|  | 동해선 | 전체 | 월별 동해선 전체 수요 |
|  |  | 주말 | 월별 동해선 주말 수요 |
|  |  | 주중 | 월별 동해선 주중 수요 |
|  | 전라선 | 전체 | 월별 전라선 전체 수요 |
|  |  | 주말 | 월별 전라선 주말 수요 |
|  |  | 주중 | 월별 전라선 주중 수요 |
|  | 호남선 | 전체 | 월별 호남선 전체 수요 |
|  |  | 주말 | 월별 호남선 주말 수요 |
|  |  | 주중 | 월별 호남선 주중 수요 |

