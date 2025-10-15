# ETF 포트폴리오 최적화 분석 도구

TQQQ, SOXL, VXX ETF의 상관관계 분석 및 포트폴리오 최적화 도구입니다.

## 🚀 라이브 데모

**[종합 비교 대시보드](https://your-vercel-app.vercel.app/TQQQ_SOXL_VXX_Comparison_Dashboard.html)** - 모든 분석 결과를 한눈에 비교

### 📊 개별 분석 대시보드
- [TQQQ vs VIX 분석](https://your-vercel-app.vercel.app/TQQQ_VIX_Simple_Dashboard.html)
- [TQQQ vs VXX 포트폴리오](https://your-vercel-app.vercel.app/TQQQ_VXX_Portfolio_Dashboard.html)
- [SOXL vs VXX 포트폴리오](https://your-vercel-app.vercel.app/SOXL_VXX_Portfolio_Dashboard.html)

## 📈 주요 기능

### 🔍 상관관계 분석
- **TQQQ-VIX**: -0.635 (중간 음의 상관)
- **TQQQ-VXX**: -0.723 (강한 음의 상관)
- **SOXL-VXX**: -0.681 (강한 음의 상관)

### 🎯 포트폴리오 최적화
- 몬테카를로 시뮬레이션 기반 최적화
- 최대 샤프 비율 포트폴리오
- 최소 변동성 포트폴리오
- 헤지 효과 분석

### 📊 시각화
- 인터랙티브 Plotly 차트
- 실시간 데이터 분석
- 다양한 리스크 지표 시각화

## 🏆 핵심 결과

### 최적 포트폴리오 추천

| ETF 조합 | 최적 비중 | 예상 수익률 | 변동성 | 샤프 비율 |
|----------|-----------|-------------|--------|-----------|
| **TQQQ-VXX** | TQQQ 74%, VXX 26% | 32.1% | 40.4% | 0.720 |
| **SOXL-VXX** | SOXL 73%, VXX 27% | 43.2% | 61.7% | 0.651 |

### 투자자 유형별 추천

#### 🛡️ 안정성 우선 투자자
- **TQQQ 74% + VXX 26%**
- 더 낮은 변동성과 높은 샤프 비율

#### 🚀 수익성 우선 투자자  
- **SOXL 73% + VXX 27%**
- 더 높은 수익률 (반도체 섹터 집중)

## 📁 파일 구조

```
ETF_correlation/
├── README.md                                    # 프로젝트 설명
├── requirements.txt                             # Python 의존성
├── TQQQ_SOXL_VXX_Comparison_Dashboard.html     # 종합 비교 대시보드
├── tqqq_vix_correlation_analyzer.py            # TQQQ-VIX 분석기
├── tqqq_vxx_analyzer.py                        # TQQQ-VXX 분석기
├── soxl_vxx_analyzer.py                        # SOXL-VXX 분석기
├── portfolio_optimizer.py                      # 포트폴리오 최적화
├── create_interactive_dashboard.py             # 대시보드 생성기
└── *.html                                      # 개별 분석 대시보드
```

## 🛠️ 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 분석 실행
```bash
# TQQQ vs VIX 분석
python tqqq_vix_correlation_analyzer.py

# TQQQ vs VXX 분석
python tqqq_vxx_analyzer.py

# SOXL vs VXX 분석
python soxl_vxx_analyzer.py

# 포트폴리오 최적화
python portfolio_optimizer.py
```

## 📊 데이터 소스

- **TQQQ**: Direxion Daily Technology Bull 3X Shares
- **SOXL**: Direxion Daily Semiconductor Bull 3X Shares  
- **VIX**: CBOE Volatility Index
- **VXX**: iPath Series B S&P 500 VIX Short-Term Futures ETN

데이터는 Yahoo Finance API를 통해 실시간으로 수집됩니다.

## ⚠️ 면책 조항

이 분석은 과거 데이터를 기반으로 하며, 미래 수익을 보장하지 않습니다. 
투자 결정 전 반드시 개인의 리스크 성향과 투자 목표를 고려하시기 바랍니다.

## 🔧 기술 스택

- **Python**: 데이터 분석 및 시뮬레이션
- **yfinance**: 금융 데이터 수집
- **pandas**: 데이터 처리
- **numpy**: 수치 계산
- **plotly**: 인터랙티브 시각화
- **scipy**: 통계 분석
- **HTML/CSS/JavaScript**: 웹 대시보드

## 📈 분석 기간

- **TQQQ-VIX**: 2010년 2월 9일 ~ 2025년 10월 14일 (3,943 거래일)
- **TQQQ-VXX**: 2018년 1월 25일 ~ 2025년 10월 14일 (1,941 거래일)
- **SOXL-VXX**: 2018년 1월 25일 ~ 2025년 10월 14일 (1,941 거래일)

## 🎯 주요 발견사항

1. **VXX 헤지 효과**: 모든 레버리지 ETF가 VXX와 음의 상관관계를 보여 효과적인 헤지 도구로 활용 가능
2. **최적 헤지 비율**: 대부분 26-27%의 VXX 헤지가 최적의 리스크-수익률 균형 제공
3. **섹터 차이**: SOXL은 반도체 집중으로 높은 수익률과 변동성, TQQQ는 다양한 테크로 안정성
4. **시장 상황**: 2025년 들어 상관관계가 더욱 강화되는 경향

## 📞 문의

프로젝트에 대한 문의사항이나 개선 제안이 있으시면 이슈를 생성해 주세요.

---

**⚡ Vercel로 배포된 고성능 웹 애플리케이션**