# 🚀 배포 가이드

## Vercel 배포 완료 후 해야 할 일

### 1. 배포 URL 확인
Vercel 대시보드에서 배포된 URL을 확인하세요:
- 예: `https://etf-portfolio-analyzer-xxx.vercel.app`

### 2. README.md 업데이트
배포 URL을 받으면 다음 파일의 링크들을 업데이트하세요:

```markdown
# 🚀 라이브 데모

**[종합 비교 대시보드](https://your-actual-vercel-url.vercel.app/TQQQ_SOXL_VXX_Comparison_Dashboard.html)** - 모든 분석 결과를 한눈에 비교

### 📊 개별 분석 대시보드
- [TQQQ vs VIX 분석](https://your-actual-vercel-url.vercel.app/TQQQ_VIX_Simple_Dashboard.html)
- [TQQQ vs VXX 포트폴리오](https://your-actual-vercel-url.vercel.app/TQQQ_VXX_Portfolio_Dashboard.html)
- [SOXL vs VXX 포트폴리오](https://your-actual-vercel-url.vercel.app/SOXL_VXX_Portfolio_Dashboard.html)
```

### 3. package.json 업데이트
```json
{
  "homepage": "https://your-actual-vercel-url.vercel.app"
}
```

### 4. 커스텀 도메인 설정 (선택사항)
Vercel 대시보드에서 Settings → Domains에서 커스텀 도메인을 추가할 수 있습니다.

## 🔄 자동 배포

GitHub에 코드를 푸시하면 Vercel이 자동으로 재배포합니다:

```bash
git add .
git commit -m "Update analysis"
git push origin main
```

## 📱 모바일 최적화

현재 대시보드는 반응형으로 설계되어 모바일에서도 잘 작동합니다.

## 🔧 추가 설정

### 환경 변수 (필요시)
- `VERCEL_ENV`: 배포 환경
- API 키 등이 필요한 경우 Vercel 대시보드에서 설정

### 빌드 설정
현재는 정적 사이트이므로 추가 빌드 설정이 필요하지 않습니다.

## 🎯 성능 최적화

- 모든 차트는 Plotly CDN을 사용하여 빠른 로딩
- 이미지 최적화는 Vercel이 자동으로 처리
- 캐싱 설정은 vercel.json에 포함됨

## 📊 분석 도구

배포된 사이트에서 다음 기능들을 사용할 수 있습니다:

1. **종합 비교 대시보드**: TQQQ vs SOXL vs VXX 전체 비교
2. **개별 분석**: 각 ETF 조합별 상세 분석
3. **인터랙티브 차트**: 실시간 데이터 조작
4. **포트폴리오 최적화**: 몬테카를로 시뮬레이션 결과
5. **투자자 유형별 추천**: 리스크 성향에 따른 포트폴리오 제안

## 🎉 완료!

이제 전 세계 어디서든 접근 가능한 ETF 포트폴리오 분석 도구가 완성되었습니다!
