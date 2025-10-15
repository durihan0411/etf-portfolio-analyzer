import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class TQQQVIXCorrelationAnalyzer:
    def __init__(self):
        self.tqqq_ticker = "TQQQ"
        self.vix_ticker = "^VIX"
        self.tqqq_data = None
        self.vix_data = None
        self.combined_data = None
        self.analysis_results = None
        
    def fetch_data(self, start_date=None, end_date=None):
        """TQQQ와 VIX 데이터를 가져옵니다."""
        print("TQQQ와 VIX 데이터를 가져오는 중...")
        
        # TQQQ는 2010년 2월 9일에 시작
        if start_date is None:
            start_date = "2010-02-09"
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        try:
            # TQQQ 데이터 수집
            print("TQQQ 데이터 수집 중...")
            tqqq_raw = yf.download(self.tqqq_ticker, start=start_date, end=end_date, progress=False)
            if tqqq_raw.empty:
                raise ValueError("TQQQ 데이터를 가져올 수 없습니다.")
            
            # VIX 데이터 수집
            print("VIX 데이터 수집 중...")
            vix_raw = yf.download(self.vix_ticker, start=start_date, end=end_date, progress=False)
            if vix_raw.empty:
                raise ValueError("VIX 데이터를 가져올 수 없습니다.")
            
            # MultiIndex 컬럼을 단일 레벨로 변환
            if isinstance(tqqq_raw.columns, pd.MultiIndex):
                tqqq_raw.columns = tqqq_raw.columns.droplevel(0)
            if isinstance(vix_raw.columns, pd.MultiIndex):
                vix_raw.columns = vix_raw.columns.droplevel(0)
            
            # 컬럼명 정리
            if len(tqqq_raw.columns) == 6:
                tqqq_raw.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            elif len(tqqq_raw.columns) == 5:
                tqqq_raw.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
            if len(vix_raw.columns) == 6:
                vix_raw.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            elif len(vix_raw.columns) == 5:
                vix_raw.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            self.tqqq_data = tqqq_raw
            self.vix_data = vix_raw
            
            # 데이터 결합
            self.combined_data = pd.DataFrame({
                'TQQQ_Close': self.tqqq_data['Close'],
                'VIX_Close': self.vix_data['Close'],
                'TQQQ_Volume': self.tqqq_data['Volume'],
                'VIX_Volume': self.vix_data['Volume']
            }).dropna()
            
            print(f"TQQQ 데이터 수집 완료: {len(self.tqqq_data)}개 거래일")
            print(f"VIX 데이터 수집 완료: {len(self.vix_data)}개 거래일")
            print(f"결합된 데이터: {len(self.combined_data)}개 거래일")
            print(f"기간: {self.combined_data.index[0].strftime('%Y-%m-%d')} ~ {self.combined_data.index[-1].strftime('%Y-%m-%d')}")
            return True
            
        except Exception as e:
            print(f"데이터 수집 중 오류 발생: {e}")
            return False
    
    def calculate_returns_and_volatility(self):
        """수익률과 변동성을 계산합니다."""
        if self.combined_data is None:
            print("먼저 데이터를 가져와주세요.")
            return
        
        # 일간 수익률 계산
        self.combined_data['TQQQ_Return'] = self.combined_data['TQQQ_Close'].pct_change()
        self.combined_data['VIX_Return'] = self.combined_data['VIX_Close'].pct_change()
        
        # 로그 수익률 계산
        self.combined_data['TQQQ_Log_Return'] = np.log(self.combined_data['TQQQ_Close'] / self.combined_data['TQQQ_Close'].shift(1))
        self.combined_data['VIX_Log_Return'] = np.log(self.combined_data['VIX_Close'] / self.combined_data['VIX_Close'].shift(1))
        
        # 이동 평균 변동성 (20일, 60일)
        self.combined_data['TQQQ_Volatility_20d'] = self.combined_data['TQQQ_Return'].rolling(window=20).std() * np.sqrt(252)
        self.combined_data['VIX_Volatility_20d'] = self.combined_data['VIX_Return'].rolling(window=20).std() * np.sqrt(252)
        
        self.combined_data['TQQQ_Volatility_60d'] = self.combined_data['TQQQ_Return'].rolling(window=60).std() * np.sqrt(252)
        self.combined_data['VIX_Volatility_60d'] = self.combined_data['VIX_Return'].rolling(window=60).std() * np.sqrt(252)
        
        # 이동 상관관계 계산 (20일, 60일)
        self.combined_data['Correlation_20d'] = self.combined_data['TQQQ_Return'].rolling(window=20).corr(self.combined_data['VIX_Return'])
        self.combined_data['Correlation_60d'] = self.combined_data['TQQQ_Return'].rolling(window=60).corr(self.combined_data['VIX_Return'])
        
        print("수익률 및 변동성 계산 완료")
    
    def analyze_correlation(self):
        """상관관계를 분석합니다."""
        if self.combined_data is None:
            print("먼저 데이터를 가져와주세요.")
            return
        
        # 전체 기간 상관관계
        overall_corr_price = self.combined_data['TQQQ_Close'].corr(self.combined_data['VIX_Close'])
        overall_corr_return = self.combined_data['TQQQ_Return'].corr(self.combined_data['VIX_Return'])
        
        # 연도별 상관관계
        yearly_corr = []
        for year in self.combined_data.index.year.unique():
            year_data = self.combined_data[self.combined_data.index.year == year]
            if len(year_data) > 20:  # 최소 20개 거래일 이상인 경우만
                price_corr = year_data['TQQQ_Close'].corr(year_data['VIX_Close'])
                return_corr = year_data['TQQQ_Return'].corr(year_data['VIX_Return'])
                yearly_corr.append({
                    'Year': year,
                    'Price_Correlation': price_corr,
                    'Return_Correlation': return_corr,
                    'Trading_Days': len(year_data)
                })
        
        yearly_corr_df = pd.DataFrame(yearly_corr)
        
        # 분기별 상관관계
        quarterly_corr = []
        for year in self.combined_data.index.year.unique():
            for quarter in [1, 2, 3, 4]:
                start_date = f"{year}-{quarter*3-2:02d}-01"
                # 각 분기별 마지막 날짜를 올바르게 설정
                if quarter == 1:
                    end_date = f"{year}-03-31"
                elif quarter == 2:
                    end_date = f"{year}-06-30"
                elif quarter == 3:
                    end_date = f"{year}-09-30"
                else:  # quarter == 4
                    end_date = f"{year}-12-31"
                
                quarter_data = self.combined_data[
                    (self.combined_data.index >= start_date) & 
                    (self.combined_data.index <= end_date)
                ]
                if len(quarter_data) > 10:  # 최소 10개 거래일 이상인 경우만
                    price_corr = quarter_data['TQQQ_Close'].corr(quarter_data['VIX_Close'])
                    return_corr = quarter_data['TQQQ_Return'].corr(quarter_data['VIX_Return'])
                    quarterly_corr.append({
                        'Year': year,
                        'Quarter': quarter,
                        'Price_Correlation': price_corr,
                        'Return_Correlation': return_corr,
                        'Trading_Days': len(quarter_data)
                    })
        
        quarterly_corr_df = pd.DataFrame(quarterly_corr)
        
        # VIX 레벨별 상관관계 분석
        vix_levels = [15, 20, 25, 30, 35, 40, 50]
        vix_level_analysis = []
        
        for i in range(len(vix_levels) - 1):
            low_vix = vix_levels[i]
            high_vix = vix_levels[i + 1]
            level_data = self.combined_data[
                (self.combined_data['VIX_Close'] >= low_vix) & 
                (self.combined_data['VIX_Close'] < high_vix)
            ]
            if len(level_data) > 20:
                price_corr = level_data['TQQQ_Close'].corr(level_data['VIX_Close'])
                return_corr = level_data['TQQQ_Return'].corr(level_data['VIX_Return'])
                vix_level_analysis.append({
                    'VIX_Range': f"{low_vix}-{high_vix}",
                    'Price_Correlation': price_corr,
                    'Return_Correlation': return_corr,
                    'Trading_Days': len(level_data),
                    'Avg_VIX': level_data['VIX_Close'].mean(),
                    'Avg_TQQQ': level_data['TQQQ_Close'].mean()
                })
        
        vix_level_df = pd.DataFrame(vix_level_analysis)
        
        # 통계적 유의성 검정
        t_stat, p_value = stats.pearsonr(
            self.combined_data['TQQQ_Return'].dropna(),
            self.combined_data['VIX_Return'].dropna()
        )
        
        self.analysis_results = {
            'overall_correlation': {
                'price_correlation': overall_corr_price,
                'return_correlation': overall_corr_return,
                't_statistic': t_stat,
                'p_value': p_value
            },
            'yearly_correlation': yearly_corr_df,
            'quarterly_correlation': quarterly_corr_df,
            'vix_level_correlation': vix_level_df,
            'data_summary': {
                'total_trading_days': len(self.combined_data),
                'analysis_start': self.combined_data.index[0].strftime('%Y-%m-%d'),
                'analysis_end': self.combined_data.index[-1].strftime('%Y-%m-%d'),
                'avg_tqqq_price': self.combined_data['TQQQ_Close'].mean(),
                'avg_vix_price': self.combined_data['VIX_Close'].mean(),
                'tqqq_volatility': self.combined_data['TQQQ_Return'].std() * np.sqrt(252),
                'vix_volatility': self.combined_data['VIX_Return'].std() * np.sqrt(252)
            }
        }
        
        return self.analysis_results
    
    def create_visualizations(self):
        """시각화를 생성합니다."""
        if self.combined_data is None or not self.analysis_results:
            print("분석 데이터가 없습니다.")
            return
        
        # 1. 기본 가격 차트
        fig1 = make_subplots(
            rows=2, cols=1,
            subplot_titles=('TQQQ vs VIX 가격 비교', 'TQQQ와 VIX 수익률 비교'),
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4]
        )
        
        # TQQQ 가격 (왼쪽 Y축)
        fig1.add_trace(
            go.Scatter(
                x=self.combined_data.index,
                y=self.combined_data['TQQQ_Close'],
                name='TQQQ Close',
                line=dict(color='blue', width=1),
                yaxis='y1'
            ),
            row=1, col=1
        )
        
        # VIX 가격 (오른쪽 Y축)
        fig1.add_trace(
            go.Scatter(
                x=self.combined_data.index,
                y=self.combined_data['VIX_Close'],
                name='VIX Close',
                line=dict(color='red', width=1),
                yaxis='y2'
            ),
            row=1, col=1
        )
        
        # 수익률 차트
        fig1.add_trace(
            go.Scatter(
                x=self.combined_data.index,
                y=self.combined_data['TQQQ_Return'] * 100,
                name='TQQQ Return (%)',
                line=dict(color='blue', width=1),
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig1.add_trace(
            go.Scatter(
                x=self.combined_data.index,
                y=self.combined_data['VIX_Return'] * 100,
                name='VIX Return (%)',
                line=dict(color='red', width=1),
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig1.update_layout(
            title='TQQQ vs VIX 가격 및 수익률 비교',
            height=800,
            showlegend=True
        )
        
        fig1.update_yaxes(title_text="TQQQ 가격 ($)", row=1, col=1)
        fig1.update_yaxes(title_text="VIX 지수", row=1, col=1, secondary_y=True)
        fig1.update_yaxes(title_text="수익률 (%)", row=2, col=1)
        
        # 2. 상관관계 산점도
        fig2 = go.Figure()
        
        fig2.add_trace(
            go.Scatter(
                x=self.combined_data['VIX_Return'] * 100,
                y=self.combined_data['TQQQ_Return'] * 100,
                mode='markers',
                marker=dict(
                    size=4,
                    color=self.combined_data.index.astype('int64') // 10**9,  # Unix timestamp로 변환
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="날짜")
                ),
                name='일간 수익률',
                text=self.combined_data.index.strftime('%Y-%m-%d'),
                hovertemplate='<b>%{text}</b><br>VIX 수익률: %{x:.2f}%<br>TQQQ 수익률: %{y:.2f}%<extra></extra>'
            )
        )
        
        # 상관관계 선 추가
        vix_returns = self.combined_data['VIX_Return'].dropna() * 100
        tqqq_returns = self.combined_data['TQQQ_Return'].dropna() * 100
        z = np.polyfit(vix_returns, tqqq_returns, 1)
        p = np.poly1d(z)
        
        # 전체 상관계수 가져오기
        overall_corr_return = self.analysis_results['overall_correlation']['return_correlation']
        
        fig2.add_trace(
            go.Scatter(
                x=vix_returns,
                y=p(vix_returns),
                mode='lines',
                name=f'상관관계 선 (R²={overall_corr_return**2:.3f})',
                line=dict(color='red', width=2)
            )
        )
        
        fig2.update_layout(
            title='TQQQ vs VIX 수익률 상관관계',
            xaxis_title='VIX 일간 수익률 (%)',
            yaxis_title='TQQQ 일간 수익률 (%)',
            height=600
        )
        
        # 3. 이동 상관관계 차트
        fig3 = go.Figure()
        
        fig3.add_trace(
            go.Scatter(
                x=self.combined_data.index,
                y=self.combined_data['Correlation_20d'],
                name='20일 이동상관관계',
                line=dict(color='blue', width=2)
            )
        )
        
        fig3.add_trace(
            go.Scatter(
                x=self.combined_data.index,
                y=self.combined_data['Correlation_60d'],
                name='60일 이동상관관계',
                line=dict(color='red', width=2)
            )
        )
        
        fig3.add_hline(
            y=0,
            line_dash="dash",
            line_color="black",
            annotation_text="무상관 기준선"
        )
        
        fig3.update_layout(
            title='TQQQ와 VIX 간 이동 상관관계',
            xaxis_title='날짜',
            yaxis_title='상관계수',
            height=500
        )
        
        # 4. 연도별 상관관계
        fig4 = go.Figure()
        
        fig4.add_trace(
            go.Bar(
                x=self.analysis_results['yearly_correlation']['Year'],
                y=self.analysis_results['yearly_correlation']['Return_Correlation'],
                name='연간 상관계수',
                marker_color='lightblue'
            )
        )
        
        fig4.update_layout(
            title='연도별 TQQQ-VIX 수익률 상관관계',
            xaxis_title='연도',
            yaxis_title='상관계수',
            height=500
        )
        
        # 5. VIX 레벨별 상관관계
        fig5 = go.Figure()
        
        fig5.add_trace(
            go.Bar(
                x=self.analysis_results['vix_level_correlation']['VIX_Range'],
                y=self.analysis_results['vix_level_correlation']['Return_Correlation'],
                name='VIX 레벨별 상관계수',
                marker_color='lightcoral',
                text=self.analysis_results['vix_level_correlation']['Trading_Days'],
                textposition='outside'
            )
        )
        
        fig5.update_layout(
            title='VIX 레벨별 TQQQ-VIX 상관관계',
            xaxis_title='VIX 레벨',
            yaxis_title='상관계수',
            height=500
        )
        
        return [fig1, fig2, fig3, fig4, fig5]
    
    def save_results(self, filename=None):
        """분석 결과를 Excel 파일로 저장합니다."""
        if not self.analysis_results:
            print("분석 결과가 없습니다.")
            return
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"TQQQ_VIX_correlation_analysis_{timestamp}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 전체 데이터
            self.combined_data.to_excel(writer, sheet_name='전체_데이터')
            
            # 연도별 상관관계
            self.analysis_results['yearly_correlation'].to_excel(writer, sheet_name='연도별_상관관계', index=False)
            
            # 분기별 상관관계
            self.analysis_results['quarterly_correlation'].to_excel(writer, sheet_name='분기별_상관관계', index=False)
            
            # VIX 레벨별 상관관계
            self.analysis_results['vix_level_correlation'].to_excel(writer, sheet_name='VIX레벨별_상관관계', index=False)
            
            # 분석 요약
            summary_data = {
                '항목': [
                    '전체 기간 가격 상관계수',
                    '전체 기간 수익률 상관계수',
                    't-통계량',
                    'p-값',
                    '분석 기간 시작',
                    '분석 기간 종료',
                    '총 거래일',
                    '평균 TQQQ 가격',
                    '평균 VIX 지수',
                    'TQQQ 연간 변동성',
                    'VIX 연간 변동성'
                ],
                '값': [
                    round(self.analysis_results['overall_correlation']['price_correlation'], 4),
                    round(self.analysis_results['overall_correlation']['return_correlation'], 4),
                    round(self.analysis_results['overall_correlation']['t_statistic'], 4),
                    round(self.analysis_results['overall_correlation']['p_value'], 6),
                    self.analysis_results['data_summary']['analysis_start'],
                    self.analysis_results['data_summary']['analysis_end'],
                    self.analysis_results['data_summary']['total_trading_days'],
                    round(self.analysis_results['data_summary']['avg_tqqq_price'], 2),
                    round(self.analysis_results['data_summary']['avg_vix_price'], 2),
                    round(self.analysis_results['data_summary']['tqqq_volatility'], 4),
                    round(self.analysis_results['data_summary']['vix_volatility'], 4)
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='분석_요약', index=False)
        
        print(f"결과가 {filename}에 저장되었습니다.")
        return filename
    
    def print_summary(self):
        """분석 결과 요약을 출력합니다."""
        if not self.analysis_results:
            print("분석 결과가 없습니다.")
            return
            
        print("\n" + "="*80)
        print("TQQQ와 VIX 상관관계 분석 결과")
        print("="*80)
        
        overall = self.analysis_results['overall_correlation']
        summary = self.analysis_results['data_summary']
        
        print(f"분석 기간: {summary['analysis_start']} ~ {summary['analysis_end']}")
        print(f"총 거래일: {summary['total_trading_days']:,}일")
        print(f"평균 TQQQ 가격: ${summary['avg_tqqq_price']:.2f}")
        print(f"평균 VIX 지수: {summary['avg_vix_price']:.2f}")
        print()
        
        print("=== 상관관계 분석 ===")
        print(f"전체 기간 가격 상관계수: {overall['price_correlation']:.4f}")
        print(f"전체 기간 수익률 상관계수: {overall['return_correlation']:.4f}")
        print(f"t-통계량: {overall['t_statistic']:.4f}")
        print(f"p-값: {overall['p_value']:.6f}")
        print()
        
        # 상관관계 해석
        corr_value = overall['return_correlation']
        if abs(corr_value) < 0.1:
            interpretation = "거의 무상관"
        elif abs(corr_value) < 0.3:
            interpretation = "약한 상관관계"
        elif abs(corr_value) < 0.5:
            interpretation = "중간 상관관계"
        elif abs(corr_value) < 0.7:
            interpretation = "강한 상관관계"
        else:
            interpretation = "매우 강한 상관관계"
        
        direction = "음의 상관관계" if corr_value < 0 else "양의 상관관계"
        print(f"해석: {direction}, {interpretation}")
        
        if overall['p_value'] < 0.05:
            print("통계적으로 유의함 (p < 0.05)")
        else:
            print("통계적으로 유의하지 않음 (p >= 0.05)")
        
        print()
        print("=== 연도별 상관관계 ===")
        yearly_data = self.analysis_results['yearly_correlation']
        for _, row in yearly_data.tail(5).iterrows():
            print(f"{int(row['Year'])}년: {row['Return_Correlation']:.4f}")
        
        print()
        print("=== VIX 레벨별 상관관계 ===")
        vix_data = self.analysis_results['vix_level_correlation']
        for _, row in vix_data.iterrows():
            print(f"VIX {row['VIX_Range']}: {row['Return_Correlation']:.4f} ({int(row['Trading_Days'])}일)")

def main():
    """메인 실행 함수"""
    print("TQQQ와 VIX 상관관계 분석을 시작합니다...")
    
    # 분석기 초기화
    analyzer = TQQQVIXCorrelationAnalyzer()
    
    # 데이터 수집
    if not analyzer.fetch_data():
        return
    
    # 수익률 및 변동성 계산
    print("\n수익률 및 변동성을 계산하는 중...")
    analyzer.calculate_returns_and_volatility()
    
    # 상관관계 분석
    print("상관관계를 분석하는 중...")
    analyzer.analyze_correlation()
    
    # 결과 출력
    analyzer.print_summary()
    
    # 시각화
    print("\n시각화를 생성하는 중...")
    figures = analyzer.create_visualizations()
    if figures:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 각 차트를 개별 HTML 파일로 저장
        chart_names = [
            'price_comparison',
            'correlation_scatter',
            'rolling_correlation',
            'yearly_correlation',
            'vix_level_correlation'
        ]
        
        for i, (fig, name) in enumerate(zip(figures, chart_names)):
            filename = f"TQQQ_VIX_{name}_{timestamp}.html"
            fig.write_html(filename)
            print(f"차트가 {filename}에 저장되었습니다.")
        
        # 대시보드 HTML 생성
        create_dashboard(figures, timestamp)
    
    # Excel 파일 저장
    print("\n결과를 Excel 파일로 저장하는 중...")
    analyzer.save_results()
    
    print("\n분석이 완료되었습니다!")

def create_dashboard(figures, timestamp):
    """대시보드 HTML을 생성합니다."""
    dashboard_filename = f"TQQQ_VIX_Dashboard_{timestamp}.html"
    
    # 첫 번째 차트를 기준으로 대시보드 생성
    dashboard_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>TQQQ vs VIX 상관관계 분석 대시보드</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .chart {{ margin: 20px 0; }}
            h1 {{ color: #333; text-align: center; }}
            .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            iframe {{ width: 100%; height: 600px; border: none; }}
        </style>
    </head>
    <body>
        <h1>TQQQ vs VIX 상관관계 분석 대시보드</h1>
        <div class="summary">
            <h3>분석 요약</h3>
            <p>생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>이 대시보드는 TQQQ와 VIX 간의 상관관계를 다양한 관점에서 분석한 결과를 보여줍니다.</p>
            <p>개별 차트 파일들을 확인하여 상세한 분석 결과를 확인하세요.</p>
        </div>
        
        <h2>개별 차트 파일들:</h2>
        <ul>
            <li>TQQQ_VIX_price_comparison_{timestamp}.html - 가격 비교 차트</li>
            <li>TQQQ_VIX_correlation_scatter_{timestamp}.html - 상관관계 산점도</li>
            <li>TQQQ_VIX_rolling_correlation_{timestamp}.html - 이동 상관관계</li>
            <li>TQQQ_VIX_yearly_correlation_{timestamp}.html - 연도별 상관관계</li>
            <li>TQQQ_VIX_vix_level_correlation_{timestamp}.html - VIX 레벨별 상관관계</li>
        </ul>
    </body>
    </html>
    """
    
    with open(dashboard_filename, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    print(f"대시보드가 {dashboard_filename}에 저장되었습니다.")

if __name__ == "__main__":
    main()
