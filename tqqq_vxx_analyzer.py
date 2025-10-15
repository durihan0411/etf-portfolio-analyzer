import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class TQQQVXXAnalyzer:
    def __init__(self):
        self.tqqq_ticker = "TQQQ"
        self.vxx_ticker = "VXX"  # VIX ETF
        self.tqqq_data = None
        self.vxx_data = None
        self.combined_data = None
        self.analysis_results = None
        self.portfolio_results = None
        
    def fetch_data(self, start_date=None, end_date=None):
        """TQQQ와 VXX 데이터를 가져옵니다."""
        print("TQQQ와 VXX 데이터를 수집하는 중...")
        
        if start_date is None:
            start_date = "2010-02-09"  # TQQQ 상장일
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        try:
            # TQQQ 데이터 수집
            print("TQQQ 데이터 수집 중...")
            tqqq_raw = yf.download(self.tqqq_ticker, start=start_date, end=end_date, progress=False)
            if tqqq_raw.empty:
                raise ValueError("TQQQ 데이터를 가져올 수 없습니다.")
            
            # VXX 데이터 수집 (VXX는 2009년 1월 30일 시작)
            print("VXX 데이터 수집 중...")
            vxx_raw = yf.download(self.vxx_ticker, start=start_date, end=end_date, progress=False)
            if vxx_raw.empty:
                raise ValueError("VXX 데이터를 가져올 수 없습니다.")
            
            # MultiIndex 컬럼 처리
            if isinstance(tqqq_raw.columns, pd.MultiIndex):
                tqqq_raw.columns = tqqq_raw.columns.droplevel(0)
            if isinstance(vxx_raw.columns, pd.MultiIndex):
                vxx_raw.columns = vxx_raw.columns.droplevel(0)
            
            # 컬럼명 정리
            if len(tqqq_raw.columns) == 6:
                tqqq_raw.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            elif len(tqqq_raw.columns) == 5:
                tqqq_raw.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
            if len(vxx_raw.columns) == 6:
                vxx_raw.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            elif len(vxx_raw.columns) == 5:
                vxx_raw.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            self.tqqq_data = tqqq_raw
            self.vxx_data = vxx_raw
            
            # 결합된 데이터
            self.combined_data = pd.DataFrame({
                'TQQQ_Close': self.tqqq_data['Close'],
                'VXX_Close': self.vxx_data['Close'],
                'TQQQ_Volume': self.tqqq_data['Volume'],
                'VXX_Volume': self.vxx_data['Volume']
            }).dropna()
            
            print(f"TQQQ 데이터 수집 완료: {len(self.tqqq_data)}개 거래일")
            print(f"VXX 데이터 수집 완료: {len(self.vxx_data)}개 거래일")
            print(f"결합된 데이터: {len(self.combined_data)}개 거래일")
            print(f"기간: {self.combined_data.index[0].strftime('%Y-%m-%d')} ~ {self.combined_data.index[-1].strftime('%Y-%m-%d')}")
            return True
            
        except Exception as e:
            print(f"데이터 수집 중 오류 발생: {e}")
            return False
    
    def calculate_returns_and_metrics(self):
        """수익률과 리스크 지표를 계산합니다."""
        if self.combined_data is None:
            print("먼저 데이터를 가져와주세요.")
            return
        
        # 일간 수익률 계산
        self.combined_data['TQQQ_Return'] = self.combined_data['TQQQ_Close'].pct_change()
        self.combined_data['VXX_Return'] = self.combined_data['VXX_Close'].pct_change()
        
        # 로그 수익률 계산
        self.combined_data['TQQQ_Log_Return'] = np.log(self.combined_data['TQQQ_Close'] / self.combined_data['TQQQ_Close'].shift(1))
        self.combined_data['VXX_Log_Return'] = np.log(self.combined_data['VXX_Close'] / self.combined_data['VXX_Close'].shift(1))
        
        # 이동 평균 변동성 (20일, 60일)
        self.combined_data['TQQQ_Volatility_20d'] = self.combined_data['TQQQ_Return'].rolling(window=20).std() * np.sqrt(252)
        self.combined_data['VXX_Volatility_20d'] = self.combined_data['VXX_Return'].rolling(window=20).std() * np.sqrt(252)
        
        self.combined_data['TQQQ_Volatility_60d'] = self.combined_data['TQQQ_Return'].rolling(window=60).std() * np.sqrt(252)
        self.combined_data['VXX_Volatility_60d'] = self.combined_data['VXX_Return'].rolling(window=60).std() * np.sqrt(252)
        
        # 이동 상관관계 계산 (20일, 60일)
        self.combined_data['Correlation_20d'] = self.combined_data['TQQQ_Return'].rolling(window=20).corr(self.combined_data['VXX_Return'])
        self.combined_data['Correlation_60d'] = self.combined_data['TQQQ_Return'].rolling(window=60).corr(self.combined_data['VXX_Return'])
        
        # 연간화된 수익률과 변동성
        tqqq_annual_return = self.combined_data['TQQQ_Return'].mean() * 252
        vxx_annual_return = self.combined_data['VXX_Return'].mean() * 252
        
        tqqq_annual_vol = self.combined_data['TQQQ_Return'].std() * np.sqrt(252)
        vxx_annual_vol = self.combined_data['VXX_Return'].std() * np.sqrt(252)
        
        # 상관관계
        correlation = self.combined_data['TQQQ_Return'].corr(self.combined_data['VXX_Return'])
        
        # 공분산 행렬
        returns_matrix = self.combined_data[['TQQQ_Return', 'VXX_Return']].dropna()
        cov_matrix = returns_matrix.cov() * 252  # 연간화
        
        self.metrics = {
            'tqqq_return': tqqq_annual_return,
            'vxx_return': vxx_annual_return,
            'tqqq_volatility': tqqq_annual_vol,
            'vxx_volatility': vxx_annual_vol,
            'correlation': correlation,
            'covariance_matrix': cov_matrix
        }
        
        print("수익률 및 리스크 지표 계산 완료")
        return self.metrics
    
    def analyze_correlation(self):
        """상관관계를 분석합니다."""
        if self.combined_data is None:
            print("먼저 데이터를 가져와주세요.")
            return
        
        # 전체 기간 상관관계
        overall_corr_price = self.combined_data['TQQQ_Close'].corr(self.combined_data['VXX_Close'])
        overall_corr_return = self.combined_data['TQQQ_Return'].corr(self.combined_data['VXX_Return'])
        
        # 연도별 상관관계
        yearly_corr = []
        for year in self.combined_data.index.year.unique():
            year_data = self.combined_data[self.combined_data.index.year == year]
            if len(year_data) > 20:  # 최소 20개 거래일 이상인 경우만
                price_corr = year_data['TQQQ_Close'].corr(year_data['VXX_Close'])
                return_corr = year_data['TQQQ_Return'].corr(year_data['VXX_Return'])
                yearly_corr.append({
                    'Year': year,
                    'Price_Correlation': price_corr,
                    'Return_Correlation': return_corr,
                    'Trading_Days': len(year_data)
                })
        
        yearly_corr_df = pd.DataFrame(yearly_corr)
        
        # VXX 레벨별 상관관계 분석
        vxx_levels = [10, 20, 30, 40, 50, 60, 80]
        vxx_level_analysis = []
        
        for i in range(len(vxx_levels) - 1):
            low_vxx = vxx_levels[i]
            high_vxx = vxx_levels[i + 1]
            level_data = self.combined_data[
                (self.combined_data['VXX_Close'] >= low_vxx) & 
                (self.combined_data['VXX_Close'] < high_vxx)
            ]
            if len(level_data) > 20:
                price_corr = level_data['TQQQ_Close'].corr(level_data['VXX_Close'])
                return_corr = level_data['TQQQ_Return'].corr(level_data['VXX_Return'])
                vxx_level_analysis.append({
                    'VXX_Range': f"{low_vxx}-{high_vxx}",
                    'Price_Correlation': price_corr,
                    'Return_Correlation': return_corr,
                    'Trading_Days': len(level_data),
                    'Avg_VXX': level_data['VXX_Close'].mean(),
                    'Avg_TQQQ': level_data['TQQQ_Close'].mean()
                })
        
        vxx_level_df = pd.DataFrame(vxx_level_analysis)
        
        # 통계적 유의성 검정
        t_stat, p_value = stats.pearsonr(
            self.combined_data['TQQQ_Return'].dropna(),
            self.combined_data['VXX_Return'].dropna()
        )
        
        self.analysis_results = {
            'overall_correlation': {
                'price_correlation': overall_corr_price,
                'return_correlation': overall_corr_return,
                't_statistic': t_stat,
                'p_value': p_value
            },
            'yearly_correlation': yearly_corr_df,
            'vxx_level_correlation': vxx_level_df,
            'data_summary': {
                'total_trading_days': len(self.combined_data),
                'analysis_start': self.combined_data.index[0].strftime('%Y-%m-%d'),
                'analysis_end': self.combined_data.index[-1].strftime('%Y-%m-%d'),
                'avg_tqqq_price': self.combined_data['TQQQ_Close'].mean(),
                'avg_vxx_price': self.combined_data['VXX_Close'].mean(),
                'tqqq_volatility': self.combined_data['TQQQ_Return'].std() * np.sqrt(252),
                'vxx_volatility': self.combined_data['VXX_Return'].std() * np.sqrt(252)
            }
        }
        
        return self.analysis_results
    
    def optimize_portfolio(self, risk_free_rate=0.03):
        """포트폴리오 최적화를 수행합니다."""
        if not hasattr(self, 'metrics'):
            self.calculate_returns_and_metrics()
        
        returns = np.array([self.metrics['tqqq_return'], self.metrics['vxx_return']])
        cov_matrix = self.metrics['covariance_matrix'].values
        
        # 효율적 프론티어 계산
        portfolio_weights = []
        portfolio_returns = []
        portfolio_volatilities = []
        sharpe_ratios = []
        
        # 다양한 TQQQ 비중으로 포트폴리오 계산 (0% ~ 100%)
        for tqqq_weight in np.arange(0.0, 1.01, 0.01):
            vxx_weight = 1 - tqqq_weight
            
            # 포트폴리오 수익률
            portfolio_return = tqqq_weight * returns[0] + vxx_weight * returns[1]
            
            # 포트폴리오 변동성
            portfolio_variance = (tqqq_weight**2 * cov_matrix[0,0] + 
                                vxx_weight**2 * cov_matrix[1,1] + 
                                2 * tqqq_weight * vxx_weight * cov_matrix[0,1])
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # 샤프 비율
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            
            portfolio_weights.append([tqqq_weight, vxx_weight])
            portfolio_returns.append(portfolio_return)
            portfolio_volatilities.append(portfolio_volatility)
            sharpe_ratios.append(sharpe_ratio)
        
        # 최적 포트폴리오 찾기
        portfolio_df = pd.DataFrame({
            'TQQQ_Weight': [w[0] for w in portfolio_weights],
            'VXX_Weight': [w[1] for w in portfolio_weights],
            'Return': portfolio_returns,
            'Volatility': portfolio_volatilities,
            'Sharpe_Ratio': sharpe_ratios
        })
        
        # 최대 샤프 비율 포트폴리오
        max_sharpe_idx = portfolio_df['Sharpe_Ratio'].idxmax()
        max_sharpe_portfolio = portfolio_df.loc[max_sharpe_idx]
        
        # 최소 변동성 포트폴리오
        min_vol_idx = portfolio_df['Volatility'].idxmin()
        min_vol_portfolio = portfolio_df.loc[min_vol_idx]
        
        # 특정 비중 조합들 분석
        specific_portfolios = {}
        for tqqq_pct in [90, 80, 70, 60, 50]:
            tqqq_weight = tqqq_pct / 100
            vxx_weight = 1 - tqqq_weight
            
            portfolio_return = tqqq_weight * returns[0] + vxx_weight * returns[1]
            portfolio_variance = (tqqq_weight**2 * cov_matrix[0,0] + 
                                vxx_weight**2 * cov_matrix[1,1] + 
                                2 * tqqq_weight * vxx_weight * cov_matrix[0,1])
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            
            specific_portfolios[f'{tqqq_pct}%_TQQQ'] = {
                'TQQQ_Weight': tqqq_weight,
                'VXX_Weight': vxx_weight,
                'Return': portfolio_return,
                'Volatility': portfolio_volatility,
                'Sharpe_Ratio': sharpe_ratio
            }
        
        self.portfolio_results = {
            'efficient_frontier': portfolio_df,
            'max_sharpe_portfolio': max_sharpe_portfolio,
            'min_volatility_portfolio': min_vol_portfolio,
            'specific_portfolios': specific_portfolios,
            'risk_free_rate': risk_free_rate
        }
        
        return self.portfolio_results
    
    def analyze_hedge_effectiveness(self):
        """헤지 효과를 분석합니다."""
        if self.combined_data is None:
            print("먼저 데이터를 가져와주세요.")
            return
        
        # 다양한 헤지 비율에서의 성과 분석
        hedge_ratios = np.arange(0.0, 0.51, 0.05)  # 0% ~ 50% VXX 헤지
        hedge_analysis = []
        
        tqqq_returns = self.combined_data['TQQQ_Return'].dropna()
        vxx_returns = self.combined_data['VXX_Return'].dropna()
        
        for hedge_ratio in hedge_ratios:
            # 포트폴리오 수익률 (TQQQ + VXX 헤지)
            portfolio_returns = tqqq_returns + hedge_ratio * vxx_returns
            
            # 성과 지표 계산
            total_return = (1 + portfolio_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # 최대 낙폭 (Maximum Drawdown)
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # VaR (Value at Risk) - 95% 신뢰구간
            var_95 = np.percentile(portfolio_returns, 5)
            
            hedge_analysis.append({
                'Hedge_Ratio': hedge_ratio,
                'TQQQ_Weight': 1 - hedge_ratio,
                'VXX_Weight': hedge_ratio,
                'Annual_Return': annual_return,
                'Volatility': volatility,
                'Sharpe_Ratio': sharpe_ratio,
                'Max_Drawdown': max_drawdown,
                'VaR_95': var_95
            })
        
        self.hedge_analysis = pd.DataFrame(hedge_analysis)
        return self.hedge_analysis
    
    def create_visualizations(self):
        """시각화를 생성합니다."""
        if self.combined_data is None or not self.analysis_results:
            print("분석 데이터가 없습니다.")
            return
        
        figures = []
        
        # 1. 기본 가격 차트
        fig1 = make_subplots(
            rows=2, cols=1,
            subplot_titles=('TQQQ vs VXX 가격 비교', 'TQQQ와 VXX 수익률 비교'),
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
        
        # VXX 가격 (오른쪽 Y축)
        fig1.add_trace(
            go.Scatter(
                x=self.combined_data.index,
                y=self.combined_data['VXX_Close'],
                name='VXX Close',
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
                y=self.combined_data['VXX_Return'] * 100,
                name='VXX Return (%)',
                line=dict(color='red', width=1),
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig1.update_layout(
            title='TQQQ vs VXX 가격 및 수익률 비교',
            height=800,
            showlegend=True
        )
        
        fig1.update_yaxes(title_text="TQQQ 가격 ($)", row=1, col=1)
        fig1.update_yaxes(title_text="VXX 가격 ($)", row=1, col=1, secondary_y=True)
        fig1.update_yaxes(title_text="수익률 (%)", row=2, col=1)
        
        figures.append(fig1)
        
        # 2. 상관관계 산점도
        fig2 = go.Figure()
        
        fig2.add_trace(
            go.Scatter(
                x=self.combined_data['VXX_Return'].dropna() * 100,
                y=self.combined_data['TQQQ_Return'].dropna() * 100,
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
                hovertemplate='<b>%{text}</b><br>VXX 수익률: %{x:.2f}%<br>TQQQ 수익률: %{y:.2f}%<extra></extra>'
            )
        )
        
        # 상관관계 선 추가
        vxx_returns = self.combined_data['VXX_Return'].dropna() * 100
        tqqq_returns = self.combined_data['TQQQ_Return'].dropna() * 100
        z = np.polyfit(vxx_returns, tqqq_returns, 1)
        p = np.poly1d(z)
        
        overall_corr_return = self.analysis_results['overall_correlation']['return_correlation']
        
        fig2.add_trace(
            go.Scatter(
                x=vxx_returns,
                y=p(vxx_returns),
                mode='lines',
                name=f'상관관계 선 (R²={overall_corr_return**2:.3f})',
                line=dict(color='red', width=2)
            )
        )
        
        fig2.update_layout(
            title='TQQQ vs VXX 수익률 상관관계',
            xaxis_title='VXX 일간 수익률 (%)',
            yaxis_title='TQQQ 일간 수익률 (%)',
            height=600
        )
        
        figures.append(fig2)
        
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
            title='TQQQ와 VXX 간 이동 상관관계',
            xaxis_title='날짜',
            yaxis_title='상관계수',
            height=500
        )
        
        figures.append(fig3)
        
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
            title='연도별 TQQQ-VXX 수익률 상관관계',
            xaxis_title='연도',
            yaxis_title='상관계수',
            height=500
        )
        
        figures.append(fig4)
        
        # 5. VXX 레벨별 상관관계
        if not self.analysis_results['vxx_level_correlation'].empty:
            fig5 = go.Figure()
            
            fig5.add_trace(
                go.Bar(
                    x=self.analysis_results['vxx_level_correlation']['VXX_Range'],
                    y=self.analysis_results['vxx_level_correlation']['Return_Correlation'],
                    name='VXX 레벨별 상관계수',
                    marker_color='lightcoral',
                    text=self.analysis_results['vxx_level_correlation']['Trading_Days'],
                    textposition='outside'
                )
            )
            
            fig5.update_layout(
                title='VXX 레벨별 TQQQ-VXX 상관관계',
                xaxis_title='VXX 레벨',
                yaxis_title='상관계수',
                height=500
            )
            
            figures.append(fig5)
        
        return figures
    
    def save_results(self, filename=None):
        """분석 결과를 Excel 파일로 저장합니다."""
        if not self.analysis_results:
            print("분석 결과가 없습니다.")
            return
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"TQQQ_VXX_analysis_{timestamp}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 전체 데이터
            self.combined_data.to_excel(writer, sheet_name='전체_데이터')
            
            # 연도별 상관관계
            self.analysis_results['yearly_correlation'].to_excel(writer, sheet_name='연도별_상관관계', index=False)
            
            # VXX 레벨별 상관관계
            if not self.analysis_results['vxx_level_correlation'].empty:
                self.analysis_results['vxx_level_correlation'].to_excel(writer, sheet_name='VXX레벨별_상관관계', index=False)
            
            # 포트폴리오 최적화 결과 (있는 경우)
            if self.portfolio_results:
                self.portfolio_results['efficient_frontier'].to_excel(writer, sheet_name='포트폴리오_최적화', index=False)
            
            # 헤지 분석 (있는 경우)
            if hasattr(self, 'hedge_analysis'):
                self.hedge_analysis.to_excel(writer, sheet_name='헤지_분석', index=False)
            
            # 분석 요약
            overall = self.analysis_results['overall_correlation']
            summary = self.analysis_results['data_summary']
            
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
                    '평균 VXX 가격',
                    'TQQQ 연간 변동성',
                    'VXX 연간 변동성'
                ],
                '값': [
                    round(overall['price_correlation'], 4),
                    round(overall['return_correlation'], 4),
                    round(overall['t_statistic'], 4),
                    round(overall['p_value'], 6),
                    summary['analysis_start'],
                    summary['analysis_end'],
                    summary['total_trading_days'],
                    round(summary['avg_tqqq_price'], 2),
                    round(summary['avg_vxx_price'], 2),
                    round(summary['tqqq_volatility'], 4),
                    round(summary['vxx_volatility'], 4)
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
        print("TQQQ와 VXX 상관관계 분석 결과")
        print("="*80)
        
        overall = self.analysis_results['overall_correlation']
        summary = self.analysis_results['data_summary']
        
        print(f"분석 기간: {summary['analysis_start']} ~ {summary['analysis_end']}")
        print(f"총 거래일: {summary['total_trading_days']:,}일")
        print(f"평균 TQQQ 가격: ${summary['avg_tqqq_price']:.2f}")
        print(f"평균 VXX 가격: ${summary['avg_vxx_price']:.2f}")
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
        
        # 포트폴리오 최적화 결과 (있는 경우)
        if self.portfolio_results:
            print("\n=== 포트폴리오 최적화 결과 ===")
            max_sharpe = self.portfolio_results['max_sharpe_portfolio']
            min_vol = self.portfolio_results['min_volatility_portfolio']
            
            print(f"최대 샤프 비율 포트폴리오:")
            print(f"  TQQQ 비중: {max_sharpe['TQQQ_Weight']:.1%}")
            print(f"  VXX 비중: {max_sharpe['VXX_Weight']:.1%}")
            print(f"  예상 수익률: {max_sharpe['Return']:.1%}")
            print(f"  예상 변동성: {max_sharpe['Volatility']:.1%}")
            print(f"  샤프 비율: {max_sharpe['Sharpe_Ratio']:.3f}")
            
            print(f"\n최소 변동성 포트폴리오:")
            print(f"  TQQQ 비중: {min_vol['TQQQ_Weight']:.1%}")
            print(f"  VXX 비중: {min_vol['VXX_Weight']:.1%}")
            print(f"  예상 수익률: {min_vol['Return']:.1%}")
            print(f"  예상 변동성: {min_vol['Volatility']:.1%}")
            print(f"  샤프 비율: {min_vol['Sharpe_Ratio']:.3f}")

def main():
    """메인 실행 함수"""
    print("TQQQ와 VXX 상관관계 분석을 시작합니다...")
    
    # 분석기 초기화
    analyzer = TQQQVXXAnalyzer()
    
    # 데이터 수집
    if not analyzer.fetch_data():
        return
    
    # 수익률 및 변동성 계산
    print("\n수익률 및 변동성을 계산하는 중...")
    analyzer.calculate_returns_and_metrics()
    
    # 상관관계 분석
    print("상관관계를 분석하는 중...")
    analyzer.analyze_correlation()
    
    # 포트폴리오 최적화
    print("포트폴리오 최적화를 수행하는 중...")
    analyzer.optimize_portfolio()
    
    # 헤지 효과 분석
    print("헤지 효과를 분석하는 중...")
    analyzer.analyze_hedge_effectiveness()
    
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
            'vxx_level_correlation'
        ]
        
        for i, (fig, name) in enumerate(zip(figures, chart_names)):
            filename = f"TQQQ_VXX_{name}_{timestamp}.html"
            fig.write_html(filename)
            print(f"차트가 {filename}에 저장되었습니다.")
    
    # Excel 파일 저장
    print("\n결과를 Excel 파일로 저장하는 중...")
    analyzer.save_results()
    
    print("\n분석이 완료되었습니다!")

if __name__ == "__main__":
    main()
