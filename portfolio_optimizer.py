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

class TQQQVIXPortfolioOptimizer:
    def __init__(self):
        self.tqqq_ticker = "TQQQ"
        self.vix_ticker = "^VIX"
        self.tqqq_data = None
        self.vix_data = None
        self.combined_data = None
        self.portfolio_results = None
        
    def fetch_data(self, start_date=None, end_date=None):
        """TQQQ와 VIX 데이터를 가져옵니다."""
        print("TQQQ와 VIX 데이터를 수집하는 중...")
        
        if start_date is None:
            start_date = "2020-01-01"  # 최근 5년 데이터 사용
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        try:
            # TQQQ 데이터 수집
            tqqq_raw = yf.download(self.tqqq_ticker, start=start_date, end=end_date, progress=False)
            if tqqq_raw.empty:
                raise ValueError("TQQQ 데이터를 가져올 수 없습니다.")
            
            # VIX 데이터 수집
            vix_raw = yf.download(self.vix_ticker, start=start_date, end=end_date, progress=False)
            if vix_raw.empty:
                raise ValueError("VIX 데이터를 가져올 수 없습니다.")
            
            # MultiIndex 컬럼 처리
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
            
            # 결합된 데이터
            self.combined_data = pd.DataFrame({
                'TQQQ_Close': self.tqqq_data['Close'],
                'VIX_Close': self.vix_data['Close'],
                'TQQQ_Volume': self.tqqq_data['Volume'],
                'VIX_Volume': self.vix_data['Volume']
            }).dropna()
            
            print(f"데이터 수집 완료: {len(self.combined_data)}개 거래일")
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
        self.combined_data['VIX_Return'] = self.combined_data['VIX_Close'].pct_change()
        
        # 연간화된 수익률과 변동성
        tqqq_annual_return = self.combined_data['TQQQ_Return'].mean() * 252
        vix_annual_return = self.combined_data['VIX_Return'].mean() * 252
        
        tqqq_annual_vol = self.combined_data['TQQQ_Return'].std() * np.sqrt(252)
        vix_annual_vol = self.combined_data['VIX_Return'].std() * np.sqrt(252)
        
        # 상관관계
        correlation = self.combined_data['TQQQ_Return'].corr(self.combined_data['VIX_Return'])
        
        # 공분산 행렬
        returns_matrix = self.combined_data[['TQQQ_Return', 'VIX_Return']].dropna()
        cov_matrix = returns_matrix.cov() * 252  # 연간화
        
        self.metrics = {
            'tqqq_return': tqqq_annual_return,
            'vix_return': vix_annual_return,
            'tqqq_volatility': tqqq_annual_vol,
            'vix_volatility': vix_annual_vol,
            'correlation': correlation,
            'covariance_matrix': cov_matrix
        }
        
        print("수익률 및 리스크 지표 계산 완료")
        return self.metrics
    
    def optimize_portfolio(self, target_return=None, risk_free_rate=0.03):
        """포트폴리오 최적화를 수행합니다."""
        if not hasattr(self, 'metrics'):
            self.calculate_returns_and_metrics()
        
        returns = np.array([self.metrics['tqqq_return'], self.metrics['vix_return']])
        cov_matrix = self.metrics['covariance_matrix'].values
        
        # 효율적 프론티어 계산
        portfolio_weights = []
        portfolio_returns = []
        portfolio_volatilities = []
        sharpe_ratios = []
        
        # 다양한 TQQQ 비중으로 포트폴리오 계산 (0% ~ 100%)
        for tqqq_weight in np.arange(0.0, 1.01, 0.01):
            vix_weight = 1 - tqqq_weight
            
            # 포트폴리오 수익률
            portfolio_return = tqqq_weight * returns[0] + vix_weight * returns[1]
            
            # 포트폴리오 변동성
            portfolio_variance = (tqqq_weight**2 * cov_matrix[0,0] + 
                                vix_weight**2 * cov_matrix[1,1] + 
                                2 * tqqq_weight * vix_weight * cov_matrix[0,1])
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # 샤프 비율
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            
            portfolio_weights.append([tqqq_weight, vix_weight])
            portfolio_returns.append(portfolio_return)
            portfolio_volatilities.append(portfolio_volatility)
            sharpe_ratios.append(sharpe_ratio)
        
        # 최적 포트폴리오 찾기
        portfolio_df = pd.DataFrame({
            'TQQQ_Weight': [w[0] for w in portfolio_weights],
            'VIX_Weight': [w[1] for w in portfolio_weights],
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
        
        # 목표 수익률이 주어진 경우
        target_portfolio = None
        if target_return is not None:
            # 목표 수익률에 가장 가까운 포트폴리오 찾기
            target_idx = (portfolio_df['Return'] - target_return).abs().idxmin()
            target_portfolio = portfolio_df.loc[target_idx]
        
        # 특정 비중 조합들 분석
        specific_portfolios = {}
        for tqqq_pct in [90, 80, 70, 60, 50]:
            tqqq_weight = tqqq_pct / 100
            vix_weight = 1 - tqqq_weight
            
            portfolio_return = tqqq_weight * returns[0] + vix_weight * returns[1]
            portfolio_variance = (tqqq_weight**2 * cov_matrix[0,0] + 
                                vix_weight**2 * cov_matrix[1,1] + 
                                2 * tqqq_weight * vix_weight * cov_matrix[0,1])
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            
            specific_portfolios[f'{tqqq_pct}%_TQQQ'] = {
                'TQQQ_Weight': tqqq_weight,
                'VIX_Weight': vix_weight,
                'Return': portfolio_return,
                'Volatility': portfolio_volatility,
                'Sharpe_Ratio': sharpe_ratio
            }
        
        self.portfolio_results = {
            'efficient_frontier': portfolio_df,
            'max_sharpe_portfolio': max_sharpe_portfolio,
            'min_volatility_portfolio': min_vol_portfolio,
            'target_portfolio': target_portfolio,
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
        hedge_ratios = np.arange(0.0, 0.51, 0.05)  # 0% ~ 50% VIX 헤지
        hedge_analysis = []
        
        tqqq_returns = self.combined_data['TQQQ_Return'].dropna()
        vix_returns = self.combined_data['VIX_Return'].dropna()
        
        for hedge_ratio in hedge_ratios:
            # 포트폴리오 수익률 (TQQQ + VIX 헤지)
            portfolio_returns = tqqq_returns + hedge_ratio * vix_returns
            
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
                'VIX_Weight': hedge_ratio,
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
        if not self.portfolio_results:
            print("먼저 포트폴리오 최적화를 수행해주세요.")
            return
        
        figures = []
        
        # 1. 효율적 프론티어
        ef_data = self.portfolio_results['efficient_frontier']
        max_sharpe = self.portfolio_results['max_sharpe_portfolio']
        min_vol = self.portfolio_results['min_volatility_portfolio']
        
        fig1 = go.Figure()
        
        # 효율적 프론티어
        fig1.add_trace(go.Scatter(
            x=ef_data['Volatility'] * 100,
            y=ef_data['Return'] * 100,
            mode='lines',
            name='효율적 프론티어',
            line=dict(color='blue', width=2),
            hovertemplate='변동성: %{x:.2f}%<br>수익률: %{y:.2f}%<extra></extra>'
        ))
        
        # 최대 샤프 비율 포트폴리오
        fig1.add_trace(go.Scatter(
            x=[max_sharpe['Volatility'] * 100],
            y=[max_sharpe['Return'] * 100],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name=f'최대 샤프 비율<br>TQQQ: {max_sharpe["TQQQ_Weight"]:.1%}<br>VIX: {max_sharpe["VIX_Weight"]:.1%}',
            hovertemplate=f'변동성: %{{x:.2f}}%<br>수익률: %{{y:.2f}}%<br>샤프 비율: {max_sharpe["Sharpe_Ratio"]:.3f}<extra></extra>'
        ))
        
        # 최소 변동성 포트폴리오
        fig1.add_trace(go.Scatter(
            x=[min_vol['Volatility'] * 100],
            y=[min_vol['Return'] * 100],
            mode='markers',
            marker=dict(size=15, color='green', symbol='diamond'),
            name=f'최소 변동성<br>TQQQ: {min_vol["TQQQ_Weight"]:.1%}<br>VIX: {min_vol["VIX_Weight"]:.1%}',
            hovertemplate='변동성: %{x:.2f}%<br>수익률: %{y:.2f}%<extra></extra>'
        ))
        
        fig1.update_layout(
            title='TQQQ-VIX 포트폴리오 효율적 프론티어',
            xaxis_title='연간 변동성 (%)',
            yaxis_title='연간 수익률 (%)',
            height=600
        )
        
        figures.append(fig1)
        
        # 2. 샤프 비율 차트
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=ef_data['TQQQ_Weight'] * 100,
            y=ef_data['Sharpe_Ratio'],
            mode='lines',
            name='샤프 비율',
            line=dict(color='purple', width=2),
            hovertemplate='TQQQ 비중: %{x:.1f}%<br>샤프 비율: %{y:.3f}<extra></extra>'
        ))
        
        # 최대 샤프 비율 지점 표시
        fig2.add_trace(go.Scatter(
            x=[max_sharpe['TQQQ_Weight'] * 100],
            y=[max_sharpe['Sharpe_Ratio']],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='최적 포트폴리오',
            hovertemplate='TQQQ: %{x:.1f}%<br>샤프 비율: %{y:.3f}<extra></extra>'
        ))
        
        fig2.update_layout(
            title='TQQQ 비중별 샤프 비율',
            xaxis_title='TQQQ 비중 (%)',
            yaxis_title='샤프 비율',
            height=500
        )
        
        figures.append(fig2)
        
        # 3. 헤지 분석 (있는 경우)
        if hasattr(self, 'hedge_analysis'):
            fig3 = go.Figure()
            
            fig3.add_trace(go.Scatter(
                x=self.hedge_analysis['Hedge_Ratio'] * 100,
                y=self.hedge_analysis['Sharpe_Ratio'],
                mode='lines+markers',
                name='샤프 비율',
                line=dict(color='blue', width=2),
                marker=dict(size=6),
                hovertemplate='VIX 헤지 비율: %{x:.1f}%<br>샤프 비율: %{y:.3f}<extra></extra>'
            ))
            
            fig3.update_layout(
                title='VIX 헤지 비율별 샤프 비율',
                xaxis_title='VIX 헤지 비율 (%)',
                yaxis_title='샤프 비율',
                height=500
            )
            
            figures.append(fig3)
            
            # 4. 리스크 지표 비교
            fig4 = make_subplots(
                rows=2, cols=2,
                subplot_titles=('연간 수익률', '변동성', '최대 낙폭', 'VaR (95%)'),
                vertical_spacing=0.1
            )
            
            fig4.add_trace(go.Scatter(
                x=self.hedge_analysis['Hedge_Ratio'] * 100,
                y=self.hedge_analysis['Annual_Return'] * 100,
                mode='lines+markers',
                name='연간 수익률',
                line=dict(color='green', width=2)
            ), row=1, col=1)
            
            fig4.add_trace(go.Scatter(
                x=self.hedge_analysis['Hedge_Ratio'] * 100,
                y=self.hedge_analysis['Volatility'] * 100,
                mode='lines+markers',
                name='변동성',
                line=dict(color='red', width=2)
            ), row=1, col=2)
            
            fig4.add_trace(go.Scatter(
                x=self.hedge_analysis['Hedge_Ratio'] * 100,
                y=self.hedge_analysis['Max_Drawdown'] * 100,
                mode='lines+markers',
                name='최대 낙폭',
                line=dict(color='orange', width=2)
            ), row=2, col=1)
            
            fig4.add_trace(go.Scatter(
                x=self.hedge_analysis['Hedge_Ratio'] * 100,
                y=self.hedge_analysis['VaR_95'] * 100,
                mode='lines+markers',
                name='VaR (95%)',
                line=dict(color='purple', width=2)
            ), row=2, col=2)
            
            fig4.update_layout(
                title='VIX 헤지 비율별 리스크 지표',
                height=700,
                showlegend=False
            )
            
            # X축 라벨 설정
            for i in range(1, 3):
                for j in range(1, 3):
                    fig4.update_xaxes(title_text='VIX 헤지 비율 (%)', row=i, col=j)
            
            figures.append(fig4)
        
        return figures
    
    def get_recommendations(self):
        """투자 추천을 제공합니다."""
        if not self.portfolio_results:
            print("먼저 포트폴리오 최적화를 수행해주세요.")
            return
        
        max_sharpe = self.portfolio_results['max_sharpe_portfolio']
        min_vol = self.portfolio_results['min_volatility_portfolio']
        
        recommendations = {
            'optimal_portfolio': {
                'description': '최대 샤프 비율 포트폴리오 (최적 리스크-수익률 균형)',
                'tqqq_weight': max_sharpe['TQQQ_Weight'],
                'vix_weight': max_sharpe['VIX_Weight'],
                'expected_return': max_sharpe['Return'],
                'volatility': max_sharpe['Volatility'],
                'sharpe_ratio': max_sharpe['Sharpe_Ratio']
            },
            'conservative_portfolio': {
                'description': '최소 변동성 포트폴리오 (안정성 우선)',
                'tqqq_weight': min_vol['TQQQ_Weight'],
                'vix_weight': min_vol['VIX_Weight'],
                'expected_return': min_vol['Return'],
                'volatility': min_vol['Volatility'],
                'sharpe_ratio': min_vol['Sharpe_Ratio']
            }
        }
        
        # 특정 비중 조합들 추가
        if 'specific_portfolios' in self.portfolio_results:
            recommendations['common_portfolios'] = self.portfolio_results['specific_portfolios']
        
        return recommendations
    
    def save_results(self, filename=None):
        """분석 결과를 Excel 파일로 저장합니다."""
        if not self.portfolio_results:
            print("분석 결과가 없습니다.")
            return
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"TQQQ_VIX_Portfolio_Optimization_{timestamp}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 효율적 프론티어
            self.portfolio_results['efficient_frontier'].to_excel(writer, sheet_name='효율적_프론티어', index=False)
            
            # 헤지 분석 (있는 경우)
            if hasattr(self, 'hedge_analysis'):
                self.hedge_analysis.to_excel(writer, sheet_name='헤지_분석', index=False)
            
            # 추천 포트폴리오
            recommendations = self.get_recommendations()
            
            optimal_data = recommendations['optimal_portfolio']
            conservative_data = recommendations['conservative_portfolio']
            
            portfolio_summary = pd.DataFrame({
                '포트폴리오': ['최적 포트폴리오', '보수적 포트폴리오'],
                '설명': [optimal_data['description'], conservative_data['description']],
                'TQQQ_비중': [optimal_data['tqqq_weight'], conservative_data['tqqq_weight']],
                'VIX_비중': [optimal_data['vix_weight'], conservative_data['vix_weight']],
                '예상_수익률': [optimal_data['expected_return'], conservative_data['expected_return']],
                '변동성': [optimal_data['volatility'], conservative_data['volatility']],
                '샤프_비율': [optimal_data['sharpe_ratio'], conservative_data['sharpe_ratio']]
            })
            
            portfolio_summary.to_excel(writer, sheet_name='포트폴리오_추천', index=False)
            
            # 기본 통계
            if hasattr(self, 'metrics'):
                stats_data = {
                    '지표': [
                        'TQQQ 연간 수익률',
                        'VIX 연간 수익률',
                        'TQQQ 연간 변동성',
                        'VIX 연간 변동성',
                        '상관계수',
                        '분석 기간 시작',
                        '분석 기간 종료',
                        '총 거래일'
                    ],
                    '값': [
                        self.metrics['tqqq_return'],
                        self.metrics['vix_return'],
                        self.metrics['tqqq_volatility'],
                        self.metrics['vix_volatility'],
                        self.metrics['correlation'],
                        self.combined_data.index[0].strftime('%Y-%m-%d'),
                        self.combined_data.index[-1].strftime('%Y-%m-%d'),
                        len(self.combined_data)
                    ]
                }
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='기본_통계', index=False)
        
        print(f"포트폴리오 최적화 결과가 {filename}에 저장되었습니다.")
        return filename
    
    def print_recommendations(self):
        """투자 추천을 출력합니다."""
        if not self.portfolio_results:
            print("분석 결과가 없습니다.")
            return
        
        recommendations = self.get_recommendations()
        
        print("\n" + "="*80)
        print("TQQQ-VIX 포트폴리오 최적화 추천")
        print("="*80)
        
        optimal = recommendations['optimal_portfolio']
        conservative = recommendations['conservative_portfolio']
        
        print(f"\n[최적 포트폴리오] (최대 샤프 비율)")
        print(f"   TQQQ 비중: {optimal['tqqq_weight']:.1%}")
        print(f"   VIX 비중: {optimal['vix_weight']:.1%}")
        print(f"   예상 연간 수익률: {optimal['expected_return']:.1%}")
        print(f"   예상 연간 변동성: {optimal['volatility']:.1%}")
        print(f"   샤프 비율: {optimal['sharpe_ratio']:.3f}")
        
        print(f"\n[보수적 포트폴리오] (최소 변동성)")
        print(f"   TQQQ 비중: {conservative['tqqq_weight']:.1%}")
        print(f"   VIX 비중: {conservative['vix_weight']:.1%}")
        print(f"   예상 연간 수익률: {conservative['expected_return']:.1%}")
        print(f"   예상 연간 변동성: {conservative['volatility']:.1%}")
        print(f"   샤프 비율: {conservative['sharpe_ratio']:.3f}")
        
        if 'common_portfolios' in recommendations:
            print(f"\n[일반적인 포트폴리오 조합들]")
            for name, portfolio in recommendations['common_portfolios'].items():
                tqqq_pct = portfolio['TQQQ_Weight'] * 100
                vix_pct = portfolio['VIX_Weight'] * 100
                print(f"   {name}: TQQQ {tqqq_pct:.0f}%, VIX {vix_pct:.0f}% (샤프: {portfolio['Sharpe_Ratio']:.3f})")
        
        # 헤지 분석 결과 (있는 경우)
        if hasattr(self, 'hedge_analysis'):
            best_hedge = self.hedge_analysis.loc[self.hedge_analysis['Sharpe_Ratio'].idxmax()]
            print(f"\n[최적 헤지 비율]")
            print(f"   VIX 헤지 비율: {best_hedge['Hedge_Ratio']:.1%}")
            print(f"   TQQQ 비중: {best_hedge['TQQQ_Weight']:.1%}")
            print(f"   샤프 비율: {best_hedge['Sharpe_Ratio']:.3f}")

def main():
    """메인 실행 함수"""
    print("TQQQ-VIX 포트폴리오 최적화 분석을 시작합니다...")
    
    # 분석기 초기화
    optimizer = TQQQVIXPortfolioOptimizer()
    
    # 데이터 수집
    if not optimizer.fetch_data():
        return
    
    # 수익률 및 리스크 지표 계산
    print("\n수익률 및 리스크 지표를 계산하는 중...")
    optimizer.calculate_returns_and_metrics()
    
    # 포트폴리오 최적화
    print("포트폴리오 최적화를 수행하는 중...")
    optimizer.optimize_portfolio()
    
    # 헤지 효과 분석
    print("헤지 효과를 분석하는 중...")
    optimizer.analyze_hedge_effectiveness()
    
    # 추천 출력
    optimizer.print_recommendations()
    
    # 시각화
    print("\n시각화를 생성하는 중...")
    figures = optimizer.create_visualizations()
    if figures:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        chart_names = [
            'efficient_frontier',
            'sharpe_ratio',
            'hedge_effectiveness',
            'risk_metrics'
        ]
        
        for i, (fig, name) in enumerate(zip(figures, chart_names)):
            filename = f"TQQQ_VIX_{name}_{timestamp}.html"
            fig.write_html(filename)
            print(f"차트가 {filename}에 저장되었습니다.")
    
    # Excel 파일 저장
    print("\n결과를 Excel 파일로 저장하는 중...")
    optimizer.save_results()
    
    print("\n포트폴리오 최적화 분석이 완료되었습니다!")

if __name__ == "__main__":
    main()
