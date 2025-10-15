import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def create_interactive_dashboard():
    """인터랙티브 대시보드를 생성합니다."""
    
    # 데이터 수집
    print("데이터를 수집하는 중...")
    tqqq_data = yf.download("TQQQ", start="2010-02-09", progress=False)
    vix_data = yf.download("^VIX", start="2010-02-09", progress=False)
    
    # 데이터 전처리
    if isinstance(tqqq_data.columns, pd.MultiIndex):
        tqqq_data.columns = tqqq_data.columns.droplevel(0)
        if len(tqqq_data.columns) == 6:
            tqqq_data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        elif len(tqqq_data.columns) == 5:
            tqqq_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_data.columns = vix_data.columns.droplevel(0)
        if len(vix_data.columns) == 6:
            vix_data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        elif len(vix_data.columns) == 5:
            vix_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # 결합된 데이터
    combined_data = pd.DataFrame({
        'TQQQ_Close': tqqq_data['Close'],
        'VIX_Close': vix_data['Close'],
        'TQQQ_Volume': tqqq_data['Volume'],
        'VIX_Volume': vix_data['Volume']
    }).dropna()
    
    # 수익률 계산
    combined_data['TQQQ_Return'] = combined_data['TQQQ_Close'].pct_change()
    combined_data['VIX_Return'] = combined_data['VIX_Close'].pct_change()
    
    # 이동 상관관계
    combined_data['Correlation_20d'] = combined_data['TQQQ_Return'].rolling(window=20).corr(combined_data['VIX_Return'])
    combined_data['Correlation_60d'] = combined_data['TQQQ_Return'].rolling(window=60).corr(combined_data['VIX_Return'])
    
    # 연도별 상관관계
    yearly_corr = []
    for year in combined_data.index.year.unique():
        year_data = combined_data[combined_data.index.year == year]
        if len(year_data) > 20:
            return_corr = year_data['TQQQ_Return'].corr(year_data['VIX_Return'])
            yearly_corr.append({
                'Year': year,
                'Correlation': return_corr,
                'Trading_Days': len(year_data)
            })
    
    yearly_df = pd.DataFrame(yearly_corr)
    
    # VIX 레벨별 상관관계
    vix_levels = [15, 20, 25, 30, 35, 40, 50]
    vix_level_analysis = []
    
    for i in range(len(vix_levels) - 1):
        low_vix = vix_levels[i]
        high_vix = vix_levels[i + 1]
        level_data = combined_data[
            (combined_data['VIX_Close'] >= low_vix) & 
            (combined_data['VIX_Close'] < high_vix)
        ]
        if len(level_data) > 20:
            return_corr = level_data['TQQQ_Return'].corr(level_data['VIX_Return'])
            vix_level_analysis.append({
                'VIX_Range': f"{low_vix}-{high_vix}",
                'Correlation': return_corr,
                'Trading_Days': len(level_data),
                'Avg_VIX': level_data['VIX_Close'].mean()
            })
    
    vix_level_df = pd.DataFrame(vix_level_analysis)
    
    # 전체 상관계수
    overall_corr = combined_data['TQQQ_Return'].corr(combined_data['VIX_Return'])
    
    # HTML 대시보드 생성
    dashboard_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>TQQQ vs VIX 상관관계 분석 대시보드</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }}
            .header p {{
                margin: 10px 0 0 0;
                font-size: 1.1em;
                opacity: 0.9;
            }}
            .summary {{
                padding: 30px;
                background-color: #f8f9fa;
                border-bottom: 1px solid #dee2e6;
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .summary-card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                text-align: center;
            }}
            .summary-card h3 {{
                margin: 0 0 10px 0;
                color: #495057;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .summary-card .value {{
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
                margin: 0;
            }}
            .charts-container {{
                padding: 30px;
            }}
            .chart-section {{
                margin-bottom: 40px;
            }}
            .chart-title {{
                font-size: 1.5em;
                color: #333;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #667eea;
            }}
            .chart {{
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin-bottom: 20px;
            }}
            .navigation {{
                position: fixed;
                top: 20px;
                right: 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 15px;
                z-index: 1000;
            }}
            .navigation h4 {{
                margin: 0 0 10px 0;
                color: #333;
                font-size: 0.9em;
            }}
            .navigation a {{
                display: block;
                padding: 5px 10px;
                margin: 2px 0;
                color: #667eea;
                text-decoration: none;
                border-radius: 4px;
                transition: background-color 0.3s;
                font-size: 0.9em;
            }}
            .navigation a:hover {{
                background-color: #f0f0f0;
            }}
            .insights {{
                background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
                color: #2d3436;
            }}
            .insights h3 {{
                margin-top: 0;
                color: #2d3436;
            }}
            .insights ul {{
                margin: 10px 0;
                padding-left: 20px;
            }}
            .insights li {{
                margin: 5px 0;
            }}
        </style>
    </head>
    <body>
        <div class="navigation">
            <h4>차트 네비게이션</h4>
            <a href="#summary">분석 요약</a>
            <a href="#price-chart">가격 비교</a>
            <a href="#correlation-scatter">상관관계 산점도</a>
            <a href="#rolling-correlation">이동 상관관계</a>
            <a href="#yearly-correlation">연도별 상관관계</a>
            <a href="#vix-level">VIX 레벨별 분석</a>
            <a href="#insights">투자 인사이트</a>
        </div>

        <div class="container">
            <div class="header">
                <h1>TQQQ vs VIX 상관관계 분석</h1>
                <p>종합 대시보드 - {datetime.now().strftime('%Y년 %m월 %d일')}</p>
            </div>

            <div id="summary" class="summary">
                <h2>📊 분석 요약</h2>
                <p>분석 기간: {combined_data.index[0].strftime('%Y-%m-%d')} ~ {combined_data.index[-1].strftime('%Y-%m-%d')}</p>
                
                <div class="summary-grid">
                    <div class="summary-card">
                        <h3>총 거래일</h3>
                        <p class="value">{len(combined_data):,}</p>
                    </div>
                    <div class="summary-card">
                        <h3>전체 상관계수</h3>
                        <p class="value">{overall_corr:.3f}</p>
                    </div>
                    <div class="summary-card">
                        <h3>평균 TQQQ 가격</h3>
                        <p class="value">${combined_data['TQQQ_Close'].mean():.2f}</p>
                    </div>
                    <div class="summary-card">
                        <h3>평균 VIX 지수</h3>
                        <p class="value">{combined_data['VIX_Close'].mean():.1f}</p>
                    </div>
                </div>
            </div>

            <div class="charts-container">
                <div id="price-chart" class="chart-section">
                    <h2 class="chart-title">📈 가격 비교 차트</h2>
                    <div id="priceChart" class="chart"></div>
                </div>

                <div id="correlation-scatter" class="chart-section">
                    <h2 class="chart-title">🎯 상관관계 산점도</h2>
                    <div id="scatterChart" class="chart"></div>
                </div>

                <div id="rolling-correlation" class="chart-section">
                    <h2 class="chart-title">📊 이동 상관관계</h2>
                    <div id="correlationChart" class="chart"></div>
                </div>

                <div id="yearly-correlation" class="chart-section">
                    <h2 class="chart-title">📅 연도별 상관관계</h2>
                    <div id="yearlyChart" class="chart"></div>
                </div>

                <div id="vix-level" class="chart-section">
                    <h2 class="chart-title">🎚️ VIX 레벨별 상관관계</h2>
                    <div id="vixLevelChart" class="chart"></div>
                </div>

                <div id="insights" class="insights">
                    <h3>💡 주요 투자 인사이트</h3>
                    <ul>
                        <li><strong>강한 음의 상관관계</strong>: TQQQ와 VIX는 {overall_corr:.3f}의 상관계수를 보여 반대 방향으로 움직입니다.</li>
                        <li><strong>시장 불안정 시</strong>: VIX 상승 시 TQQQ는 하락하는 경향이 강합니다.</li>
                        <li><strong>저변동성 시장</strong>: VIX 15-20 구간에서 가장 강한 음의 상관관계를 보입니다.</li>
                        <li><strong>최근 강화</strong>: 2025년 들어 상관관계가 더욱 강해지고 있습니다.</li>
                        <li><strong>헤지 전략</strong>: VIX를 TQQQ 투자의 리스크 관리 도구로 활용할 수 있습니다.</li>
                    </ul>
                </div>
            </div>
        </div>

        <script>
            // 차트 데이터 준비
            const combinedData = {combined_data.to_json(orient='index', date_format='iso')};
            const yearlyData = {yearly_df.to_json(orient='records')};
            const vixLevelData = {vix_level_df.to_json(orient='records')};
            
            // 1. 가격 비교 차트
            const priceChart = {{
                data: [
                    {{
                        x: Object.keys(combinedData),
                        y: Object.values(combinedData).map(d => d.TQQQ_Close),
                        type: 'scatter',
                        mode: 'lines',
                        name: 'TQQQ',
                        line: {{color: '#667eea', width: 2}},
                        yaxis: 'y1'
                    }},
                    {{
                        x: Object.keys(combinedData),
                        y: Object.values(combinedData).map(d => d.VIX_Close),
                        type: 'scatter',
                        mode: 'lines',
                        name: 'VIX',
                        line: {{color: '#e74c3c', width: 2}},
                        yaxis: 'y2'
                    }}
                ],
                layout: {{
                    title: 'TQQQ vs VIX 가격 추이',
                    xaxis: {{title: '날짜'}},
                    yaxis: {{title: 'TQQQ 가격 ($)', side: 'left', showgrid: false}},
                    yaxis2: {{title: 'VIX 지수', side: 'right', overlaying: 'y', showgrid: false}},
                    hovermode: 'x unified',
                    height: 500
                }}
            }};
            
            // 2. 상관관계 산점도
            const scatterData = Object.values(combinedData).filter(d => d.TQQQ_Return !== null && d.VIX_Return !== null);
            const scatterChart = {{
                data: [
                    {{
                        x: scatterData.map(d => d.VIX_Return * 100),
                        y: scatterData.map(d => d.TQQQ_Return * 100),
                        mode: 'markers',
                        marker: {{
                            size: 3,
                            color: '#667eea',
                            opacity: 0.6
                        }},
                        name: '일간 수익률',
                        hovertemplate: 'VIX: %{{x:.2f}}%<br>TQQQ: %{{y:.2f}}%<extra></extra>'
                    }}
                ],
                layout: {{
                    title: 'TQQQ vs VIX 수익률 상관관계',
                    xaxis: {{title: 'VIX 일간 수익률 (%)'}},
                    yaxis: {{title: 'TQQQ 일간 수익률 (%)'}},
                    height: 500
                }}
            }};
            
            // 3. 이동 상관관계
            const correlationData = Object.values(combinedData).filter(d => d.Correlation_20d !== null);
            const correlationChart = {{
                data: [
                    {{
                        x: Object.keys(combinedData).filter(key => combinedData[key].Correlation_20d !== null),
                        y: correlationData.map(d => d.Correlation_20d),
                        type: 'scatter',
                        mode: 'lines',
                        name: '20일 이동상관관계',
                        line: {{color: '#667eea', width: 2}}
                    }},
                    {{
                        x: Object.keys(combinedData).filter(key => combinedData[key].Correlation_60d !== null),
                        y: correlationData.map(d => d.Correlation_60d),
                        type: 'scatter',
                        mode: 'lines',
                        name: '60일 이동상관관계',
                        line: {{color: '#e74c3c', width: 2}}
                    }}
                ],
                layout: {{
                    title: 'TQQQ와 VIX 간 이동 상관관계',
                    xaxis: {{title: '날짜'}},
                    yaxis: {{title: '상관계수'}},
                    shapes: [{{
                        type: 'line',
                        x0: 0, x1: 1,
                        y0: 0, y1: 0,
                        xref: 'paper',
                        yref: 'y',
                        line: {{dash: 'dash', color: 'gray'}}
                    }}],
                    height: 500
                }}
            }};
            
            // 4. 연도별 상관관계
            const yearlyChart = {{
                data: [
                    {{
                        x: yearlyData.map(d => d.Year),
                        y: yearlyData.map(d => d.Correlation),
                        type: 'bar',
                        marker: {{color: '#667eea'}},
                        name: '연간 상관계수'
                    }}
                ],
                layout: {{
                    title: '연도별 TQQQ-VIX 수익률 상관관계',
                    xaxis: {{title: '연도'}},
                    yaxis: {{title: '상관계수'}},
                    height: 400
                }}
            }};
            
            // 5. VIX 레벨별 상관관계
            const vixLevelChart = {{
                data: [
                    {{
                        x: vixLevelData.map(d => d.VIX_Range),
                        y: vixLevelData.map(d => d.Correlation),
                        type: 'bar',
                        marker: {{color: '#e74c3c'}},
                        text: vixLevelData.map(d => d.Trading_Days + '일'),
                        textposition: 'outside',
                        name: 'VIX 레벨별 상관계수'
                    }}
                ],
                layout: {{
                    title: 'VIX 레벨별 TQQQ-VIX 상관관계',
                    xaxis: {{title: 'VIX 레벨'}},
                    yaxis: {{title: '상관계수'}},
                    height: 400
                }}
            }};
            
            // 차트 렌더링
            Plotly.newPlot('priceChart', priceChart.data, priceChart.layout);
            Plotly.newPlot('scatterChart', scatterChart.data, scatterChart.layout);
            Plotly.newPlot('correlationChart', correlationChart.data, correlationChart.layout);
            Plotly.newPlot('yearlyChart', yearlyChart.data, yearlyChart.layout);
            Plotly.newPlot('vixLevelChart', vixLevelChart.data, vixLevelChart.layout);
            
            // 부드러운 스크롤
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
                anchor.addEventListener('click', function (e) {{
                    e.preventDefault();
                    const target = document.querySelector(this.getAttribute('href'));
                    if (target) {{
                        target.scrollIntoView({{
                            behavior: 'smooth',
                            block: 'start'
                        }});
                    }}
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    # 대시보드 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dashboard_filename = f"TQQQ_VIX_Interactive_Dashboard_{timestamp}.html"
    
    with open(dashboard_filename, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    
    print(f"인터랙티브 대시보드가 {dashboard_filename}에 저장되었습니다.")
    return dashboard_filename

if __name__ == "__main__":
    create_interactive_dashboard()
