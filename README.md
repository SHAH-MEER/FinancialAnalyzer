# Financial Portfolio Analyzer

A comprehensive financial portfolio analysis tool built with Python and Streamlit.

## Features

- 📊 Portfolio Dashboard with real-time analytics
- 💼 Portfolio Management with stock tracking
- 📈 Technical Analysis with multiple indicators
- 🎯 Advanced Risk Assessment
- 📰 News & Sentiment Analysis
- 🤖 AI-Driven Portfolio Insights

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FinancialAnalyzer.git
cd FinancialAnalyzer
```

2. Create and activate virtual environment:
```bash
python -m venv fa
source fa/bin/activate  # On Windows: fa\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
- Create a `.env` file in the project root
- Add your API keys:
```
NEWS_API_KEY=your_news_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
```

## Usage

Run the application:
```bash
streamlit run app.py
```

## Directory Structure

```
FinancialAnalyzer/
├── app.py                 # Main application file
├── requirements.txt       # Project dependencies
├── components/            # UI components
│   └── news_card.py       # News article card component
├── docs/                  # Documentation
│   ├── core_features.py
│   ├── page_guides.py
│   └── ai_insights_guide.py
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── ai_insights.py
│   ├── alpha_vantage.py
│   ├── anomaly_detection.py
│   ├── cache_manager.py
│   ├── chart_themes.py
│   ├── dashboard_config.py
│   ├── data_import.py
│   ├── database.py
│   ├── export_utils.py
│   ├── factor_analysis.py
│   ├── news_api.py
│   ├── pattern_recognition.py
│   ├── portfolio_optimizer.py
│   ├── report_generator.py
│   ├── risk_analysis.py
│   ├── sentiment_analyzer.py
│   ├── theme.py
│   └── time_series.py
└── api/                   # API integrations
    └── gateway.py
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

MIT License - see LICENSE file for details