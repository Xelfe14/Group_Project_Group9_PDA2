# Automated Daily Trading System 🚀

## Project Overview
This project implements an automated daily trading system that combines machine learning predictions with real-time market data to provide trading insights. The system analyzes historical data from five major US companies (AAPL, MSFT, GOOGL, AMZN, FB) and provides daily trading recommendations through an interactive web interface.

## Team Members
- Team 9
- Master's in Data Analytics
- Academic Year 2023-2024

## Features 🌟

### Data Analytics Module
- **ETL Pipeline**: Processes historical financial data from SimFin
- **ML Models**:
  - Classification model to predict price movement (rise/fall)
  - Regression model to predict next-day price
- **Trading Strategies**:
  - Buy and Hold: Purchases shares when rise predicted, holds until profit target
  - Buy and Sell: Dynamic trading based on daily predictions

### Web Application
- **Multi-page Streamlit interface**:
  - Home page with project overview
  - Overview page explaining methodology
  - Live trading dashboard
  - Trading strategy backtesting

## Technical Architecture 🏗️

### Core Components
1. **Data Processing**:
   - Pandas for data manipulation
   - Historical data from SimFin bulk download
   - Real-time data via SimFin API

2. **Machine Learning**:
   - Classification model for price movement
   - Regression model for price prediction
   - Feature engineering from financial metrics

3. **Web Interface**:
   - Streamlit for frontend
   - Interactive visualizations
   - Real-time data updates

## Installation Guide 🔧

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up SimFin API key:
   - Create a free account at [SimFin](https://www.simfin.com/)
   - Get your API key from the dashboard
   - Add to your environment variables:
     ```bash
     export SIMFIN_API_KEY="your-api-key"
     ```

## Usage 📱

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Navigate to the web interface:
   - Home: Project overview and team information
   - Overview: System methodology explanation
   - Go Live: Real-time trading dashboard
   - Trading Strategy: Backtest trading strategies

## Project Structure 📁

```
Streamlit_app2/
├── app.py                 # Main Streamlit application
├── API_wrapper.py         # SimFin API integration
├── trading_strategy.py    # Trading strategy implementations
├── requirements.txt       # Project dependencies
├── Data/                  # Historical data storage
└── README.md             # Project documentation
```

## Dependencies 📚
- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Requests

## Limitations ⚠️
- Free SimFin API tier limited to 2 requests per second
- Historical data limited to 5 years
- Limited to 5 major US companies
- Predictions should not be used as sole financial advice

## Future Improvements 🔮
- Add more sophisticated trading strategies
- Implement additional technical indicators
- Expand company coverage
- Add portfolio optimization features
- Enhance UI/UX with more interactive features

## Disclaimer ⚖️
This project is for educational purposes only. The trading signals and predictions should not be used as financial advice. Always conduct your own research and consult with financial professionals before making investment decisions.

## License 📄
MIT License - see LICENSE file for details
