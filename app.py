import streamlit as st
import pandas as pd
import pickle  # Using built-in pickle
import datetime
import matplotlib.pyplot as plt
import os
import numpy as np
from API_wrapper import PySimFin
from trading_strategy import BuyAndHoldStrategy, BuyAndSellStrategy, backtest_strategy, plot_backtest_results

st.markdown(
    """
    <style>
    .stApp {
        background-color: #fdfaf0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

try:
    import xgboost
except ImportError:
    st.error("XGBoost is not installed. Please check if the deployment environment has all required dependencies.")

# Initialize API wrapper
api = PySimFin()  # Will automatically use API key from secrets

# Creation functions
def get_latest_financial_metrics(fin_data: pd.DataFrame) -> dict:
            """Extract the latest financial metrics from financial statements."""
            metrics = {}

            try:
                # Get the most recent data for each statement type
                pl_data = fin_data[fin_data['Statement_Type'] == 'PL'].sort_values('Report Date').iloc[-1]
                bs_data = fin_data[fin_data['Statement_Type'] == 'BS'].sort_values('Report Date').iloc[-1]
                derived_data = fin_data[fin_data['Statement_Type'] == 'DERIVED'].sort_values('Report Date').iloc[-1]

                # Income Statement Metrics
                metrics['Revenue'] = pl_data.get('Revenue')
                metrics['Net Income'] = pl_data.get('Net Income')
                metrics['Operating Income'] = pl_data.get('Operating Income (Loss)')

                # Balance Sheet Metrics
                metrics['Total Assets'] = bs_data.get('Total Assets')
                metrics['Total Liabilities'] = bs_data.get('Total Liabilities')
                metrics['Total Equity'] = bs_data.get('Total Equity')

                # Key Ratios from Derived Metrics
                metrics['Gross Profit Margin'] = derived_data.get('Gross Profit Margin')
                metrics['Operating Margin'] = derived_data.get('Operating Margin')
                metrics['Return on Equity'] = derived_data.get('Return on Equity')
                metrics['Current Ratio'] = derived_data.get('Current Ratio')

            except Exception as e:
                st.warning(f"Some financial metrics could not be loaded: {str(e)}")

            return metrics

# Number formatting for get_financial_statement function
def format_number(value):
    """Format large numbers to millions/billions with 2 decimal places."""
    if value is None or pd.isna(value):
        return "N/A"

    abs_value = abs(value)
    if abs_value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif abs_value >= 1e6:
        return f"${value/1e6:.2f}M"
    else:
        return f"${value:.2f}"


# Cache Data for performance
@st.cache_data
def load_historical_data():
    """Load processed historical data."""
    try:
        data_path = "data/merged_features.csv"  # Updated path to lowercase
        if not os.path.exists(data_path):
            st.error(f"Data file not found at {data_path}. Please check if the file exists.")
            return None
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Error loading historical data: {str(e)}")
        return None

@st.cache_resource
def load_classification_model():
    """Load the pre-trained classification model."""
    try:
        model_path = "best_clf_model.pkl"
        if not os.path.exists(model_path):
            st.error(f"Classification model file not found at {model_path}")
            return None
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading classification model: {str(e)}")
        return None

@st.cache_resource
def load_regression_model():
    """Load the pre-trained regression model."""
    try:
        model_path = "best_reg_model.pkl"
        if not os.path.exists(model_path):
            st.error(f"Regression model file not found at {model_path}")
            return None
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading regression model: {str(e)}")
        return None

@st.cache_resource
def init_api_wrapper(api_key: str):
    """Initialize the API wrapper."""
    return PySimFin(api_key)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home","Overview", "Go Live","Trading Strategy"])

# Define tickers
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']  # Updated tickers

# Home Page
if page == "Home":
    st.title("📊 Automated Daily Trading System")
    st.title("🏠 Home Page")

    st.markdown(
        """
        Welcome to the Automated Daily Trading System web app,

        We are **Team 9**, a team of finance and data analytics professionals committed to leveraging technology for smarter trading decisions.

        Today, we are excited to present our **Automated Daily Trading System**, designed to **forecast market movements** and **enhance decision-making** in dynamic financial markets.
        """
    )

    tabs = st.tabs(["👥 Meet the Team", "🏢 Meet the Companies"])

    with tabs[0]:
        st.header("👥 Meet the Team")
        st.image('./Team2.png', caption=' ', width=700)

    with tabs[1]:
        st.header("🏢 Meet the Companies")
        st.image('./Companies3.png', caption=' ', width=700)



# Overview
if page == "Overview":  # Correct indentation
    st.title("🔍 Overview")
    tab1, tab2, tab3 = st.tabs([ "🏗️ Technical Architecture", "📊 Data Analytics Module", "🌐 Web Application"])

    # Pestaña 3: Technical Architecture
    with tab1:
        st.subheader("🏗️ Technical Architecture")

        st.markdown("### 1️⃣ Data Processing")
        st.markdown("""
        - **Pandas** for data manipulation
        - **Historical data** from SimFin bulk download
        - **Real-time data** via SimFin API
        """)

        st.markdown("### 2️⃣ Machine Learning")
        st.markdown("""
        - **Classification model** for price movement
        - **Regression model** for price prediction
        - **Feature engineering** from financial metrics
        """)

        st.markdown("### 3️⃣ Web Interface")
        st.markdown("""
        - **Streamlit** for frontend
        - **Interactive visualizations**
        - **Real-time data updates**
        """)
    
    # Pestaña 2: Data Analytics Module
    with tab2:
        st.subheader("📊 Data Analytics Module")
        st.markdown("""
        - **ETL Pipeline**: Processes historical financial data from **SimFin**
        - **ML Models**:
          - Classification model to predict **price movement (rise/fall)**
          - Regression model to predict **next-day price**
        - **Trading Strategies**:
          - **Buy and Hold**: Purchases shares when **rise predicted**, holds until **profit target**
          - **Buy and Sell**: Dynamic trading based on **daily predictions**
        """)

    # Pestaña 3: Web Application
    with tab3:
        st.subheader("🌐 Web Application")
        st.markdown("""
        - **PySimFin API Wrapper**:
            - 📊 **Retrieve Share Prices**: Fetch daily share prices for a given company within a date range.
            - 📜 **Retrieve Financial Statements**: Access compact financial statement data (balance sheets, income statements, cash flow statements).
            - 📝 **Logging Configuration**:  
                - 🕒 Timestamped API logs  
                - 📌 Module name tracking  
                - ⚠️ Log level  
                - 💬 Messages  
        """)
        st.markdown("""
        - **Multi-page Streamlit interface**:
          - 📌 Home page with **project overview**
          - 📖 Overview page explaining **methodology**
          - 📈 Live trading **dashboard**
          - 🔄 Trading strategy **backtesting**
        """)

    # Final message
    st.success("🚀 Ready to explore the automated trading system!")

# Go Live Page
# Initialize API wrapper
api = init_api_wrapper("460e1696-925e-41ba-b556-fe23ce2cefd4")
if page == "Go Live":  # Correct indentation
    st.title("🚀 Go Live – Trading Dashboard")

    # Sidebar: Ticker Selection
    selected_ticker = st.sidebar.selectbox("📈 Select Stock Ticker", tickers, help="Choose a stock to analyze")

    # Load Data and Models
    historical_data = load_historical_data()
    clf_model = load_classification_model()
    reg_model = load_regression_model()

    # Filter historical data
    ticker_data = historical_data[historical_data['Ticker'] == selected_ticker].sort_values("Date")

    # Create Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📜 Historical Data", "📡 Predictions", "📊 Live Data", "📑 Financial Statements"])

    with tab1:  # Historical Data
        st.subheader(f"📜 Historical Data for {selected_ticker}")
        if not ticker_data.empty:
            st.dataframe(ticker_data.tail(10))

            # Plot Stock Price Trend
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(ticker_data["Date"], ticker_data["Close"], label="Closing Price", linestyle='-')
            ax.set_xlabel("Date")
            ax.set_ylabel("Stock Price ($)")
            ax.set_title(f"{selected_ticker} Closing Price Trend")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("No historical data available.")

    with tab2:  # Live data
        st.subheader("📊 Live Stock Data")

        today = datetime.date.today()
        default_start = today - datetime.timedelta(days=60)
        date_range = st.date_input("📆 Select live data range", [default_start, today])

        # Validate date input
        if len(date_range) == 2:
            start_date, end_date = date_range[0].strftime("%Y-%m-%d"), date_range[1].strftime("%Y-%m-%d")
        else:
            st.warning("Please select a valid date range.")
            start_date, end_date = default_start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

        if st.button("🔄 Refresh Live Data"):
            with st.spinner("Fetching live market data..."):
                live_data = api.get_share_prices(selected_ticker, start_date, end_date)

            if live_data is not None and not live_data.empty:
                st.success(f"✅ Live data retrieved for {selected_ticker} ({start_date} to {end_date})")
                st.dataframe(live_data.tail(10))

                # Plot Live Price Trend
                if "Date" in live_data.columns:
                    live_data["Date"] = pd.to_datetime(live_data["Date"])
                    st.line_chart(live_data.set_index("Date")["Last Closing Price"])
            else:
                st.error("❌ No live data available.")



    with tab3:  # Model Predictions
        st.subheader("📡 Model Predictions")

        # Check if live data exists and is not empty
        if 'live_data' in locals() and live_data is not None and not live_data.empty:
            try:
                # Get the most recent record from live data
                latest_record = live_data.iloc[-1]

                # Create features from live data
                feature_dict = {
                    'Close': latest_record['Last Closing Price'],
                    'Volume': latest_record['Volume'],
                    'ema_20': latest_record['Last Closing Price'],  # Using closing price as proxy for EMA
                    'day_of_week': pd.to_datetime(latest_record['Date']).dayofweek
                }
                latest_features_df = pd.DataFrame([feature_dict])

                # Classification Prediction (Rise or Fall)
                if clf_model is not None:
                    try:
                        clf_prediction = clf_model.predict(latest_features_df)[0]
                        prediction_text = "⬆ Rise" if clf_prediction == 1 else "⬇ Fall"
                        # Convert numpy float32 to Python float
                        confidence = float(clf_model.predict_proba(latest_features_df)[0][clf_prediction] * 100)  # Confidence score

                        # Display prediction with color
                        st.markdown("### Next-Day Price Movement Prediction")
                        if prediction_text == "⬆ Rise":
                            st.markdown(f"""<span style='font-size:24px; color:green;'>{prediction_text}</span>""", unsafe_allow_html=True)
                        else:
                            st.markdown(f"""<span style='font-size:24px; color:red;'>{prediction_text}</span>""", unsafe_allow_html=True)

                        # Display confidence
                        st.markdown(f"*📈 Model Confidence: {confidence:.2f}%*")
                        st.progress(confidence / 100)  # Show confidence as a progress bar
                    except Exception as e:
                        st.error(f"Error making classification prediction: {str(e)}")
                else:
                    st.warning("Classification model could not be loaded.")

                # Regression Prediction (Exact Price)
                if reg_model is not None:
                    try:
                        # Convert numpy float32 to Python float
                        reg_prediction = float(reg_model.predict(latest_features_df)[0])
                        current_price = latest_record['Last Closing Price']
                        price_change = ((reg_prediction - current_price) / current_price) * 100

                        # Display current and predicted prices
                        st.markdown("### Price Predictions")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                label="Current Price",
                                value=f"${current_price:.2f}"
                            )
                        with col2:
                            st.metric(
                                label="Predicted Next-Day Price",
                                value=f"${reg_prediction:.2f}",
                                delta=f"{price_change:+.2f}%"
                            )
                    except Exception as e:
                        st.error(f"Error making regression prediction: {str(e)}")
                else:
                    st.warning("Regression model could not be loaded.")

            except Exception as e:
                st.error(f"Error preparing prediction data: {str(e)}")
        else:
            st.warning("Please fetch live data first using the 'Refresh Live Data' button above to see predictions.")


    with tab4:  # Financial Statements
        st.subheader(f"📑 {selected_ticker}Financial Statements for 2024")
        if st.button("📜 Fetch Financial Statements"):
            with st.spinner('Fetching financial data...'):
                # Fetch financial statement data for 2023 instead of 2024
                fin_data = api.get_financial_statement(
                    ticker=selected_ticker,
                    start="2023-01-01",
                    end="2023-12-31",
                    statement_type='all'
                )

                if fin_data is not None and not fin_data.empty:
                    metrics = get_latest_financial_metrics(fin_data)

                    # Create three columns for better organization
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("##### Income Statement Metrics")
                        st.metric("Revenue", format_number(metrics.get('Revenue')))
                        st.metric("Net Income", format_number(metrics.get('Net Income')))
                        st.metric("Operating Income", format_number(metrics.get('Operating Income')))

                    with col2:
                        st.markdown("##### Balance Sheet Metrics")
                        st.metric("Total Assets", format_number(metrics.get('Total Assets')))
                        st.metric("Total Liabilities", format_number(metrics.get('Total Liabilities')))
                        st.metric("Total Equity", format_number(metrics.get('Total Equity')))

                    with col3:
                        st.markdown("##### Key Ratios")
                        st.metric("Gross Profit Margin", f"{metrics.get('Gross Profit Margin', 0)*100:.2f}%" if metrics.get('Gross Profit Margin') is not None else "N/A")
                        st.metric("Operating Margin", f"{metrics.get('Operating Margin', 0)*100:.2f}%" if metrics.get('Operating Margin') is not None else "N/A")
                        st.metric("Return on Equity", f"{metrics.get('Return on Equity', 0)*100:.2f}%" if metrics.get('Return on Equity') is not None else "N/A")
                        st.metric("Current Ratio", f"{metrics.get('Current Ratio', 0):.2f}" if metrics.get('Current Ratio') is not None else "N/A")

                    # Show detailed financial statements in expandable sections
                    with st.expander("View Detailed Financial Statements"):
                        statement_type = st.selectbox(
                            "Select Statement Type",
                            ["Income Statement", "Balance Sheet", "Cash Flow", "Key Ratios"]
                        )

                        # Filter and display the selected statement
                        statement_map = {
                            "Income Statement": "PL",
                            "Balance Sheet": "BS",
                            "Cash Flow": "CF",
                            "Key Ratios": "DERIVED"
                        }

                        filtered_data = fin_data[fin_data['Statement_Type'] == statement_map[statement_type]]
                        if not filtered_data.empty:
                            # Sort by date and select relevant columns
                            display_cols = [col for col in filtered_data.columns if col not in ['Company_ID', 'Currency']]
                            st.dataframe(
                                filtered_data[display_cols].sort_values('Report Date', ascending=False),
                                use_container_width=True
                            )
                        else:
                            st.info(f"No data available for {statement_type}")
                else:
                    st.warning(
                        "No financial data available for this company. "
                        "This could be because the data hasn't been reported yet."
                    )


elif page == "Trading Strategy":
    st.title("Trading Strategy Backtesting")

    # Load data and models
    historical_data = load_historical_data()
    clf_model = load_classification_model()

    # Sidebar inputs
    st.sidebar.subheader("Backtest Parameters")
    selected_ticker = st.sidebar.selectbox("Select Ticker", sorted(historical_data['Ticker'].unique()))

    # Filter data for selected ticker
    ticker_data = historical_data[historical_data['Ticker'] == selected_ticker].sort_values('Date')

    if ticker_data.empty:
        st.error(f"No historical data available for {selected_ticker}")
    else:
        # Date range selection
        min_date = ticker_data['Date'].min()
        max_date = ticker_data['Date'].max()
        start_date = st.sidebar.date_input(
            "Start Date",
            min_date,
            min_value=min_date,
            max_value=max_date
        )
        end_date = st.sidebar.date_input(
            "End Date",
            max_date,
            min_value=start_date,
            max_value=max_date
        )

        # Strategy selection and parameters
        strategy_type = st.sidebar.selectbox(
            "Select Strategy",
            ["Buy and Hold", "Buy and Sell"]
        )

        initial_capital = st.sidebar.number_input(
            "Initial Capital ($)",
            min_value=1000,
            value=10000,
            step=1000
        )

        if strategy_type == "Buy and Hold":
            profit_target = st.sidebar.slider(
                "Profit Target (%)",
                min_value=1,
                max_value=50,
                value=10
            ) / 100
            strategy = BuyAndHoldStrategy(initial_capital, profit_target)
        else:
            strategy = BuyAndSellStrategy(initial_capital)

        # Generate predictions for the entire dataset
        try:
            feature_cols = ['Close', 'Volume', 'ema_20', 'day_of_week']
            predictions = clf_model.predict(ticker_data[feature_cols])

            # Run backtest
            if st.button("Run Backtest"):
                with st.spinner("Running backtest..."):
                    try:
                        # Filter data for the selected date range
                        mask = (ticker_data['Date'] >= pd.Timestamp(start_date)) & (ticker_data['Date'] <= pd.Timestamp(end_date))
                        backtest_data = ticker_data[mask].copy()
                        backtest_predictions = predictions[mask]

                        if len(backtest_data) > 0:
                            history_df, metrics = backtest_strategy(
                                strategy,
                                backtest_data,
                                backtest_predictions,
                                start_date.strftime("%Y-%m-%d"),
                                end_date.strftime("%Y-%m-%d")
                            )

                            # Display metrics
                            col1, col2 = st.columns(2)

                            with col1:
                                st.subheader("Performance Metrics")
                                st.metric("Initial Portfolio Value", f"${metrics['Initial Portfolio Value']:,.2f}")
                                st.metric("Final Portfolio Value", f"${metrics['Final Portfolio Value']:,.2f}")
                                st.metric("Total Return", f"{metrics['Total Return %']:.2f}%")
                                st.metric("Number of Trades", metrics['Number of Trades'])

                            with col2:
                                st.subheader("Risk Metrics")
                                st.metric("Average Position Size", f"{metrics['Average Position Size']:.2f} shares")
                                st.metric("Max Drawdown", f"{metrics['Max Drawdown %']:.2f}%")
                                st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")

                            # Plot results
                            st.plotly_chart(plot_backtest_results(history_df, selected_ticker))

                            # Display trade history
                            st.subheader("Trade History")
                            trade_history = history_df[history_df['action'].isin(['BUY', 'SELL'])].copy()
                            trade_history['date'] = pd.to_datetime(trade_history['date']).dt.strftime('%Y-%m-%d')
                            trade_history['price'] = trade_history['price'].map('${:,.2f}'.format)
                            trade_history['portfolio_value'] = trade_history['portfolio_value'].map('${:,.2f}'.format)
                            st.dataframe(
                                trade_history[['date', 'action', 'price', 'shares', 'portfolio_value']],
                                use_container_width=True
                            )
                        else:
                            st.warning("No data available for the selected date range.")
                    except Exception as e:
                        st.error(f"Error during backtesting: {str(e)}")
        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")
