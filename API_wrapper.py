import requests
import logging
import pandas as pd
from typing import Optional, Dict, List, Any
import time
import streamlit as st

class PySimFin:
    def __init__(self, api_key: str = None, base_url: str = "https://backend.simfin.com/api/v3"):
        # Use provided API key or get from Streamlit secrets
        self.api_key = api_key or st.secrets["SIMFIN_API_KEY"]
        self.base_url = base_url

        # Configure logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)  # Set to INFO for more detailed logs

        self.logger.info("PySimFin initialized with base_url=%s", self.base_url)

    def get_share_prices(self, ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """
        Retrieve share prices for the specified ticker between start and end dates.
        """
        endpoint = f"{self.base_url}/companies/prices/compact"
        headers = {
            "accept": "application/json",
            "Authorization": self.api_key
        }
        # Build the query parameters
        params = {
            "ticker": ticker,
            "start": start,
            "end": end,
            "api-key": self.api_key
        }
        self.logger.info("Requesting share prices for %s from %s to %s", ticker, start, end)

        # Lets get the response from the API endpoint
        try:
            response = requests.get(endpoint, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                self.logger.warning(f"No data returned for {ticker} in the specified date range.")
                return None
            columns = data[0]['columns']
            rows = data[0]['data']
            df = pd.DataFrame(rows, columns=columns)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            self.logger.info(f"Successfully retrieved {ticker} rows of share prices for {len(df)}")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching share prices: {e}")
            return None

    def get_financial_statement(self, ticker: str, start: str, end: str, statement_type: str = "all") -> Optional[pd.DataFrame]:
        """
        Retrieve financial statement data for the specified ticker between start and end dates.
        """
        # Map statement types to their API identifiers
        statement_map = {
            "all": "PL,BS,CF,DERIVED",
            "pl": "PL",
            "bs": "BS",
            "cf": "CF",
            "derived": "DERIVED"
        }

        if statement_type.lower() not in statement_map:
            self.logger.error("Invalid statement type. Must be one of: %s", list(statement_map.keys()))
            return None

        # Construct the endpoint URL with query parameters
        endpoint = f"{self.base_url}/companies/statements/compact"

        # Build the query parameters
        params = {
            "ticker": ticker,
            "statements": statement_map[statement_type.lower()],
            "start": start,
            "end": end,
            "api-key": self.api_key,
            "ratios": "true",  # Include financial ratios
            "ttm": "true"     # Include trailing twelve months data
        }

        headers = {
            "accept": "application/json",
            "Authorization": self.api_key
        }

        self.logger.info("Requesting %s statements for %s from %s to %s",
                        statement_type, ticker, start, end)

        # Lets get the response from the API endpoint
        try:
            response = requests.get(endpoint, headers=headers, params=params, timeout=30)
            # Log the response status and content for debugging
            self.logger.debug("Response Status: %s", response.status_code)
            self.logger.debug("Response Content: %s", response.text[:500])  # First 500 char
            response.raise_for_status()
            data = response.json()

            if not data or not isinstance(data, list) or len(data) == 0:
                self.logger.warning("No financial statement data returned for %s", ticker)
                return None

            # Extract company info and statements
            company_data = data[0]
            statements = company_data.get('statements', [])

            if not statements:
                self.logger.warning("No statements found in the response for %s", ticker)
                return None

            # Process based on statement type
            if statement_type.lower() == 'all':
                # Combine all statements into one DataFrame
                all_data = []
                for stmt in statements:
                    stmt_type = stmt['statement']
                    columns = stmt.get('columns', [])
                    rows = stmt.get('data', [])
                    if columns and rows:
                        df = pd.DataFrame(rows, columns=columns)
                        df['Statement_Type'] = stmt_type
                        all_data.append(df)

                if not all_data:
                    self.logger.warning("No valid statement data found in the response")
                    return None

                final_df = pd.concat(all_data, axis=0, ignore_index=True)

            # Add company information
            final_df['Company_Name'] = company_data.get('name')
            final_df['Company_ID'] = company_data.get('id')
            final_df['Currency'] = company_data.get('currency')

            # Convert date columns
            date_columns = ['Report Date', 'Publish Date']
            for col in date_columns:
                if col in final_df.columns:
                    final_df[col] = pd.to_datetime(final_df[col])

            # Convert numeric columns and handle percentages
            numeric_cols = final_df.select_dtypes(include=['object']).columns
            for col in numeric_cols:
                try:
                    final_df[col] = pd.to_numeric(final_df[col], errors='ignore')
                    # Convert percentage values to decimals if they're in percentage format
                    if col.lower().endswith(('margin', 'ratio', 'return', 'growth')):
                        mask = final_df[col].notna()
                        final_df.loc[mask, col] = final_df.loc[mask, col].astype(float) / 100
                except:
                    pass

            self.logger.info("Successfully retrieved %d rows of financial data for %s",
                           len(final_df), ticker)
            return final_df


        except Exception as e:
            self.logger.error("Error fetching financial statements: %s", e)
            return None

if __name__ == '__main__':
   pass
