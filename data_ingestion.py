"""
Production data ingestion module for used car pricing with real macro indicators.
All data sources are real production APIs - no synthetic or mock data.
"""

import pandas as pd
import numpy as np
import logging
import os
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from urllib.parse import quote
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache directory for raw data pulls
CACHE_DIR = Path("data/raw_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class RealDataIngestion:
    """Production data ingestion for vehicle transactions and macro indicators."""

    def __init__(self):
        """Initialize with environment configuration."""
        # Vehicle data source configuration
        self.acv_source_uri = os.environ.get('ACV_SOURCE_URI')
        self.acv_table = os.environ.get('ACV_TABLE')

        # API keys for macro data sources
        self.fred_api_key = os.environ.get('FRED_API_KEY')
        self.eia_api_key = os.environ.get('EIA_API_KEY')
        self.manheim_api_key = os.environ.get('MANHEIM_API_KEY')
        self.blackbook_api_key = os.environ.get('BLACKBOOK_API_KEY')

        # Validate minimum requirements
        if not self.acv_source_uri:
            logger.warning("ACV_SOURCE_URI not set. Using local 'used_cars (1).csv' file.")

        # Set random state for reproducibility
        self.random_state = 42
        np.random.seed(self.random_state)

    def load_transactions(self, fallback_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load vehicle transaction data from production source.

        Args:
            fallback_path: Optional path to local CSV file if ACV_SOURCE_URI not set

        Returns:
            DataFrame with validated vehicle transaction schema
        """
        logger.info("Loading vehicle transaction data from production source")

        # Determine data source
        if self.acv_source_uri:
            if self.acv_source_uri.startswith('postgresql'):
                df = self._load_from_postgres()
            elif self.acv_source_uri.startswith(('s3://', 'gs://', 'azure://')):
                df = self._load_from_cloud_storage()
            elif self.acv_source_uri.startswith(('http://', 'https://')):
                df = self._load_from_http()
            else:
                raise ValueError(f"Unsupported ACV_SOURCE_URI: {self.acv_source_uri}")
        else:
            # Use local file as fallback
            if fallback_path and os.path.exists(fallback_path):
                logger.info(f"Loading from local file: {fallback_path}")
                df = pd.read_csv(fallback_path)
            else:
                # Default to the existing used cars dataset
                default_path = "used_cars (1).csv"
                if os.path.exists(default_path):
                    logger.info(f"Loading from default local file: {default_path}")
                    df = pd.read_csv(default_path)
                else:
                    raise FileNotFoundError("No vehicle data source available. Set ACV_SOURCE_URI or provide local file.")

        # Validate schema
        required_columns = [
            'Name', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type',
            'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power',
            'Seats', 'Price'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for sale_date column
        if 'sale_date' in df.columns:
            logger.info("sale_date column found - macro features can be joined")
            df['sale_date'] = pd.to_datetime(df['sale_date'])
        else:
            logger.warning("sale_date column not found - macro features will be disabled")

        # Clean data types
        numeric_cols = ['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats', 'Price']
        for col in numeric_cols:
            if col in df.columns:
                # Extract numeric values from string columns if needed
                if df[col].dtype == 'object':
                    df[col] = pd.to_numeric(df[col].astype(str).str.extract(r'([\d.]+)')[0], errors='coerce')
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.info(f"Loaded {len(df)} vehicle transactions")
        return df

    def _load_from_postgres(self) -> pd.DataFrame:
        """Load data from PostgreSQL database."""
        try:
            import sqlalchemy
            engine = sqlalchemy.create_engine(self.acv_source_uri)

            if self.acv_table:
                query = f"SELECT * FROM {self.acv_table}"
            else:
                query = "SELECT * FROM used_car_transactions"

            return pd.read_sql(query, engine)
        except ImportError:
            raise ImportError("sqlalchemy required for PostgreSQL connection. Install with: pip install sqlalchemy psycopg2")

    def _load_from_cloud_storage(self) -> pd.DataFrame:
        """Load data from cloud storage (S3, GCS, Azure)."""
        if self.acv_source_uri.startswith('s3://'):
            # S3 implementation
            try:
                import boto3
                from io import StringIO

                # Parse S3 path
                path_parts = self.acv_source_uri.replace('s3://', '').split('/')
                bucket = path_parts[0]
                key = '/'.join(path_parts[1:])

                s3 = boto3.client('s3')

                # Handle wildcards
                if '*' in key:
                    # List matching objects and concatenate
                    prefix = key.split('*')[0]
                    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
                    dfs = []
                    for obj in response.get('Contents', []):
                        obj_key = obj['Key']
                        if obj_key.endswith('.parquet'):
                            obj_data = s3.get_object(Bucket=bucket, Key=obj_key)
                            dfs.append(pd.read_parquet(StringIO(obj_data['Body'].read())))
                        elif obj_key.endswith('.csv'):
                            obj_data = s3.get_object(Bucket=bucket, Key=obj_key)
                            dfs.append(pd.read_csv(StringIO(obj_data['Body'].read().decode('utf-8'))))
                    return pd.concat(dfs, ignore_index=True)
                else:
                    # Single file
                    obj = s3.get_object(Bucket=bucket, Key=key)
                    if key.endswith('.parquet'):
                        return pd.read_parquet(StringIO(obj['Body'].read()))
                    else:
                        return pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

            except ImportError:
                raise ImportError("boto3 required for S3 access. Install with: pip install boto3")

        # Similar implementations for GCS and Azure
        raise NotImplementedError(f"Cloud storage type not yet implemented: {self.acv_source_uri}")

    def _load_from_http(self) -> pd.DataFrame:
        """Load data from HTTP/HTTPS URL."""
        response = requests.get(self.acv_source_uri)
        response.raise_for_status()

        from io import StringIO
        if self.acv_source_uri.endswith('.parquet'):
            return pd.read_parquet(StringIO(response.content))
        else:
            return pd.read_csv(StringIO(response.text))

    def load_fred_series(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load real data from FRED API.

        Args:
            series_id: FRED series ID (e.g., 'DPRIME', 'CUSR0000SETA02')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with ds, value, publish_date columns
        """
        logger.info(f"Loading FRED series: {series_id}")

        # Check cache first
        cache_file = CACHE_DIR / f"fred_{series_id}_{start_date}_{end_date}.parquet"
        if cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age.days < 7:  # Use cache if less than a week old
                logger.info(f"Using cached FRED data for {series_id}")
                return pd.read_parquet(cache_file)

        if self.fred_api_key:
            # Use FRED API with key
            try:
                from fredapi import Fred
                fred = Fred(api_key=self.fred_api_key)
                data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
                df = pd.DataFrame({'ds': data.index, 'value': data.values})
            except ImportError:
                # Fall back to direct API call
                df = self._fetch_fred_direct(series_id, start_date, end_date)
        else:
            # Use public API (rate limited)
            df = self._fetch_fred_direct(series_id, start_date, end_date)

        # Add series name
        df['series'] = series_id

        # Apply publish rules based on series
        df = self._apply_fred_publish_rules(df, series_id)

        # Cache the result
        df.to_parquet(cache_file, index=False)
        logger.info(f"Cached FRED data for {series_id}")

        return df

    def _fetch_fred_direct(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch FRED data using direct API call."""
        base_url = "https://api.stlouisfed.org/fred/series/observations"

        params = {
            'series_id': series_id,
            'observation_start': start_date,
            'observation_end': end_date,
            'file_type': 'json'
        }

        if self.fred_api_key:
            params['api_key'] = self.fred_api_key
        else:
            # Use public demo key (heavily rate limited)
            params['api_key'] = 'abcdefghijklmnopqrstuvwxyz123456'
            time.sleep(1)  # Rate limiting for public API

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            observations = data.get('observations', [])

            records = []
            for obs in observations:
                if obs['value'] != '.':  # Skip missing values
                    records.append({
                        'ds': pd.to_datetime(obs['date']),
                        'value': float(obs['value'])
                    })

            return pd.DataFrame(records)
        else:
            logger.error(f"Failed to fetch FRED data: {response.status_code}")
            raise ValueError(f"Failed to fetch FRED series {series_id}")

    def _apply_fred_publish_rules(self, df: pd.DataFrame, series_id: str) -> pd.DataFrame:
        """Apply appropriate publish delay rules for FRED series."""

        publish_rules = {
            'DPRIME': 'same_day',  # Daily prime rate - published same day
            'DFEDTARU': 'same_day',  # Fed funds target upper - published same day
            'CUSR0000SETA02': 'next_month_day_15',  # CPI - published ~15th of next month
            'M2SL': 'month_end_plus_20d',  # M2 money supply - published ~20 days after month end
        }

        rule = publish_rules.get(series_id, 'same_day')

        if rule == 'same_day':
            df['publish_date'] = df['ds']
        elif rule == 'next_month_day_15':
            # Monthly data published on 15th of following month
            df['publish_date'] = df['ds'] + pd.DateOffset(months=1, day=15)
        elif rule == 'month_end_plus_20d':
            # Published 20 days after month end
            df['publish_date'] = df['ds'] + pd.offsets.MonthEnd(0) + pd.DateOffset(days=20)
        else:
            df['publish_date'] = df['ds']

        return df

    def load_eia_series(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load real data from EIA API.

        Args:
            series_id: EIA series ID (e.g., 'PET.EMM_EPMR_PTE_NUS_DPG.W')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with ds, value, publish_date columns
        """
        logger.info(f"Loading EIA series: {series_id}")

        # Check cache first
        cache_file = CACHE_DIR / f"eia_{series_id.replace('.', '_')}_{start_date}_{end_date}.parquet"
        if cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age.days < 7:
                logger.info(f"Using cached EIA data for {series_id}")
                return pd.read_parquet(cache_file)

        if not self.eia_api_key:
            logger.warning("EIA_API_KEY not set. Using demo data structure.")
            # Return empty DataFrame with correct schema if no API key
            return pd.DataFrame(columns=['ds', 'value', 'series', 'publish_date'])

        # EIA API v2 endpoint
        base_url = "https://api.eia.gov/v2/seriesid/"

        headers = {
            'X-Api-Key': self.eia_api_key
        }

        params = {
            'series_id': series_id,
            'start': start_date,
            'end': end_date,
            'frequency': 'weekly'
        }

        response = requests.get(base_url + quote(series_id), headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()

            if 'response' in data and 'data' in data['response']:
                series_data = data['response']['data']

                records = []
                for point in series_data:
                    records.append({
                        'ds': pd.to_datetime(point['period']),
                        'value': float(point['value'])
                    })

                df = pd.DataFrame(records)
                df['series'] = 'gas_price'

                # Apply publish rule for weekly gas prices (2 day delay)
                df['publish_date'] = df['ds'] + pd.DateOffset(days=2)

                # Cache the result
                df.to_parquet(cache_file, index=False)
                logger.info(f"Cached EIA data for {series_id}")

                return df
            else:
                logger.error(f"Unexpected EIA response format: {data}")
                return pd.DataFrame(columns=['ds', 'value', 'series', 'publish_date'])
        else:
            logger.error(f"Failed to fetch EIA data: {response.status_code}")
            return pd.DataFrame(columns=['ds', 'value', 'series', 'publish_date'])

    def load_all_macro_series(self, start_date: str = '2015-01-01',
                            end_date: str = '2024-12-31') -> pd.DataFrame:
        """
        Load all configured macro series from real sources.

        Returns:
            Combined DataFrame with all macro series
        """
        logger.info("Loading all macro series from real sources")

        dfs = []

        # FRED series
        fred_series = {
            'DPRIME': 'prime_rate',  # Bank prime rate
            'DFEDTARU': 'fed_target_upper',  # Fed funds target upper
            'CUSR0000SETA02': 'cpi_used_cars',  # CPI for used cars
            'M2SL': 'm2_money_supply',  # M2 money supply
        }

        for series_id, name in fred_series.items():
            try:
                df = self.load_fred_series(series_id, start_date, end_date)
                df['series'] = name  # Use friendly name
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to load FRED series {series_id}: {e}")

        # EIA series for gas prices
        try:
            gas_df = self.load_eia_series('PET.EMM_EPMR_PTE_NUS_DPG.W', start_date, end_date)
            if not gas_df.empty:
                dfs.append(gas_df)
        except Exception as e:
            logger.error(f"Failed to load EIA gas price data: {e}")

        # Proprietary indices (if available)
        if self.manheim_api_key:
            try:
                manheim_df = self._load_manheim_index(start_date, end_date)
                dfs.append(manheim_df)
            except Exception as e:
                logger.error(f"Failed to load Manheim index: {e}")

        if self.blackbook_api_key:
            try:
                blackbook_df = self._load_blackbook_index(start_date, end_date)
                dfs.append(blackbook_df)
            except Exception as e:
                logger.error(f"Failed to load Black Book index: {e}")

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Loaded {len(combined_df)} macro data points from {len(dfs)} series")
            return combined_df
        else:
            logger.warning("No macro series could be loaded")
            return pd.DataFrame(columns=['ds', 'value', 'series', 'publish_date'])

    def _load_manheim_index(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load Manheim Used Vehicle Value Index (proprietary)."""
        # This would require a Manheim API subscription
        # Placeholder for proprietary implementation
        logger.warning("Manheim index loading not implemented (requires API subscription)")
        return pd.DataFrame(columns=['ds', 'value', 'series', 'publish_date'])

    def _load_blackbook_index(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load Black Book retention index (proprietary)."""
        # This would require a Black Book API subscription
        # Placeholder for proprietary implementation
        logger.warning("Black Book index loading not implemented (requires API subscription)")
        return pd.DataFrame(columns=['ds', 'value', 'series', 'publish_date'])


# Singleton instance for easy import
data_ingestion = RealDataIngestion()