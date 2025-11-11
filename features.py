"""
Production-ready feature engineering module for used car pricing with macro indicators.
Handles temporal joins, macro data generation, and feature transforms.
Uses ONLY real production data - no synthetic or mock data.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import json
from data_ingestion import data_ingestion

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class MacroFeatureEngineering:
    """Create macro features with proper as-of joins to prevent leakage"""

    def __init__(self, macro_df: pd.DataFrame, config: Optional[Dict] = None):
        """
        Initialize macro feature engineering with real data.

        Args:
            macro_df: DataFrame with real macro data from production sources
            config: Optional configuration dictionary
        """
        self.macro_df = macro_df.copy()
        self.config = config or {}
        self.feature_dfs = {}
        self.validation_results = {}

    def create_transforms(self, series_name: str) -> pd.DataFrame:
        """
        Create YoY, MoM, and other transforms for a series.

        Args:
            series_name: Name of the macro series

        Returns:
            DataFrame with transformed features
        """
        logger.info(f"Creating transforms for {series_name}")

        df = self.macro_df[self.macro_df['series'] == series_name].copy()
        df = df.sort_values('ds').reset_index(drop=True)

        # Level
        df[f'{series_name}_level'] = df['value']

        # Changes and percentages based on series type
        if 'cpi' in series_name or 'm2' in series_name:
            # Monthly data
            df[f'{series_name}_mom'] = df['value'].pct_change(1) * 100
            df[f'{series_name}_yoy'] = df['value'].pct_change(12) * 100
            df[f'{series_name}_3m_ma'] = df['value'].rolling(3).mean()

        elif 'rate' in series_name or 'fed_target' in series_name:
            # Daily rate data
            df[f'{series_name}_delta'] = df['value'].diff()
            df[f'{series_name}_30d_ma'] = df['value'].rolling(30).mean()

            # Event-based changes (for FOMC decisions)
            df[f'{series_name}_delta_event'] = df[f'{series_name}_delta'].where(df[f'{series_name}_delta'] != 0)

        elif 'gas' in series_name:
            # Weekly gas price data
            df[f'{series_name}_wow'] = df['value'].pct_change(1) * 100
            df[f'{series_name}_mom'] = df['value'].pct_change(4) * 100  # ~4 weeks per month
            df[f'{series_name}_4w_ma'] = df['value'].rolling(4).mean()
            df[f'{series_name}_52w_ma'] = df['value'].rolling(52).mean()

        # Add lags
        lags = self.config.get('lags', [1, 3, 6])
        for lag in lags:
            df[f'{series_name}_lag{lag}'] = df[f'{series_name}_level'].shift(lag)

        return df

    def expand_to_daily(self, df: pd.DataFrame, series_name: str) -> pd.DataFrame:
        """
        Expand macro data to daily frequency for as-of joins.

        Args:
            df: DataFrame with macro data
            series_name: Name of the series

        Returns:
            Daily-frequency DataFrame with forward-filled values
        """
        logger.info(f"Expanding {series_name} to daily frequency")

        # Create daily date range
        min_date = df['ds'].min()
        max_date = df['ds'].max()
        daily_dates = pd.date_range(min_date, max_date, freq='D')

        # Create daily DataFrame
        daily_df = pd.DataFrame({'ds': daily_dates})

        # Merge with original data
        daily_df = daily_df.merge(df, on='ds', how='left')

        # Forward fill values (as-of logic)
        feature_cols = [col for col in df.columns if col != 'ds']
        daily_df[feature_cols] = daily_df[feature_cols].fillna(method='ffill')

        # Ensure publish_date is handled correctly
        if 'publish_date' in daily_df.columns:
            daily_df['publish_date'] = daily_df['publish_date'].fillna(method='ffill')

        return daily_df

    def perform_asof_join(self, vehicle_df: pd.DataFrame, macro_df: pd.DataFrame,
                         date_col: str = 'sale_date') -> pd.DataFrame:
        """
        Perform as-of join between vehicle transactions and macro data.

        Args:
            vehicle_df: Vehicle transaction data
            macro_df: Daily macro data with publish_date
            date_col: Date column in vehicle data

        Returns:
            Joined DataFrame with leakage check flags
        """
        logger.info(f"Performing as-of join on {date_col}")

        if date_col not in vehicle_df.columns:
            logger.warning(f"Date column '{date_col}' not found. Cannot join macro features.")
            return vehicle_df

        # Ensure date column is datetime
        vehicle_df[date_col] = pd.to_datetime(vehicle_df[date_col])
        macro_df['publish_date'] = pd.to_datetime(macro_df['publish_date'])
        macro_df['ds'] = pd.to_datetime(macro_df['ds'])

        # Sort both DataFrames by date
        vehicle_df = vehicle_df.sort_values(date_col)
        macro_df = macro_df.sort_values('publish_date')

        # Perform as-of merge
        # For each vehicle transaction, get the latest macro data where publish_date <= sale_date
        merged_df = pd.merge_asof(
            vehicle_df,
            macro_df,
            left_on=date_col,
            right_on='publish_date',
            direction='backward',
            suffixes=('', '_macro')
        )

        # Add leakage check flag
        merged_df['__leak_check'] = merged_df['publish_date'] <= merged_df[date_col]

        if not merged_df['__leak_check'].all():
            n_leaks = (~merged_df['__leak_check']).sum()
            logger.error(f"LEAKAGE DETECTED: {n_leaks} transactions have future macro data!")
            raise ValueError(f"Temporal leakage detected in {n_leaks} rows")

        return merged_df

    def leakage_report(self, df: pd.DataFrame, date_col: str = 'sale_date') -> pd.DataFrame:
        """
        Generate leakage validation report.

        Args:
            df: DataFrame with joined features
            date_col: Transaction date column

        Returns:
            Report DataFrame with leakage statistics
        """
        logger.info("Generating leakage report")

        report = []

        # Check each macro series
        macro_cols = [col for col in df.columns if col.startswith('__leak_')]

        for col in macro_cols:
            if col in df.columns:
                violations = (~df[col]).sum()
                pct_violations = 100 * violations / len(df)

                report.append({
                    'check': col.replace('__leak_', ''),
                    'violations': violations,
                    'pct_violations': pct_violations,
                    'status': 'PASS' if violations == 0 else 'FAIL'
                })

        # Overall check
        if '__leak_check' in df.columns:
            total_violations = (~df['__leak_check']).sum()
            report.append({
                'check': 'OVERALL',
                'violations': total_violations,
                'pct_violations': 100 * total_violations / len(df),
                'status': 'PASS' if total_violations == 0 else 'FAIL'
            })

        report_df = pd.DataFrame(report)

        if report_df.empty:
            logger.warning("No leakage checks found in DataFrame")
            return pd.DataFrame({
                'check': ['NO_CHECKS'],
                'violations': [0],
                'pct_violations': [0.0],
                'status': ['N/A']
            })

        return report_df


class VehicleFeatureEngineering:
    """Feature engineering for vehicle attributes using real data only."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize vehicle feature engineering.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create vehicle features from raw data.

        Args:
            df: Raw vehicle DataFrame

        Returns:
            DataFrame with engineered features
        """
        logger.info("Creating vehicle features from real data")

        df = df.copy()

        # Age of vehicle (if sale_date exists)
        if 'sale_date' in df.columns:
            df['sale_date'] = pd.to_datetime(df['sale_date'])
            df['vehicle_age'] = (df['sale_date'].dt.year - df['Year']).clip(lower=0)
        else:
            # If no sale date, cannot compute age at sale time
            df['vehicle_age'] = datetime.now().year - df['Year']

        # Mileage per year
        df['km_per_year'] = df['Kilometers_Driven'] / df['vehicle_age'].replace(0, 1)

        # Create make from name
        df['make'] = df['Name'].str.split().str[0]

        # Owner type encoding
        owner_map = {'First': 1, 'Second': 2, 'Third': 3, 'Fourth': 4}
        df['owner_numeric'] = df['Owner_Type'].map(owner_map)

        # Fuel efficiency category
        df['fuel_efficiency_category'] = pd.cut(
            df['Mileage'],
            bins=[0, 15, 20, 25, 100],
            labels=['Low', 'Medium', 'High', 'Very High']
        )

        # Engine size category
        df['engine_category'] = pd.cut(
            df['Engine'],
            bins=[0, 1000, 1500, 2000, 10000],
            labels=['Small', 'Medium', 'Large', 'Very Large']
        )

        # Power category
        df['power_category'] = pd.cut(
            df['Power'],
            bins=[0, 80, 120, 180, 1000],
            labels=['Low', 'Medium', 'High', 'Very High']
        )

        # Location-based features (major city indicator)
        major_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune']
        df['is_major_city'] = df['Location'].isin(major_cities).astype(int)

        # Premium segment indicator
        premium_makes = ['BMW', 'Mercedes', 'Audi', 'Jaguar', 'Land', 'Volvo', 'Porsche']
        df['is_premium'] = df['make'].isin(premium_makes).astype(int)

        # Transmission type binary
        df['is_automatic'] = (df['Transmission'] == 'Automatic').astype(int)

        # Fuel type encoding
        fuel_encoding = {
            'Petrol': 0,
            'Diesel': 1,
            'CNG': 2,
            'LPG': 3,
            'Electric': 4
        }
        df['fuel_type_encoded'] = df['Fuel_Type'].map(fuel_encoding)

        # Log transforms for skewed features
        df['log_km_driven'] = np.log1p(df['Kilometers_Driven'])
        df['log_price'] = np.log1p(df['Price'])

        # Interaction features
        df['age_km_interaction'] = df['vehicle_age'] * df['log_km_driven']
        df['power_weight_ratio'] = df['Power'] / df['Engine']

        logger.info(f"Created {len(df.columns)} total features")

        return df

    def select_features(self, df: pd.DataFrame, feature_groups: Optional[Dict] = None) -> pd.DataFrame:
        """
        Select feature subsets for modeling.

        Args:
            df: DataFrame with all features
            feature_groups: Dictionary of feature groups

        Returns:
            DataFrame with selected features
        """
        if feature_groups is None:
            # Default feature groups
            feature_groups = {
                'base': [
                    'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission',
                    'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats', 'Location'
                ],
                'engineered': [
                    'vehicle_age', 'km_per_year', 'make', 'owner_numeric',
                    'is_major_city', 'is_premium', 'is_automatic',
                    'log_km_driven', 'age_km_interaction', 'power_weight_ratio'
                ],
                'macro': [
                    'cpi_used_cars_yoy', 'prime_rate_level', 'gas_price_4w_ma',
                    'm2_money_supply_yoy'
                ]
            }

        selected_cols = []
        for group_name, cols in feature_groups.items():
            available_cols = [col for col in cols if col in df.columns]
            selected_cols.extend(available_cols)
            logger.info(f"Feature group '{group_name}': {len(available_cols)}/{len(cols)} features available")

        # Always include target and ID columns if present
        for col in ['Price', 'S.No.']:
            if col in df.columns and col not in selected_cols:
                selected_cols.append(col)

        return df[selected_cols]


def create_macro_feature_schema(macro_features_df: pd.DataFrame) -> Dict:
    """
    Generate schema documentation for macro features.

    Args:
        macro_features_df: DataFrame with macro features

    Returns:
        Schema dictionary
    """
    schema = {
        'version': '1.0',
        'created_at': datetime.now().isoformat(),
        'features': {}
    }

    # Identify macro feature columns
    macro_cols = [col for col in macro_features_df.columns if any(
        indicator in col for indicator in ['cpi', 'prime', 'fed', 'gas', 'm2', 'muvvi', 'blackbook']
    )]

    for col in macro_cols:
        # Determine feature type
        feature_type = 'continuous'
        if 'delta' in col or 'mom' in col or 'yoy' in col or 'wow' in col:
            feature_type = 'change'
        elif '_ma' in col:
            feature_type = 'moving_average'
        elif 'lag' in col:
            feature_type = 'lagged'

        # Get statistics
        stats = {
            'count': int(macro_features_df[col].count()),
            'mean': float(macro_features_df[col].mean()) if pd.api.types.is_numeric_dtype(macro_features_df[col]) else None,
            'std': float(macro_features_df[col].std()) if pd.api.types.is_numeric_dtype(macro_features_df[col]) else None,
            'min': float(macro_features_df[col].min()) if pd.api.types.is_numeric_dtype(macro_features_df[col]) else None,
            'max': float(macro_features_df[col].max()) if pd.api.types.is_numeric_dtype(macro_features_df[col]) else None,
        }

        schema['features'][col] = {
            'type': feature_type,
            'description': f'Real {col} from production data sources',
            'source': 'FRED/EIA/Proprietary',
            'frequency': 'daily/weekly/monthly',
            'statistics': stats
        }

    return schema


# Configuration for macro sources (using real data)
MACRO_SOURCES = [
    {
        "name": "cpi_used_cars",
        "series_id": "CUSR0000SETA02",
        "source": "FRED",
        "freq": "monthly",
        "publish_rule": "next_month_day_15",
        "transforms": ["level", "pct_mom", "pct_yoy"],
        "lags": [1, 3],
        "enabled": True
    },
    {
        "name": "prime_rate",
        "series_id": "DPRIME",
        "source": "FRED",
        "freq": "daily",
        "publish_rule": "same_day",
        "transforms": ["level", "delta"],
        "lags": [1],
        "enabled": True
    },
    {
        "name": "fed_target_upper",
        "series_id": "DFEDTARU",
        "source": "FRED",
        "freq": "daily",
        "publish_rule": "same_day",
        "transforms": ["level", "delta_event"],
        "lags": [1],
        "enabled": True
    },
    {
        "name": "m2_money_supply",
        "series_id": "M2SL",
        "source": "FRED",
        "freq": "monthly",
        "publish_rule": "month_end_plus_20d",
        "transforms": ["pct_yoy"],
        "lags": [1],
        "enabled": True
    },
    {
        "name": "gas_price",
        "series_id": "PET.EMM_EPMR_PTE_NUS_DPG.W",
        "source": "EIA",
        "freq": "weekly",
        "publish_rule": "weekly_release_plus_2d",
        "transforms": ["level", "pct_mom", "rolling_4w_ma"],
        "lags": [1, 2],
        "enabled": True
    },
    {
        "name": "muvvi_index",
        "series_id": "MANHEIM_MUVVI",
        "source": "Manheim",
        "freq": "monthly",
        "publish_rule": "month_end_plus_5d",
        "transforms": ["yoy"],
        "lags": [1],
        "enabled": False  # Requires API subscription
    },
    {
        "name": "blackbook_retention_index",
        "series_id": "BB_RETENTION",
        "source": "BlackBook",
        "freq": "monthly",
        "publish_rule": "month_end_plus_7d",
        "transforms": ["yoy"],
        "lags": [1],
        "enabled": False  # Requires API subscription
    }
]


def main():
    """Main execution for feature engineering with real data."""

    logger.info("Starting production feature engineering pipeline")

    # Load real vehicle data
    vehicle_df = data_ingestion.load_transactions()

    # Check if date column exists
    has_dates = 'sale_date' in vehicle_df.columns

    if has_dates:
        logger.info("sale_date found - loading real macro data")

        # Load real macro data from production sources
        start_date = str(vehicle_df['sale_date'].min().date()) if has_dates else '2015-01-01'
        end_date = str(vehicle_df['sale_date'].max().date()) if has_dates else '2024-12-31'

        macro_df = data_ingestion.load_all_macro_series(start_date, end_date)

        if not macro_df.empty:
            # Process macro features
            macro_fe = MacroFeatureEngineering(macro_df)

            # Create transforms for each series
            transformed_dfs = []
            for source in MACRO_SOURCES:
                if source['enabled']:
                    series_name = source['name']
                    series_df = macro_df[macro_df['series'] == series_name]

                    if not series_df.empty:
                        transformed = macro_fe.create_transforms(series_name)
                        daily_expanded = macro_fe.expand_to_daily(transformed, series_name)
                        transformed_dfs.append(daily_expanded)

            # Combine all macro features
            if transformed_dfs:
                combined_macro = transformed_dfs[0]
                for df in transformed_dfs[1:]:
                    combined_macro = combined_macro.merge(df, on='ds', how='outer')

                # Perform as-of join with vehicle data
                vehicle_df = macro_fe.perform_asof_join(vehicle_df, combined_macro, 'sale_date')

                # Run leakage report
                leakage_df = macro_fe.leakage_report(vehicle_df)
                print("\n=== Leakage Report ===")
                print(leakage_df)

                if (leakage_df['violations'] > 0).any():
                    raise ValueError("Temporal leakage detected! Fix before proceeding.")

    else:
        logger.warning("No sale_date column - proceeding without macro features")

    # Engineer vehicle features
    vehicle_fe = VehicleFeatureEngineering()
    final_df = vehicle_fe.create_features(vehicle_df)

    # Save features
    output_dir = Path('data/features')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'features.parquet'
    final_df.to_parquet(output_path, index=False)
    logger.info(f"Saved features to {output_path}")

    # Generate and save schema
    schema = create_macro_feature_schema(final_df)
    schema_path = output_dir / 'macro_feature_schema.json'
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2, default=str)
    logger.info(f"Saved schema to {schema_path}")

    print(f"\n=== Feature Engineering Complete ===")
    print(f"Total records: {len(final_df)}")
    print(f"Total features: {len(final_df.columns)}")
    print(f"Macro features included: {has_dates and not macro_df.empty}")

    return final_df


if __name__ == "__main__":
    final_features = main()