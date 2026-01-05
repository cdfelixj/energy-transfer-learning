import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import os


def load_electricity_data(data_dir=None):
    """Load filtered electricity data (Education buildings from Rat site only).
    
    Automatically filters for:
    - Building type: Education (most data: 561 buildings)
    - Site: Rat (most buildings: 267 buildings)
    - Meter type: Electricity only (most coverage: 1578 buildings)
    
    Args:
        data_dir: Base directory containing the data. If None, uses default relative path.
        
    Returns:
        electricity_df: Filtered electricity consumption dataframe (only Education + Rat)
        metadata: Full metadata dataframe
        valid_buildings: List of valid building IDs after filtering
    """
    # Set default paths
    if data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_dir = os.path.join(project_root, 'data', 'raw', 'building-data-genome-project-2', 'data')
    
    metadata_path = os.path.join(data_dir, 'metadata', 'metadata.csv')
    electricity_path = os.path.join(data_dir, 'meters', 'cleaned', 'electricity_cleaned.csv')
    
    print("\n" + "="*70)
    print("LOADING FILTERED ELECTRICITY DATA")
    print("="*70)
    print("Filtering criteria:")
    print("  ✓ Building type: Education")
    print("  ✓ Site: Rat")
    print("  ✓ Meter type: Electricity only")
    print("="*70 + "\n")
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    
    # Load electricity data
    electricity_full = pd.read_csv(electricity_path, index_col=0, parse_dates=True)
    
    # Filter for Education buildings from Rat site
    filtered_metadata = metadata[
        (metadata['primaryspaceusage'] == 'Education') & 
        (metadata['site_id'] == 'Rat')
    ]
    
    # Get valid building IDs that exist in both metadata and electricity data
    valid_buildings = [
        bid for bid in filtered_metadata['building_id'].values 
        if bid in electricity_full.columns
    ]
    
    if len(valid_buildings) == 0:
        raise ValueError("No buildings found matching criteria (Education + Rat site + Electricity)")
    
    # Filter electricity dataframe to only include valid buildings
    electricity_df = electricity_full[valid_buildings].copy()
    
    print(f"✓ Loaded {len(valid_buildings)} Education buildings from Rat site")
    print(f"  Date range: {electricity_df.index.min()} to {electricity_df.index.max()}")
    print(f"  Total timestamps: {len(electricity_df):,}")
    print(f"  Buildings: {valid_buildings[:5]}{'...' if len(valid_buildings) > 5 else ''}")
    print()
    
    return electricity_df, metadata, valid_buildings


class BuildingEnergyDataset(Dataset):
    """PyTorch Dataset for building energy time series"""
    
    def __init__(self, data, seq_length=24, target_col='energy'):
        self.seq_length = seq_length
        self.target_col = target_col
        
        # Separate features and target
        self.features = data.drop(columns=[target_col]).values
        self.target = data[target_col].values
        
    def __len__(self):
        return len(self.features) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        y = self.target[idx + self.seq_length]
        return torch.FloatTensor(x), torch.FloatTensor([y])


def preprocess_building_data(electricity_df, building_id, weather_df=None):
    """Preprocess single building's data with comprehensive cleaning.
    
    Note: Assumes electricity_df is already filtered for Education + Rat site buildings.
    Use load_electricity_data() to get the filtered dataframe.
    
    Args:
        electricity_df: DataFrame with electricity consumption data (filtered)
        building_id: ID of the building to preprocess
        weather_df: Optional weather data DataFrame
    """
    # Extract single building
    data = pd.DataFrame()
    
    # Check if building exists
    if building_id not in electricity_df.columns:
        raise ValueError(f"Building '{building_id}' not found in electricity data. "
                        f"Available buildings: {list(electricity_df.columns)[:10]}...")
    
    data['energy'] = electricity_df[building_id]
    
    # Data Quality Report
    initial_size = len(data)
    initial_non_null = data['energy'].notna().sum()
    print(f"\n=== Data Quality Report for {building_id} ===")
    print(f"Initial data points: {initial_size}")
    print(f"Initial non-null values: {initial_non_null} ({initial_non_null/initial_size*100:.2f}%)")
    
    # 1. Remove duplicate timestamps
    duplicates = data.index.duplicated().sum()
    if duplicates > 0:
        print(f"Removing {duplicates} duplicate timestamps ({duplicates/initial_size*100:.2f}%)")
        data = data[~data.index.duplicated(keep='first')]
    
    # 2. Validate data types and convert to numeric
    data['energy'] = pd.to_numeric(data['energy'], errors='coerce')
    
    # 3. Data validation and cleaning
    # Check for negative values
    negative_count = (data['energy'] < 0).sum()
    if negative_count > 0:
        print(f"Warning: Found {negative_count} negative energy values ({negative_count/len(data)*100:.2f}%) - setting to NaN")
        data.loc[data['energy'] < 0, 'energy'] = np.nan
    
    # Check for extreme outliers (values > 10x the 95th percentile)
    valid_data = data['energy'].dropna()
    if len(valid_data) > 0:
        percentile_95 = valid_data.quantile(0.95)
        extreme_threshold = percentile_95 * 10
        extreme_count = (data['energy'] > extreme_threshold).sum()
        if extreme_count > 0:
            print(f"Warning: Found {extreme_count} extreme outliers (>10x 95th percentile) - setting to NaN")
            data.loc[data['energy'] > extreme_threshold, 'energy'] = np.nan
    
    # Check for extended zero periods (more than 72 consecutive hours)
    zero_mask = (data['energy'] == 0)
    if zero_mask.any():
        zero_groups = (zero_mask != zero_mask.shift()).cumsum()
        zero_lengths = zero_mask.groupby(zero_groups).transform('sum')
        extended_zeros = (zero_lengths > 72) & zero_mask
        extended_zero_count = extended_zeros.sum()
        if extended_zero_count > 0:
            print(f"Warning: Found {extended_zero_count} values in extended zero periods (>72 hours) - setting to NaN")
            data.loc[extended_zeros, 'energy'] = np.nan
    
    # 4. Handle missing values
    missing_count = data['energy'].isna().sum()
    print(f"Missing/invalid values after validation: {missing_count} ({missing_count/len(data)*100:.2f}%)")
    
    data['was_interpolated'] = data['energy'].isna().astype(int)
    data = data.interpolate(method='linear', limit=3)
    data = data.dropna()
    
    final_size = len(data)
    removed = initial_size - final_size
    print(f"Final data points: {final_size} (removed {removed}, {removed/initial_size*100:.2f}%)")
    print(f"Energy range: [{data['energy'].min():.2f}, {data['energy'].max():.2f}]")
    print(f"Energy mean: {data['energy'].mean():.2f}, std: {data['energy'].std():.2f}")
    print("=" * 50 + "\n")
    
    # Add cyclical time features (so hour 23 is close to hour 0)
    data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
    data['day_of_week_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
    data['day_of_week_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)
    data['month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
    data['month_cos'] = np.cos(2 * np.pi * data.index.month / 12)
    data['day_of_year_sin'] = np.sin(2 * np.pi * data.index.dayofyear / 365)
    data['day_of_year_cos'] = np.cos(2 * np.pi * data.index.dayofyear / 365)
    
    # Add weekend and business hours flags
    data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
    data['is_business_hours'] = ((data.index.hour >= 8) & (data.index.hour <= 18)).astype(int)
    
    # Add more comprehensive lag features
    for lag in [1, 2, 24, 48, 72, 168, 336, 720]:
        data[f'lag_{lag}'] = data['energy'].shift(lag)
    
    # Add multiple rolling statistics
    data['rolling_mean_24h'] = data['energy'].rolling(window=24, min_periods=1).mean()
    data['rolling_std_24h'] = data['energy'].rolling(window=24, min_periods=1).std()
    data['rolling_mean_168h'] = data['energy'].rolling(window=168, min_periods=1).mean()
    data['rolling_std_168h'] = data['energy'].rolling(window=168, min_periods=1).std()
    data['rolling_min_24h'] = data['energy'].rolling(window=24, min_periods=1).min()
    data['rolling_max_24h'] = data['energy'].rolling(window=24, min_periods=1).max()
    
    # Add weather features if available
    if weather_df is not None:
        try:
            # Merge weather data
            weather_features = ['airTemperature', 'dewTemperature', 'cloudCoverage', 
                              'precipDepth1HR', 'seaLvlPressure', 'windSpeed']
            
            for feature in weather_features:
                if feature in weather_df.columns:
                    data[feature] = weather_df[feature]
                    
                    # Check if feature is mostly missing
                    missing_pct = data[feature].isna().sum() / len(data) * 100
                    if missing_pct > 95:
                        print(f"Dropping weather feature '{feature}' ({missing_pct:.1f}% missing)")
                        data = data.drop(columns=[feature])
                    else:
                        # Interpolate missing weather values (more aggressive for partially missing)
                        data[feature] = data[feature].interpolate(method='linear', limit_direction='both')
                        # Fill any remaining NaN with median
                        if data[feature].isna().any():
                            median_val = data[feature].median()
                            data[feature] = data[feature].fillna(median_val)
                            print(f"Filled remaining NaN in '{feature}' with median: {median_val:.2f}")
        except Exception as e:
            print(f"Warning: Could not add weather features: {e}")
    
    # Drop NaN
    data = data.dropna()
    
    if len(data) == 0:
        raise ValueError(f"All data was dropped after removing NaN values. "
                        f"This building may have insufficient data or too many missing values.")
    
    # Normalize features (but not energy target)
    scaler = StandardScaler()
    feature_cols = [col for col in data.columns if col != 'energy']
    data[feature_cols] = scaler.fit_transform(data[feature_cols])
    
    return data, scaler


def create_dataloaders(data, seq_length=24, batch_size=256, 
                       train_split=0.6, val_split=0.2, shuffle_split=True):
    """Create train/val/test DataLoaders
    
    Args:
        data: Preprocessed dataframe with features and target
        seq_length: Length of input sequences in hours
        batch_size: Batch size for dataloaders
        train_split: Proportion of data for training (default 0.6)
        val_split: Proportion of data for validation (default 0.2)
        shuffle_split: If True, use stratified random split to avoid distribution mismatch
                      If False, use chronological split (default True)
        
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Split data
    n = len(data)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))
    
    if shuffle_split:
        # Use stratified random split to ensure similar distributions
        # Group by month to maintain some temporal structure
        from sklearn.model_selection import train_test_split
        
        # Create month-based stratification
        data['_month'] = data.index.month
        
        # First split: train vs (val+test)
        train_data, temp_data = train_test_split(
            data, train_size=train_split, random_state=42, 
            stratify=data['_month'], shuffle=True
        )
        
        # Second split: val vs test
        val_size = val_split / (val_split + (1 - train_split - val_split))
        val_data, test_data = train_test_split(
            temp_data, train_size=val_size, random_state=42,
            stratify=temp_data['_month'], shuffle=True
        )
        
        # Remove temporary stratification column
        train_data = train_data.drop(columns=['_month']).sort_index()
        val_data = val_data.drop(columns=['_month']).sort_index()
        test_data = test_data.drop(columns=['_month']).sort_index()
        
        print(f"\n=== Using STRATIFIED RANDOM SPLIT (avoids distribution mismatch) ===")
    else:
        # Original chronological split
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        
        print(f"\n=== Using CHRONOLOGICAL SPLIT (may have distribution shift) ===")
    
    print(f"\n=== DataLoader Creation ===")
    print(f"Total data: {n}, Sequence length: {seq_length}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Show distribution stats to verify no mismatch
    print(f"\nDistribution check:")
    print(f"  Train - Mean: {train_data['energy'].mean():.2f}, Std: {train_data['energy'].std():.2f}")
    print(f"  Val   - Mean: {val_data['energy'].mean():.2f}, Std: {val_data['energy'].std():.2f}")
    print(f"  Test  - Mean: {test_data['energy'].mean():.2f}, Std: {test_data['energy'].std():.2f}")
    
    train_mean = train_data['energy'].mean()
    test_mean = test_data['energy'].mean()
    mean_shift = ((test_mean - train_mean) / train_mean) * 100
    print(f"  Mean shift (train→test): {mean_shift:+.1f}%")
    if abs(mean_shift) > 20:
        print(f"  ⚠ WARNING: Large distribution shift may cause poor generalization!")
    
    # Validate that each split has enough data
    min_required_size = seq_length + 1
    if len(val_data) < min_required_size:
        raise ValueError(f"Validation data ({len(val_data)}) is too small for sequence length {seq_length}. "
                        f"Need at least {min_required_size} samples. Consider reducing seq_length or val_split.")
    if len(test_data) < min_required_size:
        raise ValueError(f"Test data ({len(test_data)}) is too small for sequence length {seq_length}. "
                        f"Need at least {min_required_size} samples. Consider reducing seq_length or test_split.")
    
    # Create datasets
    train_dataset = BuildingEnergyDataset(train_data, seq_length)
    val_dataset = BuildingEnergyDataset(val_data, seq_length)
    test_dataset = BuildingEnergyDataset(test_data, seq_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=0)  # num_workers=0 for Windows
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader
