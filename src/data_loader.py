import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

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
    """Preprocess single building's data"""
    
    # Extract single building
    data = pd.DataFrame()
    data['energy'] = electricity_df[building_id]
    
    # Handle missing values
    data = data.interpolate(method='linear', limit=3)
    data = data.dropna()
    
    # Add time features
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['day_of_year'] = data.index.dayofyear
    
    # Add lag features
    for lag in [1, 2, 24, 168]:
        data[f'lag_{lag}'] = data['energy'].shift(lag)
    
    # Add rolling statistics
    data['rolling_mean_24h'] = data['energy'].rolling(window=24, min_periods=1).mean()
    data['rolling_std_24h'] = data['energy'].rolling(window=24, min_periods=1).std()
    
    # Drop NaN
    data = data.dropna()
    
    # Normalize
    scaler = StandardScaler()
    feature_cols = [col for col in data.columns if col != 'energy']
    data[feature_cols] = scaler.fit_transform(data[feature_cols])
    
    return data, scaler


def create_dataloaders(data, seq_length=24, batch_size=256, 
                       train_split=0.7, val_split=0.15):
    """Create train/val/test DataLoaders"""
    
    # Split data
    n = len(data)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))
    
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
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
