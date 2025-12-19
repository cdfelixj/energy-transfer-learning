import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

class EnergyLSTM(pl.LightningModule):
    """Baseline LSTM model for building energy forecasting"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, 
                 dropout=0.2, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        mae = torch.mean(torch.abs(y_hat - y))
        rmse = torch.sqrt(loss)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True)
        self.log('val_rmse', rmse, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        mae = torch.mean(torch.abs(y_hat - y))
        rmse = torch.sqrt(loss)
        
        self.log('test_loss', loss)
        self.log('test_mae', mae)
        self.log('test_rmse', rmse)
        return {'y_true': y, 'y_pred': y_hat}
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
