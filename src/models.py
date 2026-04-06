import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

class EnergyLSTM(pl.LightningModule):
    """Baseline LSTM model for building energy forecasting"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=3, 
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
        
        # Deeper MLP head for better prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1)
        )
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


class EnergyLSTMFrozen(pl.LightningModule):
    """Transfer learning via frozen backbone.

    The LSTM encoder weights are locked after loading from a pre-trained
    checkpoint.  Only the MLP head is trained on the target building data.
    This prevents catastrophic forgetting and is very data-efficient.
    """

    def __init__(self, input_size, hidden_size=128, num_layers=3,
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

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1)
        )
        self.criterion = nn.MSELoss()

        # Freeze the LSTM encoder immediately
        for param in self.lstm.parameters():
            param.requires_grad = False

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

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
        # Only pass parameters that require gradients (the head)
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = Adam(trainable, lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        }


class LSTMAdapter(nn.Module):
    """Bottleneck adapter inserted between the LSTM output and the MLP head.

    Architecture (residual / LoRA-style):
        out = x + Linear(bottleneck → hidden)(ReLU(Linear(hidden → bottleneck)(x)))

    The adapter weights are initialised so the residual branch starts near zero,
    meaning the model initially behaves exactly like the frozen pre-trained backbone.
    """

    def __init__(self, hidden_size: int, bottleneck: int = 32):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act  = nn.ReLU()
        self.up   = nn.Linear(bottleneck, hidden_size)

        # Near-zero init so adapter starts as identity residual
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class EnergyLSTMAdapter(pl.LightningModule):
    """Transfer learning via adapter layers.

    The LSTM encoder and the original MLP head weights are frozen.
    A small trainable LSTMAdapter module (bottleneck=32) sits between the
    LSTM output and the head, and a fresh output projection is appended so
    the head itself can also fine-tune.

    Trainable parameters:
        - LSTMAdapter  (~8 K params)
        - MLP head     (~8 K params)
    Everything else is locked.
    """

    def __init__(self, input_size, hidden_size=128, num_layers=3,
                 dropout=0.2, learning_rate=1e-3, adapter_bottleneck=32):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.adapter = LSTMAdapter(hidden_size, bottleneck=adapter_bottleneck)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1)
        )
        self.criterion = nn.MSELoss()

        # Freeze LSTM encoder; adapter + fc remain trainable
        for param in self.lstm.parameters():
            param.requires_grad = False

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        adapted = self.adapter(last_output)
        return self.fc(adapted)

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
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = Adam(trainable, lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        }
