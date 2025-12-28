"""
Neural network models for stock price prediction and trading
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod
from src.utils.logger import logger


class TradingModel(ABC):
    """Abstract base class for trading models"""
    
    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def save(self, path):
        pass
    
    @abstractmethod
    def load(self, path):
        pass


class LSTMModel(TradingModel, nn.Module):
    """LSTM-based neural network for price prediction"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, 
                 output_size=1, device='cpu'):
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.device = device
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
        self.to(device)
    
    def forward(self, x):
        """Forward pass"""
        lstm_out, (hidden, cell) = self.lstm(x)
        # Use the last output
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        output = self.fc(last_out)
        return output
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32,
                   learning_rate=0.001, patience=10):
        """Train the LSTM model"""
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                        factor=0.5, patience=5, 
                                                        verbose=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        training_losses = []
        validation_losses = []
        
        self.train()
        
        for epoch in range(epochs):
            # Training
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.eval()
            with torch.no_grad():
                val_outputs = self.forward(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            self.train()
            
            training_losses.append(train_loss)
            validation_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                           f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        return training_losses, validation_losses
    
    def predict(self, X):
        """Make predictions"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.forward(X_tensor)
            return predictions.cpu().numpy()
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Training interface"""
        return self.train_model(X_train, y_train, X_val, y_val, epochs, batch_size)
    
    def save(self, path):
        """Save model"""
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path):
        """Load model"""
        self.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")


class GRUModel(TradingModel, nn.Module):
    """GRU-based neural network for price prediction"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2,
                 output_size=1, device='cpu'):
        super(GRUModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.device = device
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
        self.to(device)
    
    def forward(self, x):
        """Forward pass"""
        gru_out, hidden = self.gru(x)
        last_out = gru_out[:, -1, :]
        last_out = self.dropout(last_out)
        output = self.fc(last_out)
        return output
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32,
                   learning_rate=0.001, patience=10):
        """Train the GRU model"""
        
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=0.5, patience=5,
                                                        verbose=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        training_losses = []
        validation_losses = []
        
        self.train()
        
        for epoch in range(epochs):
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            self.eval()
            with torch.no_grad():
                val_outputs = self.forward(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            self.train()
            
            training_losses.append(train_loss)
            validation_losses.append(val_loss)
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                           f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        return training_losses, validation_losses
    
    def predict(self, X):
        """Make predictions"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.forward(X_tensor)
            return predictions.cpu().numpy()
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Training interface"""
        return self.train_model(X_train, y_train, X_val, y_val, epochs, batch_size)
    
    def save(self, path):
        """Save model"""
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path):
        """Load model"""
        self.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")


class EnsembleModel(TradingModel):
    """Ensemble model combining multiple neural networks"""
    
    def __init__(self, models: list, weights: list = None, device='cpu'):
        self.models = models
        self.device = device
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = np.array(weights) / np.sum(weights)
    
    def predict(self, X):
        """Predict using all models and average"""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        return ensemble_pred
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train all models"""
        losses = {}
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}")
            losses[f"model_{i}"] = model.train(X_train, y_train, X_val, y_val, epochs, batch_size)
        return losses
    
    def save(self, path):
        """Save all models"""
        for i, model in enumerate(self.models):
            model.save(f"{path}_model_{i}.pt")
    
    def load(self, path):
        """Load all models"""
        for i, model in enumerate(self.models):
            model.load(f"{path}_model_{i}.pt")


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size=50, hidden_size=128, num_layers=2, device=device)
    
    # Create dummy data
    X_train = np.random.randn(100, 60, 50).astype(np.float32)
    y_train = np.random.randn(100).astype(np.float32)
    X_val = np.random.randn(20, 60, 50).astype(np.float32)
    y_val = np.random.randn(20).astype(np.float32)
    
    model.train(X_train, y_train, X_val, y_val, epochs=5, batch_size=32)
    predictions = model.predict(X_val)
    print(f"Predictions shape: {predictions.shape}")
