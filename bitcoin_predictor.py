"""
Bitcoin Price Prediction using GRU Neural Network
A machine learning model for predicting Bitcoin prices using historical data
"""

import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from binance.client import Client
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class OptimizerClass:
    """RMSprop optimizer for neural network training"""
    
    def __init__(self, weights: List[np.ndarray], learning_rate: float = 0.01):
        self.lr = learning_rate
        self.mems = []
        for tensor in weights:
            self.mems.append(np.zeros_like(tensor))

    def update_weights(self, params: List[np.ndarray], dparams: List[np.ndarray]):
        """Update weights using RMSprop algorithm"""
        for param, dparam, mem in zip(params, dparams, self.mems):
            dparam = np.clip(dparam, -1, 1)
            mem += dparam * dparam
            param += -self.lr * dparam / np.sqrt(mem + 1e-8)


class GRU:
    """Gated Recurrent Unit (GRU) Neural Network implementation"""
    
    def __init__(self, in_size: int, out_size: int, hidden_size: int):
        """
        Initialize GRU network
        
        Args:
            in_size: Input feature size
            out_size: Output size
            hidden_size: Hidden layer size
        """
        # Input to hidden weights
        self.Wxc = np.random.randn(hidden_size, in_size) * 0.1  # input to candidate
        self.Wxr = np.random.randn(hidden_size, in_size) * 0.1  # input to reset
        self.Wxz = np.random.randn(hidden_size, in_size) * 0.1  # input to interpolate
        
        # Hidden to hidden weights
        self.Rhc = np.random.randn(hidden_size, hidden_size) * 0.1  # hidden to candidate
        self.Rhr = np.random.randn(hidden_size, hidden_size) * 0.1  # hidden to reset
        self.Rhz = np.random.randn(hidden_size, hidden_size) * 0.1  # hidden to interpolate
        
        # Output weights
        self.Why = np.random.randn(out_size, hidden_size) * 0.1  # hidden to output
        
        # Biases
        self.bc = np.zeros((hidden_size, 1))
        self.br = np.zeros((hidden_size, 1))
        self.bz = np.zeros((hidden_size, 1))
        self.by = np.zeros((out_size, 1))
        
        # Store weights for optimization
        self.weights = [
            self.Wxc, self.Wxr, self.Wxz,
            self.Rhc, self.Rhr, self.Rhz,
            self.Why, self.bc, self.br, self.bz, self.by
        ]
        
        self.names = [
            'Wxc', 'Wxr', 'Wxz',
            'Rhc', 'Rhr', 'Rhz',
            'Why', 'bc', 'br', 'bz', 'by'
        ]

    def forward(self, inputs: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """
        Forward pass through GRU network
        
        Args:
            inputs: Input sequence
            
        Returns:
            Tuple of (hidden states, predictions)
        """
        xs, rbars, rs, zbars, zs, cbars, cs, ps, hs = {}, {}, {}, {}, {}, {}, {}, {}, {}
        hs[-1] = np.zeros((self.Wxc.shape[0], 1))
        
        for t in range(len(inputs)):
            xs[t] = np.matrix(inputs[t]).T
            
            # Reset gate
            rbars[t] = np.dot(self.Wxr, xs[t]) + np.dot(self.Rhr, hs[t - 1]) + self.br
            rs[t] = 1 / (1 + np.exp(-rbars[t]))
            
            # Update gate
            zbars[t] = np.dot(self.Wxz, xs[t]) + np.dot(self.Rhz, hs[t - 1]) + self.bz
            zs[t] = 1 / (1 + np.exp(-zbars[t]))
            
            # Candidate hidden state
            cbars[t] = np.dot(self.Wxc, xs[t]) + np.dot(self.Rhc, np.multiply(rs[t], hs[t - 1])) + self.bc
            cs[t] = np.tanh(cbars[t])
            
            # Hidden state
            ones = np.ones_like(zs[t])
            hs[t] = np.multiply(cs[t], zs[t]) + np.multiply(hs[t - 1], ones - zs[t])
            
            # Output
            ps[t] = np.dot(self.Why, hs[t]) + self.by
            
        return (xs, rbars, rs, zbars, zs, cbars, cs, ps, hs), ps

    def train_model(self, inputs: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """
        Train the GRU model
        
        Args:
            inputs: Input sequences
            targets: Target values
            
        Returns:
            Tuple of (loss, gradients, outputs)
        """
        # Forward pass
        (xs, rbars, rs, zbars, zs, cbars, cs, ps, hs), ys = self.forward(inputs)
        
        # Calculate loss
        loss = np.square(ys - targets)
        total_loss = np.sum(loss)
        
        # Backward pass
        dWxc, dWxr, dWxz = np.zeros_like(self.Wxc), np.zeros_like(self.Wxr), np.zeros_like(self.Wxz)
        dRhc, dRhr, dRhz = np.zeros_like(self.Rhc), np.zeros_like(self.Rhr), np.zeros_like(self.Rhz)
        dWhy = np.zeros_like(self.Why)
        dbc, dbr, dbz = np.zeros_like(self.bc), np.zeros_like(self.br), np.zeros_like(self.bz)
        dby = np.zeros_like(self.by)
        
        dhnext = np.zeros_like(hs[0])
        drbarnext = np.zeros_like(rbars[0])
        dzbarnext = np.zeros_like(zbars[0])
        dcbarnext = np.zeros_like(cbars[0])
        
        for t in reversed(range(len(inputs))):
            dy = ys[t] - targets[t]
            
            # Backprop through output layer
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            
            # Backprop through hidden state
            dh = np.dot(self.Why.T, dy) + dhnext
            
            # Backprop through gates
            dc = np.multiply(dh, zs[t])
            dcbar = np.multiply(dc, 1 - np.square(cs[t]))
            
            dr = np.multiply(hs[t-1], np.dot(self.Rhc.T, dcbar))
            dz = np.multiply(dh, cs[t] - hs[t-1])
            
            # Backprop through sigmoids
            drbar = np.multiply(dr, np.multiply(rs[t], 1 - rs[t]))
            dzbar = np.multiply(dz, np.multiply(zs[t], 1 - zs[t]))
            
            # Update gradients
            dWxr += np.dot(drbar, xs[t].T)
            dWxz += np.dot(dzbar, xs[t].T)
            dWxc += np.dot(dcbar, xs[t].T)
            
            dRhr += np.dot(drbar, hs[t-1].T)
            dRhz += np.dot(dzbar, hs[t-1].T)
            dRhc += np.dot(dcbar, np.multiply(rs[t], hs[t-1]).T)
            
            dbr += drbar
            dbc += dcbar
            dbz += dzbar
            
            # Prepare for next iteration
            dhnext = dh
            drbarnext = drbar
            dzbarnext = dzbar
            dcbarnext = dcbar
        
        deltas = [dWxc, dWxr, dWxz, dRhc, dRhr, dRhz, dWhy, dbc, dbr, dbz, dby]
        
        return total_loss, deltas, ys

    def predict(self, inputs: np.ndarray) -> List[np.ndarray]:
        """
        Make predictions using trained model
        
        Args:
            inputs: Input sequences
            
        Returns:
            List of predictions
        """
        y_pred = []
        xs, rbars, rs, zbars, zs, cbars, cs, ps, hs = {}, {}, {}, {}, {}, {}, {}, {}, {}
        hs[-1] = np.zeros((self.Wxc.shape[0], 1))
        
        for t in range(len(inputs)):
            xs[t] = np.matrix(inputs[t]).T
            
            # Reset gate
            rbars[t] = np.dot(self.Wxr, xs[t]) + np.dot(self.Rhr, hs[t - 1]) + self.br
            rs[t] = 1 / (1 + np.exp(-rbars[t]))
            
            # Update gate
            zbars[t] = np.dot(self.Wxz, xs[t]) + np.dot(self.Rhz, hs[t - 1]) + self.bz
            zs[t] = 1 / (1 + np.exp(-zbars[t]))
            
            # Candidate hidden state
            cbars[t] = np.dot(self.Wxc, xs[t]) + np.dot(self.Rhc, np.multiply(rs[t], hs[t - 1])) + self.bc
            cs[t] = np.tanh(cbars[t])
            
            # Hidden state
            ones = np.ones_like(zs[t])
            hs[t] = np.multiply(cs[t], zs[t]) + np.multiply(hs[t - 1], ones - zs[t])
            
            # Output
            y_pred.append(np.dot(self.Why, hs[t]) + self.by)
            
        return y_pred


class BitcoinPredictor:
    """Main class for Bitcoin price prediction"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        """
        Initialize Bitcoin predictor
        
        Args:
            api_key: Binance API key (optional)
            api_secret: Binance API secret (optional)
        """
        self.client = Client(api_key, api_secret)
        self.feature_scaler = StandardScaler()
        self.label_scaler = StandardScaler()
        self.model = None
        self.optimizer = None
        
    def fetch_bitcoin_data(self, symbol: str = "BTCUSDT", interval: str = Client.KLINE_INTERVAL_1DAY, 
                          start_date: str = "1 Jan 2019") -> pd.DataFrame:
        """
        Fetch Bitcoin historical data from Binance
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            start_date: Start date for data
            
        Returns:
            DataFrame with Bitcoin data
        """
        print(f"Fetching {symbol} data from {start_date}...")
        
        klines = self.client.get_historical_klines(symbol, interval, start_date)
        bitcoin_df = pd.DataFrame(klines)
        
        bitcoin_df.columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote", "no_trades", "base_buy", "quote_buy", "ignore"
        ]
        
        # Convert timestamp to datetime
        bitcoin_df["date"] = bitcoin_df["open_time"].apply(
            lambda x: datetime.datetime.fromtimestamp(x/1000)
        )
        
        # Extract date features
        bitcoin_df['month'] = bitcoin_df['date'].dt.month
        bitcoin_df['day'] = bitcoin_df['date'].dt.day
        bitcoin_df['year'] = bitcoin_df['date'].dt.year
        
        # Drop unnecessary columns
        bitcoin_df.drop(['date', 'ignore'], axis=1, inplace=True)
        
        # Convert to numeric
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            bitcoin_df[col] = pd.to_numeric(bitcoin_df[col])
            
        print(f"Fetched {len(bitcoin_df)} records")
        return bitcoin_df
    
    def prepare_data(self, df: pd.DataFrame, train_cols: List[str] = None, 
                    label_col: str = 'close', test_size: float = 0.2) -> Tuple:
        """
        Prepare data for training
        
        Args:
            df: Input DataFrame
            train_cols: Feature columns
            label_col: Target column
            test_size: Test set size
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if train_cols is None:
            train_cols = ['open', 'high', 'low', 'volume']
            
        X = df[train_cols]
        y = df[[label_col]]
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.label_scaler.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   hidden_size: int = 100, learning_rate: float = 0.01, 
                   epochs: int = 500) -> Dict[str, Any]:
        """
        Train the GRU model
        
        Args:
            X_train: Training features
            y_train: Training labels
            hidden_size: Hidden layer size
            learning_rate: Learning rate
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        print("Initializing GRU model...")
        
        # Initialize model
        in_size = X_train.shape[1]
        out_size = y_train.shape[1]
        
        self.model = GRU(in_size, out_size, hidden_size)
        self.optimizer = OptimizerClass(self.model.weights, learning_rate)
        
        # Training history
        history = {'loss': [], 'epoch': []}
        
        print(f"Training for {epochs} epochs...")
        for epoch in range(epochs):
            error, deltas, outputs = self.model.train_model(X_train, y_train)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {error:.2f}")
                
            history['loss'].append(error)
            history['epoch'].append(epoch)
            
            self.optimizer.update_weights(self.model.weights, deltas)
        
        print("Training completed!")
        return history
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        predictions = self.model.predict(X_test)
        predictions = np.array(predictions).flatten()
        y_test_flat = y_test.flatten()
        
        # Calculate metrics
        mse = np.mean((predictions - y_test_flat) ** 2)
        mae = np.mean(np.abs(predictions - y_test_flat))
        rmse = np.sqrt(mse)
        
        # Inverse transform for actual price values
        predictions_actual = self.label_scaler.inverse_transform(predictions.reshape(-1, 1))
        y_test_actual = self.label_scaler.inverse_transform(y_test)
        
        mse_actual = np.mean((predictions_actual - y_test_actual) ** 2)
        rmse_actual = np.sqrt(mse_actual)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mse_actual': mse_actual,
            'rmse_actual': rmse_actual
        }
    
    def plot_training_history(self, history: Dict[str, List]):
        """Plot training loss history"""
        plt.figure(figsize=(10, 6))
        plt.plot(history['epoch'], history['loss'])
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
    
    def plot_predictions(self, X_test: np.ndarray, y_test: np.ndarray, 
                        predictions: np.ndarray = None):
        """Plot actual vs predicted values"""
        if predictions is None:
            predictions = self.model.predict(X_test)
            predictions = np.array(predictions).flatten()
        
        # Inverse transform
        y_test_actual = self.label_scaler.inverse_transform(y_test)
        predictions_actual = self.label_scaler.inverse_transform(predictions.reshape(-1, 1))
        
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_actual, label='Actual', alpha=0.7)
        plt.plot(predictions_actual, label='Predicted', alpha=0.7)
        plt.title('Bitcoin Price: Actual vs Predicted')
        plt.xlabel('Time')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()


def main():
    """Main function to demonstrate Bitcoin prediction"""
    # Initialize predictor
    predictor = BitcoinPredictor()
    
    try:
        # Fetch data
        df = predictor.fetch_bitcoin_data()
        
        # Prepare data
        X_train, X_test, y_train, y_test = predictor.prepare_data(df)
        
        # Train model
        history = predictor.train_model(X_train, y_train, epochs=500)
        
        # Evaluate model
        metrics = predictor.evaluate_model(X_test, y_test)
        print("\nModel Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Plot results
        predictor.plot_training_history(history)
        predictor.plot_predictions(X_test, y_test)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have internet connection for fetching data.")


if __name__ == "__main__":
    main() 