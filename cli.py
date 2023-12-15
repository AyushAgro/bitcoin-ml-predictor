"""
Command Line Interface for Bitcoin Price Prediction System
"""

import click
import json
import os
from bitcoin_predictor import BitcoinPredictor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


@click.group()
def cli():
    """Bitcoin Price Prediction CLI"""
    pass


@cli.command()
@click.option('--symbol', default='BTCUSDT', help='Trading pair symbol')
@click.option('--start-date', default='1 Jan 2019', help='Start date for data')
@click.option('--output', default='bitcoin_data.csv', help='Output CSV file')
def fetch_data(symbol, start_date, output):
    """Fetch Bitcoin historical data from Binance"""
    try:
        predictor = BitcoinPredictor()
        df = predictor.fetch_bitcoin_data(symbol, start_date=start_date)
        df.to_csv(output, index=False)
        click.echo(f"✅ Data saved to {output}")
        click.echo(f"📊 Fetched {len(df)} records")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
@click.option('--data-file', default='bitcoin_data.csv', help='Input CSV file')
@click.option('--hidden-size', default=100, help='Hidden layer size')
@click.option('--learning-rate', default=0.01, help='Learning rate')
@click.option('--epochs', default=500, help='Number of training epochs')
@click.option('--test-size', default=0.2, help='Test set size')
def train(data_file, hidden_size, learning_rate, epochs, test_size):
    """Train the Bitcoin prediction model"""
    try:
        if not os.path.exists(data_file):
            click.echo(f"❌ Data file {data_file} not found. Run fetch-data first.")
            return
            
        predictor = BitcoinPredictor()
        
        # Load data
        df = pd.read_csv(data_file)
        click.echo(f"📊 Loaded {len(df)} records from {data_file}")
        
        # Prepare data
        X_train, X_test, y_train, y_test = predictor.prepare_data(df, test_size=test_size)
        click.echo(f"📈 Training set: {len(X_train)} samples")
        click.echo(f"📊 Test set: {len(X_test)} samples")
        
        # Train model
        history = predictor.train_model(
            X_train, y_train, 
            hidden_size=hidden_size, 
            learning_rate=learning_rate, 
            epochs=epochs
        )
        
        # Evaluate model
        metrics = predictor.evaluate_model(X_test, y_test)
        click.echo("\n📊 Model Performance:")
        for metric, value in metrics.items():
            click.echo(f"  {metric}: {value:.4f}")
        
        # Save model info
        model_info = {
            'hidden_size': hidden_size,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'metrics': metrics,
            'history': history
        }
        
        with open('model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
            
        click.echo("✅ Model training completed!")
        click.echo("📁 Model info saved to model_info.json")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
@click.option('--data-file', default='bitcoin_data.csv', help='Input CSV file')
def predict(data_file):
    """Make predictions using trained model"""
    try:
        if not os.path.exists(data_file):
            click.echo(f"❌ Data file {data_file} not found.")
            return
            
        predictor = BitcoinPredictor()
        
        # Load data
        df = pd.read_csv(data_file)
        
        # Prepare data
        X_train, X_test, y_train, y_test = predictor.prepare_data(df)
        
        # Train model (if not already trained)
        if predictor.model is None:
            click.echo("🤖 Training model first...")
            predictor.train_model(X_train, y_train, epochs=100)
        
        # Make predictions
        predictions = predictor.model.predict(X_test)
        predictions = predictor.label_scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        )
        
        # Save predictions
        results_df = pd.DataFrame({
            'actual': predictor.label_scaler.inverse_transform(y_test).flatten(),
            'predicted': predictions.flatten()
        })
        
        results_df.to_csv('predictions.csv', index=False)
        click.echo("✅ Predictions saved to predictions.csv")
        
        # Show sample predictions
        click.echo("\n📊 Sample Predictions:")
        click.echo(results_df.head(10).to_string(index=False))
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
@click.option('--data-file', default='bitcoin_data.csv', help='Input CSV file')
def analyze(data_file):
    """Analyze Bitcoin data and show statistics"""
    try:
        if not os.path.exists(data_file):
            click.echo(f"❌ Data file {data_file} not found.")
            return
            
        df = pd.read_csv(data_file)
        
        click.echo("📊 Bitcoin Data Analysis:")
        click.echo(f"  📅 Date range: {df['date'].min()} to {df['date'].max()}")
        click.echo(f"  📈 Total records: {len(df)}")
        click.echo(f"  💰 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        click.echo(f"  📊 Average price: ${df['close'].mean():.2f}")
        click.echo(f"  📈 Volume range: {df['volume'].min():.2f} - {df['volume'].max():.2f}")
        
        # Show recent prices
        click.echo("\n📈 Recent Prices:")
        recent = df[['date', 'open', 'high', 'low', 'close', 'volume']].tail(5)
        click.echo(recent.to_string(index=False))
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
@click.option('--data-file', default='bitcoin_data.csv', help='Input CSV file')
def plot(data_file):
    """Plot Bitcoin price data"""
    try:
        if not os.path.exists(data_file):
            click.echo(f"❌ Data file {data_file} not found.")
            return
            
        df = pd.read_csv(data_file)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Price over time
        axes[0, 0].plot(df['date'], df['close'])
        axes[0, 0].set_title('Bitcoin Price Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price (USD)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Volume over time
        axes[0, 1].plot(df['date'], df['volume'])
        axes[0, 1].set_title('Trading Volume Over Time')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Volume')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Price distribution
        axes[1, 0].hist(df['close'], bins=50, alpha=0.7)
        axes[1, 0].set_title('Price Distribution')
        axes[1, 0].set_xlabel('Price (USD)')
        axes[1, 0].set_ylabel('Frequency')
        
        # OHLC box plot
        price_data = [df['open'], df['high'], df['low'], df['close']]
        axes[1, 1].boxplot(price_data, labels=['Open', 'High', 'Low', 'Close'])
        axes[1, 1].set_title('Price Statistics')
        axes[1, 1].set_ylabel('Price (USD)')
        
        plt.tight_layout()
        plt.savefig('bitcoin_analysis.png', dpi=300, bbox_inches='tight')
        click.echo("✅ Plot saved as bitcoin_analysis.png")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
def demo():
    """Run a complete demo of the Bitcoin prediction system"""
    try:
        click.echo("🚀 Starting Bitcoin Prediction Demo...")
        
        # Step 1: Fetch data
        click.echo("\n📊 Step 1: Fetching Bitcoin data...")
        predictor = BitcoinPredictor()
        df = predictor.fetch_bitcoin_data()
        
        # Step 2: Prepare data
        click.echo("\n📈 Step 2: Preparing data...")
        X_train, X_test, y_train, y_test = predictor.prepare_data(df)
        
        # Step 3: Train model
        click.echo("\n🤖 Step 3: Training model...")
        history = predictor.train_model(X_train, y_train, epochs=100)
        
        # Step 4: Evaluate
        click.echo("\n📊 Step 4: Evaluating model...")
        metrics = predictor.evaluate_model(X_test, y_test)
        
        click.echo("\n📈 Results:")
        for metric, value in metrics.items():
            click.echo(f"  {metric}: {value:.4f}")
        
        click.echo("\n✅ Demo completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Demo failed: {e}")


if __name__ == '__main__':
    cli() 