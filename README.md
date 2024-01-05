# Bitcoin Price Prediction using GRU Neural Network

A comprehensive machine learning system for predicting Bitcoin prices using Gated Recurrent Unit (GRU) neural networks. This project fetches real-time Bitcoin data from Binance and implements a custom GRU architecture for time series prediction.

## 🚀 Features

- **Real-time Data Fetching**: Automatically fetches Bitcoin price data from Binance API
- **Custom GRU Implementation**: Hand-crafted GRU neural network from scratch
- **Advanced Preprocessing**: Feature scaling and data preparation
- **Comprehensive CLI**: Command-line interface for all operations
- **Visualization**: Interactive plots and analysis tools
- **Model Evaluation**: Multiple performance metrics and validation

## 📋 Prerequisites

- Python 3.8 or higher
- Internet connection for fetching data
- Binance API access (optional, for enhanced features)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/bitcoin-ml-predictor.git
   cd bitcoin-ml-predictor
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package** (optional):
   ```bash
   pip install -e .
   ```

## 🚀 Quick Start

### Using the CLI

1. **Fetch Bitcoin data**:
   ```bash
   python cli.py fetch-data
   ```

2. **Train the model**:
   ```bash
   python cli.py train
   ```

3. **Run complete demo**:
   ```bash
   python cli.py demo
   ```

### Using the Python API

```python
from bitcoin_predictor import BitcoinPredictor

# Initialize predictor
predictor = BitcoinPredictor()

# Fetch data
df = predictor.fetch_bitcoin_data()

# Prepare data
X_train, X_test, y_train, y_test = predictor.prepare_data(df)

# Train model
history = predictor.train_model(X_train, y_train, epochs=500)

# Evaluate model
metrics = predictor.evaluate_model(X_test, y_test)
print(metrics)
```

## 📚 Usage Examples

### Fetching Data
```bash
# Fetch BTCUSDT data from 2019
python cli.py fetch-data --symbol BTCUSDT --start-date "1 Jan 2019"

# Fetch data for different cryptocurrency
python cli.py fetch-data --symbol ETHUSDT --start-date "1 Jan 2020"
```

### Training Models
```bash
# Train with default parameters
python cli.py train

# Train with custom parameters
python cli.py train --hidden-size 200 --learning-rate 0.005 --epochs 1000
```

### Analysis and Visualization
```bash
# Analyze Bitcoin data
python cli.py analyze

# Create visualization plots
python cli.py plot
```

## 🏗️ Project Structure

```
bitcoin-ml-predictor/
├── bitcoin_predictor.py      # Main prediction module
├── cli.py                   # Command-line interface
├── notebook.ipynb           # Original Jupyter notebook
├── requirements.txt          # Python dependencies
├── setup.py                # Package setup
├── README.md               # This file
└── LICENSE                 # MIT License
```

## 📊 Model Architecture

### GRU Neural Network
The model implements a custom Gated Recurrent Unit with:

- **Input Layer**: Processes OHLCV (Open, High, Low, Close, Volume) data
- **Hidden Layer**: GRU cells with configurable hidden size
- **Output Layer**: Single neuron for price prediction
- **Optimizer**: RMSprop with gradient clipping

### Data Preprocessing
- **Feature Scaling**: StandardScaler for input features
- **Label Scaling**: StandardScaler for target values
- **Train/Test Split**: 80/20 split with time series consideration

### Training Process
- **Loss Function**: Mean Squared Error (MSE)
- **Optimization**: RMSprop with learning rate scheduling
- **Regularization**: Gradient clipping to prevent exploding gradients
- **Validation**: Real-time loss monitoring

## 📈 Performance Metrics

The model provides comprehensive evaluation metrics:

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **Actual Price Metrics**: Metrics on real USD values

## 🔗 API Reference

### BitcoinPredictor Class

#### Methods

- `fetch_bitcoin_data(symbol, interval, start_date)`: Fetch historical data
- `prepare_data(df, train_cols, label_col, test_size)`: Prepare data for training
- `train_model(X_train, y_train, hidden_size, learning_rate, epochs)`: Train the model
- `evaluate_model(X_test, y_test)`: Evaluate model performance
- `plot_training_history(history)`: Plot training loss
- `plot_predictions(X_test, y_test)`: Plot actual vs predicted values

### GRU Class

#### Methods

- `forward(inputs)`: Forward pass through the network
- `train_model(inputs, targets)`: Train the model
- `predict(inputs)`: Make predictions

## 🛡️ Security Considerations

- **API Keys**: Store Binance API keys securely in environment variables
- **Data Privacy**: No personal data is collected or stored
- **Rate Limiting**: Respect Binance API rate limits
- **Error Handling**: Comprehensive error handling for network issues

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `make test`
5. Format code: `make format`
6. Commit changes: `git commit -am 'Add feature'`
7. Push to branch: `git push origin feature-name`
8. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Ayush Agrawal**
- Email: ayushagrwal031220@gmail.com
- GitHub: [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- Binance for providing cryptocurrency data API
- NumPy and Pandas for data manipulation
- Matplotlib and Seaborn for visualization
- Scikit-learn for preprocessing utilities

## 📈 Roadmap

- [ ] Support for multiple cryptocurrencies
- [ ] Advanced technical indicators
- [ ] Real-time prediction API
- [ ] Web interface using Flask/FastAPI
- [ ] Ensemble methods (LSTM + GRU)
- [ ] Sentiment analysis integration
- [ ] Trading strategy recommendations
- [ ] Docker containerization
- [ ] CI/CD pipeline setup

## ⚠️ Disclaimer

This project is for educational and research purposes only. Cryptocurrency trading involves significant risk, and past performance does not guarantee future results. Always do your own research and consider consulting with financial advisors before making investment decisions.

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/bitcoin-ml-predictor/issues) page
2. Create a new issue with detailed information
3. Contact the author directly

---

**Happy Predicting! 🚀📈** 