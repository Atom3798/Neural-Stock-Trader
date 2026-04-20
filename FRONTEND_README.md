# NeuralStockTrader Web Frontend

A rich, interactive web-based dashboard for managing and monitoring neural network-powered stock trading operations.

## 📊 Features

### Dashboard
- **Real-time Metrics**: Portfolio returns, win rate, Sharpe ratio, and more
- **Portfolio Value Chart**: Visual representation of account growth over time
- **Strategy Performance**: Compare performance across different trading strategies
- **Active Positions**: Monitor open trades with real-time P&L
- **Recent Trades**: Quick view of latest executed trades

### Backtesting
- **Strategy Configuration**: Easy-to-use interface for backtesting setup
- **Multi-Strategy Support**: Test multiple strategies simultaneously
- **Risk Controls**: Set position sizing, stop-loss, and other parameters
- **Detailed Results**: Comprehensive metrics including Sharpe ratio, max drawdown, win rate
- **Equity Curve**: Visual representation of backtest performance over time
- **Trade Statistics**: Detailed analysis of winning and losing trades

### Model Training
- **Neural Network Configuration**: Select and configure different architectures (LSTM, GRU, Transformer, Ensemble)
- **Hyperparameter Tuning**: Adjust learning rates, dropout, batch size, and more
- **Training Progress**: Real-time monitoring of training metrics
- **Loss Tracking**: Monitor training and validation loss convergence
- **Model Management**: Save and load trained models

### Live Trading
- **Trading Control**: Start/stop trading with a single click
- **Market Data**: Real-time prices and trading signals
- **Account Monitor**: Track account balance, open P&L, and active positions
- **Signal Confidence**: View model confidence levels for trading signals
- **Live Charts**: Real-time price charts with technical indicators

### Portfolio Management
- **Asset Allocation**: Pie chart view of portfolio composition
- **Top Holdings**: Detailed breakdown of your largest positions
- **Performance Tracking**: Monitor asset performance and allocation percentages
- **Diversification Analysis**: Ensure proper portfolio diversification

### Risk Management
- **Risk Metrics**: VaR, Beta, correlation, and volatility monitoring
- **Position Limits**: Set maximum position sizes and sector allocations
- **Stop Loss & Take Profit**: Configure automatic order management
- **Daily Limits**: Set maximum daily losses and drawdown limits
- **Sector Risk**: Monitor risk exposure by sector
- **Risk Control Panel**: Centralized risk parameter management

### Trading History
- **Trade Log**: Complete history of all executed trades
- **Monthly Performance**: Track performance by month
- **Cumulative Returns**: Visualize long-term return accumulation
- **Trade Analysis**: Detailed statistics on wins/losses

### Settings
- **General Settings**: Configure trading symbols, update frequency, timezone
- **API Configuration**: Manage Alpaca API keys and data providers
- **Notifications**: Set up email, SMS, and push alerts
- **Advanced Settings**: Log levels, backtesting cache, file logging options

## 🚀 Getting Started

### Installation

#### Windows
```bash
# Run the launcher script (easiest method)
run_frontend.bat

# Or manually:
pip install -r frontend_requirements.txt
streamlit run frontend.py
```

#### Linux/Mac
```bash
# Make script executable and run
chmod +x run_frontend.sh
./run_frontend.sh

# Or manually:
pip install -r frontend_requirements.txt
streamlit run frontend.py
```

### Access the Dashboard

Once running, open your browser and navigate to:
```
http://localhost:8501
```

## 📁 Project Structure

```
NeuralStockTrader/
├── frontend.py              # Main Streamlit application
├── frontend_api.py          # API wrapper for trading engine integration
├── frontend_requirements.txt # Python dependencies for frontend
├── run_frontend.py          # Python launcher script
├── run_frontend.bat         # Windows batch launcher
├── run_frontend.sh          # Linux/Mac shell launcher
└── FRONTEND_README.md       # This file
```

## 🛠️ Architecture

### Components

1. **frontend.py** - Main UI application
   - Multi-page dashboard using Streamlit
   - Interactive charts with Plotly
   - Real-time data visualization
   - User input handling and validation

2. **frontend_api.py** - Backend integration layer
   - `TradingAPI` class provides interface to trading engine
   - Methods for dashboard, backtesting, training, portfolio, and risk management
   - Session management and data caching
   - Simulated data generation for demo

3. **Streamlit Framework**
   - Rapid web app development
   - Interactive widgets and forms
   - Real-time chart updates
   - Session state management

### Data Flow

```
User Interface (Streamlit)
        ↓
Frontend Widgets & Forms
        ↓
frontend_api.py (TradingAPI)
        ↓
Trading Engine (src/execution_layer)
        ↓
Database / File Storage
```

## 💻 Usage Examples

### Access Dashboard
1. Start the frontend using one of the launcher scripts
2. Browser automatically opens or navigate to `http://localhost:8501`
3. View real-time metrics and portfolio status

### Run a Backtest
1. Navigate to "📈 Backtest" page
2. Select symbol and date range
3. Choose trading strategies
4. Adjust risk parameters
5. Click "▶️ Run Backtest"
6. Review equity curve and detailed results

### Train a Model
1. Go to "🤖 Model Training" page
2. Select neural network architecture
3. Configure hyperparameters
4. Set dataset splits
5. Click "🚀 Start Training"
6. Monitor real-time loss metrics

### Monitor Live Trading
1. Navigate to "💹 Live Trading"
2. Select trading strategy
3. Click "▶️ Start Trading"
4. Monitor market data and signals
5. Track active positions in real-time

## 📊 Key Metrics Explained

- **Total Returns**: Overall profit/loss percentage since inception
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Sortino Ratio**: Similar to Sharpe but only penalizes downside volatility
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **VaR (Value at Risk)**: Estimated maximum loss at 95% confidence

## ⚙️ Configuration

### General Settings
- **Trading Symbols**: Default stocks to monitor (AAPL, MSFT, GOOGL, etc.)
- **Update Frequency**: How often to refresh market data (1 min to 1 hour)
- **Timezone**: Set your preferred timezone (EST, UTC, etc.)

### API Settings
- **Alpaca API**: Connect to live trading via Alpaca
- **Data Provider**: Choose between Yahoo Finance, Alpha Vantage, or IEX Cloud

### Notification Settings
- **Email**: Receive trade confirmations and alerts
- **SMS**: Get critical alerts via text message
- **Push**: Browser push notifications for important events

## 🔧 Advanced Features

### Backtesting Cache
Enable to speed up repeated backtests by caching historical data

### File Logging
Log all system events and trading activity to files for audit trails

### Advanced Log Levels
- DEBUG: Most verbose, show all details
- INFO: Normal logging
- WARNING: Show warnings and errors only
- ERROR: Only show errors

## 🎨 Customization

### Themes & Styling
The frontend uses custom CSS for a modern gradient design. Modify the CSS in `frontend.py` to customize colors and styling.

### Adding Custom Pages
1. Create a new render function in `frontend.py`
2. Add it to the sidebar navigation
3. Call the function in the main routing logic

### Extending API
1. Add new methods to `TradingAPI` class in `frontend_api.py`
2. Call these methods from frontend pages
3. Update UI components as needed

## 📦 Dependencies

### Core
- **streamlit**: Web app framework
- **plotly**: Interactive charting
- **pandas**: Data manipulation
- **numpy**: Numerical computing

### Optional
- **streamlit-option-menu**: Enhanced navigation
- **streamlit-authenticator**: User authentication
- **streamlit-echarts**: Additional chart types

## 🐛 Troubleshooting

### Port Already in Use
If port 8501 is already in use:
```bash
streamlit run frontend.py --server.port 8502
```

### Dependencies Installation Issues
Clear pip cache and reinstall:
```bash
pip cache purge
pip install -r frontend_requirements.txt --force-reinstall
```

### Streamlit Not Found
Make sure you're in the virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Cannot Connect to Trading Engine
1. Verify the trading engine is properly initialized
2. Check `config/config.yaml` is correctly configured
3. Review logs for detailed error messages

## 📝 Logs

Application logs are stored in:
- Default logs directory: `logs/`
- Frontend logs: `logs/frontend.log`
- Trading logs: `logs/trading.log`

## 🔐 Security

- API keys are stored securely (use environment variables in production)
- All network traffic should be HTTPS in production
- Consider implementing user authentication for multi-user access
- Regularly backup trading data and model files

## 📈 Performance Tips

1. **Cache Backtest Data**: Enable in settings to speed up repeated tests
2. **Reduce Chart Resolution**: Show fewer data points for faster rendering
3. **Use Lighter Data**: Limit historical data range for faster loading
4. **Optimize Browser**: Use a modern browser for best performance

## 🤝 Integration

### Connecting to Live Trading
The frontend is designed to integrate with:
- Alpaca Trading API
- Interactive Brokers (via extension)
- TD Ameritrade (via extension)

### Data Sources
Currently supports:
- Yahoo Finance (free)
- Alpaca API (with account)
- IEX Cloud (with API key)

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Plotly Documentation](https://plotly.com/python)
- [NeuralStockTrader Main README](./README.md)
- [Trading Engine Documentation](./src/execution_layer/README.md)

## 🎯 Future Enhancements

- [ ] Dark/Light theme toggle
- [ ] Multi-user support with authentication
- [ ] Advanced charting with technical indicators
- [ ] Strategy builder UI (no-code)
- [ ] Mobile-responsive design
- [ ] WebSocket support for real-time data
- [ ] Database integration for historical data
- [ ] Export reports as PDF/Excel
- [ ] Performance comparison tools
- [ ] Risk heat maps

## 📞 Support

For issues or feature requests:
1. Check the troubleshooting section
2. Review trading engine logs
3. Ensure all dependencies are properly installed
4. Verify configuration files are correct

---

**Happy Trading! 📈🚀**
