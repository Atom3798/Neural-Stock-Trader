# 🚀 NeuralStockTrader Frontend - Quick Start Guide

Get your neural network stock trading dashboard up and running in 5 minutes!

## ⚡ Quick Start (Windows)

### Option 1: Click to Run (Easiest)
1. Double-click **`run_frontend.bat`** in the project folder
2. Wait for dependencies to install (first time only)
3. Your browser opens automatically at `http://localhost:8501`
4. Start trading! 📈

### Option 2: Command Line
```bash
# Open command prompt in project folder and type:
pip install -r frontend_requirements.txt
streamlit run frontend.py
```

---

## ⚡ Quick Start (Linux/Mac)

### Option 1: Shell Script
```bash
# Make script executable
chmod +x run_frontend.sh

# Run it
./run_frontend.sh
```

### Option 2: Command Line
```bash
pip install -r frontend_requirements.txt
streamlit run frontend.py
```

---

## 📊 Dashboard Overview

Once the dashboard opens, you'll see:

### 1. **Left Sidebar** 🗂️
- Navigation menu with 8 pages
- Market overview
- Quick settings
- Trading mode selector

### 2. **Top Pages** (Click in sidebar to navigate)

| Page | What It Does |
|------|-------------|
| 📊 Dashboard | View portfolio metrics, positions, and recent trades |
| 📈 Backtest | Test strategies on historical data |
| 🤖 Model Training | Train neural network models |
| 💹 Live Trading | Monitor and execute live trades |
| 📉 Portfolio | Analyze asset allocation and holdings |
| ⚙️ Risk Management | Configure risk controls and limits |
| 📋 History | View trading history and performance |
| ⚙️ Settings | Configure API keys and preferences |

---

## 🎯 Getting Started: Step-by-Step

### Step 1: Explore the Dashboard
1. Open the frontend (see Quick Start above)
2. You're on the **Dashboard** page by default
3. Explore the metrics cards at the top
4. Scroll down to see portfolio charts and positions

### Step 2: Run Your First Backtest
1. Click **📈 Backtest** in the sidebar
2. Select a stock symbol (e.g., AAPL)
3. Pick a date range (e.g., 2023-01-01 to 2024-01-01)
4. Choose a strategy (e.g., "ML Ensemble")
5. Click **▶️ Run Backtest**
6. View the equity curve and results!

### Step 3: Train a Model
1. Click **🤖 Model Training** in the sidebar
2. Select model architecture (LSTM, GRU, Transformer, Ensemble)
3. Adjust hyperparameters if desired
4. Click **🚀 Start Training**
5. Watch the training progress and loss metrics

### Step 4: Monitor Live Trading
1. Click **💹 Live Trading** in the sidebar
2. Review market data and trading signals
3. Click **▶️ Start Trading** to begin (demo mode)
4. Watch positions, P&L, and account balance update

---

## 🎨 Key Features

### 📈 Real-Time Charts
- Interactive Plotly charts
- Hover for detailed information
- Zoom and pan capabilities
- Download as PNG

### 📊 Performance Metrics
- **Returns**: Total and annualized performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

### 💰 Portfolio Management
- Asset allocation pie chart
- Top holdings breakdown
- Diversification analysis
- Performance by sector

### ⚠️ Risk Management
- Position size limits
- Stop-loss & take-profit controls
- Daily loss limits
- VaR calculations

---

## 🔧 Common Tasks

### Change Trading Symbol
In sidebar, select from dropdown under "Trading Mode"

### Configure Risk Parameters
1. Go to **⚙️ Risk Management**
2. Adjust position limits
3. Set stop-loss percentage
4. Click **✅ Update Risk Parameters**

### Connect Alpaca API
1. Go to **⚙️ Settings** → **API** tab
2. Enter your Alpaca API Key
3. Enter your API Secret
4. Click **✅ Test Connection**

### View Trading History
1. Go to **📋 History**
2. See all executed trades
3. Analyze monthly performance
4. View cumulative returns

### Enable Notifications
1. Go to **⚙️ Settings** → **Notifications** tab
2. Check "Email Notifications"
3. Enter your email address
4. Select which alerts you want
5. Click **✅ Save**

---

## 💾 Saving Data

The frontend automatically saves:
- ✅ Settings (in `frontend_config.json`)
- ✅ Backtest results (in `results/` folder)
- ✅ Training metrics (in `logs/` folder)
- ✅ Trading history (in database)

---

## 🆘 Troubleshooting

### Dashboard won't load?
1. Check Python is installed: `python --version`
2. Activate virtual environment: 
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```
3. Reinstall dependencies:
   ```bash
   pip install -r frontend_requirements.txt
   ```

### Port 8501 already in use?
```bash
streamlit run frontend.py --server.port 8502
```

### Cannot see trading engine data?
1. Ensure trading engine is initialized
2. Check `config/config.yaml` exists
3. Verify data in `data/` folder
4. Check logs in `logs/` folder

### Charts not displaying?
1. Clear Streamlit cache:
   ```bash
   streamlit cache clear
   ```
2. Refresh browser (Ctrl+R or Cmd+R)
3. Restart frontend application

---

## 📱 Viewing on Different Devices

### On Your Computer
```
http://localhost:8501
```

### On Your Phone/Tablet (Same Network)
1. Find your computer's IP address:
   - Windows: `ipconfig` → look for "IPv4 Address"
   - Mac/Linux: `ifconfig` → look for "inet"
2. On your phone, open browser and go to:
   ```
   http://YOUR_COMPUTER_IP:8501
   ```

---

## 🔐 Security Notes

For development/testing:
- Default settings are not encrypted
- API keys should be in environment variables in production
- Use HTTPS when accessing remotely

To use environment variables for API keys:
```python
import os
api_key = os.getenv('ALPACA_API_KEY')
api_secret = os.getenv('ALPACA_API_SECRET')
```

---

## 📚 Next Steps

1. **Explore all pages** to understand available features
2. **Run backtests** on different strategies
3. **Train models** with various architectures
4. **Configure risk limits** for your trading style
5. **Read FRONTEND_README.md** for advanced features

---

## 🎓 Tips for Best Experience

✅ **DO:**
- Start with backtesting before live trading
- Monitor risk metrics regularly
- Save important configurations
- Check trading logs for insights
- Use multiple strategies

❌ **DON'T:**
- Skip risk configuration
- Trade with money you can't afford to lose
- Ignore warning messages
- Leave trading running unattended
- Use default settings for live trading

---

## 📞 Help & Support

Need help? Check:
1. [FRONTEND_README.md](./FRONTEND_README.md) - Full documentation
2. [README.md](./README.md) - Main project documentation
3. [Streamlit Docs](https://docs.streamlit.io) - Framework help
4. Application logs in `logs/` folder

---

## 🎉 You're Ready!

Your NeuralStockTrader frontend is now ready to use! 

**Quick reminder:**
- Run `run_frontend.bat` (Windows) or `./run_frontend.sh` (Linux/Mac)
- Open `http://localhost:8501`
- Start exploring and trading! 📈

---

**Happy Trading! 🚀📊**

For questions or feature requests, refer to the [FRONTEND_README.md](./FRONTEND_README.md) documentation.
