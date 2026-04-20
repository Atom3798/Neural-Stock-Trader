#!/usr/bin/env python3
"""
Run the NeuralStockTrader Frontend
Starts the Streamlit web application
"""

import subprocess
import sys
from pathlib import Path


def run_frontend():
    """Run the Streamlit frontend"""
    
    # Check if frontend.py exists
    frontend_path = Path("frontend.py")
    if not frontend_path.exists():
        print("❌ frontend.py not found!")
        sys.exit(1)
    
    print("🚀 Starting NeuralStockTrader Frontend...")
    print("=" * 60)
    print("📊 Dashboard: http://localhost:8501")
    print("=" * 60)
    print()
    
    try:
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "frontend.py",
            "--logger.level=info",
            "--client.showErrorDetails=true"
        ])
    except KeyboardInterrupt:
        print("\n\n✋ Frontend stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_frontend()
