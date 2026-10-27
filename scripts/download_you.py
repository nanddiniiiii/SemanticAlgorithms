"""Monitor/retrieve your saved notebook version; run using your Kaggle identity."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semantic_lossless.team_suite.monitor import main
if __name__=='__main__':main('you')
