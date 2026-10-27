"""Monitor/retrieve teammate two's saved notebook version using their identity."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.launch import main
if __name__=='__main__':main('3')
