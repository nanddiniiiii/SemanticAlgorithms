"""Monitor/retrieve your saved notebook version; run using your Kaggle identity."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
if __name__=='__main__':
    if '--campaign' in sys.argv:
        sys.argv.remove('--campaign')
        from src.person1.campaign import main
        main()
    else:
        from src.launch import main
        main('1')
