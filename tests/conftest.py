import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
warnings.filterwarnings("ignore", category=DeprecationWarning)

RAW_DATA = ROOT / "data" / "raw" / "HR_Attrition_Dataset.csv"
PROCESSED = ROOT / "data" / "processed"


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA)


@pytest.fixture(scope="session")
def app_module():
    import app

    return app
