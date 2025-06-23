from dataclasses import dataclass
from pathlib import Path
import pickle
import pandas as pd

@dataclass
class TimeSeriesData:
    data: pd.DataFrame
    file_path: Path
    saved_path:Path
