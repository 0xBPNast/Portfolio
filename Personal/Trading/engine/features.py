from dataclasses import dataclass
import pandas as pd

@dataclass
class FeatureSet:
    df: pd.DataFrame  # indexed by datetime, columns = features

class FeatureBuilder:
    def build(self, market_data: pd.DataFrame) -> FeatureSet:
        raise NotImplementedError
