import pandas as pd
from typing import Dict, Any
import json

def export_to_file(data: Dict[str, Any], filename: str, format: str = 'json'):
    """
    Export data to a file in the specified format
    """
    if format == 'json':
        with open(f"{filename}.json", 'w') as f:
            json.dump(data, f, indent=4)
    elif format == 'csv':
        pd.DataFrame(data).to_csv(f"{filename}.csv", index=False)
    elif format == 'xlsx':
        pd.DataFrame(data).to_excel(f"{filename}.xlsx", index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
