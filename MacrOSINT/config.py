import os
from pathlib import Path
from dotenv import load_dotenv
import sys
# Load environment variables
load_dotenv('.env')

# Project paths

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = Path("F:", 'Macro', 'OSINT')
ASSETS_DIR = PROJECT_ROOT / "assets"
COMPONENTS_DIR = PROJECT_ROOT / "components"
DOT_ENV = PROJECT_ROOT / ".env"

# Data paths from environment or defaults
DATA_PATH = os.getenv('data_path', str(DATA_DIR))
MARKET_DATA_PATH = os.getenv('market_data_path', Path("F:", "data", "market_data.hd5"))
COT_PATH = os.getenv('cot_path', str(DATA_DIR / 'cot'))
APP_PATH = os.getenv('APP_PATH', str(PROJECT_ROOT))

# API Keys
NASS_TOKEN = os.getenv('NASS_TOKEN', '')
FAS_TOKEN = os.getenv('FAS_TOKEN', '')
EIA_KEY = os.getenv('EIA_API_KEY', '')
NCEI_TOKEN = os.getenv('NCEI_TOKEN', '')

# Create directories if they don't exist
for path in [DATA_DIR, Path(DATA_PATH), Path(MARKET_DATA_PATH), Path(COT_PATH)]:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

# Mapping file paths
MAPPINGS = {
    'eia': PROJECT_ROOT / 'data' / 'sources' / 'eia' / 'data_mapping.toml',
    'usda': PROJECT_ROOT / 'data' / 'sources' / 'usda' / 'data_mapping.toml',
    'chart': COMPONENTS_DIR / 'plotting' / 'chart_mappings.toml',
    'cot': PROJECT_ROOT / 'data' / 'sources' / 'COT' / 'futures_mappings.toml'
}

def get_mapping_file(name: str) -> Path:
    """Get the path to a mapping file"""
    return MAPPINGS.get(name, Path())

def load_mapping(name: str) -> dict:
    """Load a TOML mapping file"""
    import toml
    mapping_file = get_mapping_file(name)
    if mapping_file.exists():
        with open(mapping_file) as f:
            return toml.load(f)
    return {}