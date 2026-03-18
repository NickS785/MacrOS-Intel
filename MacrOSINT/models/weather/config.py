"""
Weather Module Configuration
============================
Configuration settings for the weather module.
"""

from pathlib import Path

# Import from parent config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from MacrOSINT.config import NCEI_TOKEN

# Re-export for backward compatibility
__all__ = ['NCEI_TOKEN']