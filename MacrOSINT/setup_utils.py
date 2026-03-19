#!/usr/bin/env python3
"""Project setup utility.

Installs dependencies, configures paths and downloads required data files."""
import subprocess
import sys
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    # Try to find project root from current working directory
    cwd = Path.cwd()

    # Check if we're in the project root
    if (cwd / "MacrOSINT").exists():
        return cwd

    # Check if we're inside MacrOSINT package
    if cwd.name == "MacrOSINT":
        return cwd.parent

    # Default to current directory
    return cwd


PROJECT_ROOT = get_project_root()
ENV_FILE = PROJECT_ROOT / ".env"


def install_dependencies() -> None:
    """Install Python packages from requirements."""
    requirements_file = PROJECT_ROOT / "requirements.txt"
    if requirements_file.exists():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
    else:
        print(f"Warning: requirements.txt not found at {requirements_file}")


def install_download_tools() -> None:
    """Install gdown for data downloads."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])


def setup_paths() -> None:
    """Create data directories and write path variables to .env."""
    data_dir = PROJECT_ROOT / "data"
    market_dir = data_dir / "market"
    cot_dir = data_dir / "cot"

    data_dir.mkdir(parents=True, exist_ok=True)
    market_dir.mkdir(parents=True, exist_ok=True)
    cot_dir.mkdir(parents=True, exist_ok=True)

    env_lines = [
        f"data_path={data_dir}",
        f"MARKET_DATA_PATH={market_dir}",
        f"COT_PATH={cot_dir}",
        f"APP_PATH={PROJECT_ROOT}",
        "# Add your API keys below:",
        "# NASS_TOKEN=your_nass_api_key",
        "# FAS_TOKEN=your_fas_api_key",
        "# EIA_API_KEY=your_eia_api_key",
        "# NCEI_TOKEN=your_ncei_token",
    ]

    if not ENV_FILE.exists():
        ENV_FILE.write_text("\n".join(env_lines) + "\n")
        print(f"Created .env file at {ENV_FILE}")
        print("Please add your API keys to the .env file.")
    else:
        print(f".env file already exists at {ENV_FILE}")


def download_h5() -> None:
    """Download .h5 files from the project Drive folder into the data path."""
    try:
        import gdown
    except ImportError:
        print("gdown not installed. Installing...")
        install_download_tools()
        import gdown

    url = "https://drive.google.com/drive/folders/1PM5dv-Acy7fgVPLQvOsmDxcRis7somiC?usp=drive_link"
    output = PROJECT_ROOT / "data"

    print(f"Downloading data files to {output}...")
    try:
        gdown.download_folder(url, output=str(output), quiet=False, use_cookies=False)
        print("Data files downloaded successfully.")
    except Exception as e:
        print(f"Error downloading data files: {e}")
        print("You may need to download the data files manually.")


def main() -> None:
    """Run the complete setup process."""
    print("Starting MacrOSINT setup...")
    print(f"Project root: {PROJECT_ROOT}")

    print("\n1. Installing dependencies...")
    install_dependencies()

    print("\n2. Setting up paths and .env file...")
    setup_paths()

    print("\n3. Downloading data files (optional, may take time)...")
    download_choice = input("Download data files from Google Drive? (y/n): ")
    if download_choice.lower() == 'y':
        download_h5()
    else:
        print("Skipping data download. You can run 'macrosint-setup' again later to download.")

    print("\nSetup complete!")
    print(f"Please configure your API keys in {ENV_FILE}")


if __name__ == "__main__":
    main()