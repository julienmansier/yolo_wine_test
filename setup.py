"""
Quick setup script for YOLO Wine Bottle Detector.

This script helps you get started by:
1. Creating a virtual environment
2. Installing dependencies
3. Downloading a sample test image
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{description}...")
    try:
        subprocess.run(cmd, check=True, shell=True)
        print(f"✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        return False


def main():
    print("=" * 60)
    print("YOLO Wine Bottle Detector - Setup")
    print("=" * 60)

    project_root = Path(__file__).parent

    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        return 1

    print(f"Python version: {sys.version}")

    # Ask user what they want to do
    print("\nSetup options:")
    print("1. Full setup (virtual env + install dependencies)")
    print("2. Install dependencies only (in current environment)")
    print("3. Download sample images only")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        # Create virtual environment
        if not run_command(
            "python3 -m venv venv",
            "Creating virtual environment"
        ):
            return 1

        # Determine activation script
        if sys.platform == "win32":
            activate = "venv\\Scripts\\activate"
            pip = "venv\\Scripts\\pip"
        else:
            activate = "source venv/bin/activate"
            pip = "venv/bin/pip"

        print(f"\nTo activate the virtual environment, run:")
        print(f"  {activate}")

        # Install dependencies
        if not run_command(
            f"{pip} install -r requirements.txt",
            "Installing dependencies"
        ):
            return 1

    elif choice == "2":
        # Install dependencies in current environment
        if not run_command(
            "pip install -r requirements.txt",
            "Installing dependencies"
        ):
            return 1

    elif choice == "3":
        print("\nSkipping dependency installation...")

    else:
        print("Invalid choice")
        return 1

    # Create sample_images directory
    sample_dir = project_root / "sample_images"
    sample_dir.mkdir(exist_ok=True)
    print(f"\n✓ Created sample_images directory: {sample_dir}")

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)

    print("\nNext steps:")
    print("1. Add your wine bottle images to the sample_images/ folder")
    print("2. Run a test detection:")
    print("   python examples/simple_detection.py")
    print("\nOther examples:")
    print("   python examples/batch_processing.py    # Process multiple images")
    print("   python examples/comparison.py          # Compare YOLO models")

    return 0


if __name__ == "__main__":
    exit(main())
