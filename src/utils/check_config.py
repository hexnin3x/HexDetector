"""
Configuration Checker for HexDetector

Validates that all necessary paths, dependencies, and settings are properly configured.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from utils.logger import Logger


def check_directories():
    """Check if all required directories exist"""
    logger = Logger()
    logger.log_step("Checking Directory Configuration")
    
    directories = {
        'IoT23 Scenarios': settings.IOT23_SCENARIOS_DIR,
        'IoT23 Attacks': settings.IOT23_ATTACKS_DIR,
        'IoT23 Data': settings.IOT23_DATA_DIR,
        'Experiments': settings.IOT23_EXPERIMENTS_DIR,
        'Logs': settings.LOGS_DIR,
        'Output': settings.OUTPUT_DIR,
        'Models': settings.MODELS_DIR
    }
    
    all_exist = True
    for name, path in directories.items():
        if os.path.exists(path):
            logger.log_info(f"✓ {name}: {path}")
        else:
            logger.log_warning(f"✗ {name} does not exist: {path}")
            all_exist = False
    
    return all_exist


def check_dependencies():
    """Check if all required Python packages are installed"""
    logger = Logger()
    logger.log_step("Checking Dependencies")
    
    required_packages = [
        'numpy',
        'pandas',
        'sklearn',
        'matplotlib',
        'seaborn',
        'xgboost',
        'scapy',
        'psutil'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.log_info(f"✓ {package} is installed")
        except ImportError:
            logger.log_warning(f"✗ {package} is not installed")
            missing_packages.append(package)
    
    if missing_packages:
        logger.log_error(f"Missing packages: {', '.join(missing_packages)}")
        logger.log_info("Install missing packages using: pip install -r requirements.txt")
        return False
    
    return True


def check_configuration():
    """Check overall configuration"""
    logger = Logger()
    logger.log_step("Configuration Validation")
    
    errors = settings.validate_config()
    
    if errors:
        logger.log_error("Configuration validation failed:")
        for error in errors:
            logger.log_error(f"  - {error}")
        return False
    else:
        logger.log_info("✓ Configuration is valid")
        return True


def print_summary(dirs_ok, deps_ok, config_ok):
    """Print summary of configuration check"""
    logger = Logger()
    logger.log_separator('=', 70)
    logger.log_info("CONFIGURATION CHECK SUMMARY")
    logger.log_separator('=', 70)
    
    status = []
    status.append(('Directories', '✓ PASS' if dirs_ok else '✗ FAIL'))
    status.append(('Dependencies', '✓ PASS' if deps_ok else '✗ FAIL'))
    status.append(('Configuration', '✓ PASS' if config_ok else '✗ FAIL'))
    
    for item, result in status:
        logger.log_info(f"{item:<20} {result}")
    
    logger.log_separator('=', 70)
    
    if all([dirs_ok, deps_ok, config_ok]):
        logger.log_info("✓ ALL CHECKS PASSED - You may proceed to the next step")
        return True
    else:
        logger.log_error("✗ SOME CHECKS FAILED - Please fix the issues above")
        logger.log_info("\nNext steps:")
        if not deps_ok:
            logger.log_info("  1. Install dependencies: pip install -r requirements.txt")
        if not dirs_ok or not config_ok:
            logger.log_info("  2. Update paths in src/config/settings.py")
            logger.log_info("  3. Run this script again to verify")
        return False


def main():
    """Main execution function"""
    logger = Logger()
    logger.log_info("HexDetector Configuration Checker")
    logger.log_info("=" * 70)
    
    # Run all checks
    dirs_ok = check_directories()
    print()
    
    deps_ok = check_dependencies()
    print()
    
    config_ok = check_configuration()
    print()
    
    # Print summary
    success = print_summary(dirs_ok, deps_ok, config_ok)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
