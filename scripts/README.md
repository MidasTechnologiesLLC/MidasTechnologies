# Scripts Directory

## Overview

This directory contains bash scripts and related tools designed to facilitate the setup and deployment of the **Midas Technologies LLC** program. The scripts automate the process of downloading, installing, and configuring the software, ensuring all dependencies are installed and the latest version of the program is fetched from the GitHub repository.

## Purpose

The primary goals of the scripts in this directory are:
1. **Dependency Installation**: Ensure that all necessary dependencies for the program are installed, tailored to the user's operating system (e.g., Linux, macOS, or Windows Subsystem for Linux).
2. **Program Setup**: Download and configure the MidasV1 trading bot and its modules.
3. **Package Management**: Fetch the latest version of the program from the GitHub repository and handle version upgrades if needed.
4. **Cross-Platform Compatibility**: Provide robust support for multiple operating systems.

## Features

- **Automated Installation**: A single command to set up the entire program.
- **Environment Configuration**: Automatically creates and activates virtual environments (for Python components).
- **Version Detection**: Checks the installed version and updates to the latest release from GitHub if necessary.
- **System Checks**: Verifies compatibility with the host operating system and handles OS-specific setup processes.
- **Clean Installation**: Removes outdated versions and installs the latest package.

## Usage

To use the scripts in this directory, follow these steps:

1. **Navigate to the scripts directory**:
   ```bash
   cd scripts
   ```

2. **Run the main setup script**:
   ```bash
   bash setup.sh
   ```

   > The setup script will:
   > - Detect your operating system.
   > - Install required system dependencies (e.g., `python3`, `pip`, `gcc`).
   > - Clone or update the latest version of the program from GitHub.
   > - Install Python dependencies using `pip`.

3. **Advanced Usage**:
   - To install specific versions of the program:
     ```bash
     bash setup.sh --version <version_number>
     ```
   - For verbose output (debugging mode):
     ```bash
     bash setup.sh --verbose
     ```

4. **Check for Updates**:
   ```bash
   bash update.sh
   ```

   > This script checks for new releases on GitHub and updates the program while preserving user data.

## Structure

```
scripts/
├── setup.sh         # Main script to install and configure the program
├── update.sh        # Script to check for and install updates
├── dependencies.sh  # Script for installing OS-specific dependencies
└── README.md        # Documentation for this directory
```

## Supported Operating Systems

The setup scripts are designed to work on the following platforms:
- **Linux**: Ubuntu, Debian, Arch Linux, and more.
- **macOS**: Requires `brew` for dependency management.
- **Windows (via WSL)**: Windows Subsystem for Linux (Ubuntu preferred).

## Contribution

If you encounter any issues or would like to contribute to improving the scripts, feel free to submit a pull request or create an issue on the main GitHub repository.

## Disclaimer

These scripts are optimized for use with **Midas Technologies LLC** programs and are not intended for general-purpose use. Always review scripts before running them on your system.
```

