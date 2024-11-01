# File Path Standards

## Table of Contents
1. [Purpose](#purpose)
2. [General Directory Structure](#general-directory-structure)
3. [Naming Conventions](#naming-conventions)
4. [Special Conventions](#special-conventions)
5. [Example Directory Structure](#example-directory-structure)
6. [Guidelines for Adding New Files](#guidelines-for-adding-new-files)

---

## Purpose

This document establishes the standards for organizing, naming, and structuring files and directories across all projects within **MidasTechnologies**. Following these guidelines ensures a clean, predictable file layout, making it easy for team members to navigate, understand, and maintain the repository. Clear structure contributes to project scalability, maintainability, and a smoother development experience.

---

## General Directory Structure

The project root directory, **MidasTechnologies**, includes key folders for code, tests, configuration, documentation, and other resources. The layout below provides an organized structure for optimal project navigation and resource management.

```
MidasTechnologies/
│
├── src/                   # Source code for the entire project
│   ├── neural-network/    # Neural network-related code
│   ├── data-collection/   # Code and data processing tools for data collection
│   │   ├── tests/         # Contains unit tests, integration tests, and test resources
│   │   └── ...            # Additional data collection code modules
│   ├── sentiment-analysis/ # Sentiment analysis-related code
│   ├── frontend/          # Frontend-related code and assets
│   └── ...                # Additional modules or features as needed
│
├── docs/                  # All project documentation
│   ├── BusinessDocumentation/ # Business-related documentation
│   ├── ManPages/          # Global code documentation
│   └── PoliciesAndStandards/ # Documentation standards, policies, and guidelines
│
├── config/                # Configuration files and environment settings
├── data/                  # Static data files for the entire project
├── scripts/               # Utility and setup scripts (e.g., deployment scripts)
├── examples/              # Examples and sample code for demonstrating project usage
└── assets/                # Static assets (images, icons, etc.) for frontend or docs
```

---

## Naming Conventions

1. **File Names**  
   - All file names should be in **lowercase** and use **underscores** to replace spaces (e.g., `data_loader.py`).
   - Avoid any whitespace, special characters, or symbols. Stick to letters, numbers, and underscores.
   - If a file is specific to a particular module or feature, use a descriptive prefix or suffix to clarify its purpose, such as `user_model.py`.

2. **Directory Names**  
   - Directory names should be in **all lowercase** and use hyphens (`-`) to replace spaces (e.g., `user-auth`).
   - Avoid whitespace and special characters to maintain consistency.
   - **src/ Directories**: Organize files by functionality, such as `api`, `models`, `services`, `data-collection`, and `utils`.

3. **Document Files**  
   - **Documentation** should be in **Markdown (.md)** or **Plain Text (.txt)** format.
   - Documentation filenames should follow a capitalized naming convention (e.g., `README.md` or `GitAndGithubStandards.md`).

4. **Environment Variables**  
   - Use **all uppercase** with underscores (e.g., `DATABASE_URL`).

5. **Configuration Files**  
   - Use `.yaml` or `.json` for configuration files to maintain readability and consistency.

6. **Classes**  
   - Class names should follow **PascalCase** (e.g., `UserService`).

### GitHub File Naming and Directory Limitations

When creating or modifying file names and directories, keep in mind GitHub's limitations and best practices:

- **Character Limit**: File and directory names should not exceed 255 characters.
- **Special Characters**: Avoid using special characters (e.g., `@`, `!`, `$`, `%`) as they may cause compatibility issues across different operating systems.
- **Case Sensitivity**: GitHub is case-sensitive, so `FilePathStandards.md` and `filepathstandards.md` are treated as distinct files. Use consistent lowercase naming for all files as per our standards.
- **Path Depth**: Aim to keep directory nesting shallow to improve readability and reduce navigation complexity.

Adhering to these guidelines will ensure compatibility across different development environments and maintain uniformity within the repository.

---

## Special Conventions

1. **Documentation Directories**  
   - Documentation directories in `docs/` are an exception to general naming conventions. Documentation files use **Capitalized Names** (e.g., `ManPages`, `BusinessDocumentation`).

2. **README Files**  
   - Each root module in the project should contain its own `README` file, which provides context and instructions for the specific module.
   - The `README` file must be in Markdown (`README.md`), Plain Text (`README.txt`), or have no extension (simply `README`).
   - **Global documentation** for the entire project is held in the `docs/ManPages` directory.

3. **Test Directories**  
   - All root modules should contain a `tests` directory, located one level deep within the main module folder (e.g., `src/data-collection/tests`). This folder will house unit tests, integration tests, and other test resources relevant to the module.

4. **Data Directories**  
   - A `data/` directory in the project root holds static data files used across the entire project.
   - If a `data/` directory exists within a `src` module, it may have specific functionality. Refer to the module’s `README` file for further clarification.

---

## Example Directory Structure

```
MidasTechnologies/
│
├── src/
│   ├── neural-network/
│   │   ├── models.py          # Neural network models
│   │   ├── training_script.py # Training scripts
│   │   └── ...
│   │
│   ├── data-collection/
│   │   ├── data_sources.py    # Sources for data collection
│   │   ├── process_data.py    # Data processing scripts
│   │   ├── tests/             # Contains unit tests, integration tests, and test resources
│   │   └── ...
│   │
│   ├── sentiment-analysis/
│   │   ├── analyze_sentiment.py # Sentiment analysis script
│   │   └── ...
│   │
│   ├── frontend/
│   │   ├── app.js             # Frontend application code
│   │   └── ...
│   │
│   └── ...
│
├── docs/
│   ├── README.md              # Project overview documentation
│   ├── BusinessDocumentation/ # Legal and business-related documentation
│   ├── ManPages/              # Global code documentation
│   └── PoliciesAndStandards/  # Markdown files on standards, policies, and guidelines
│
├── config/
│   ├── config.yaml            # Main project configuration file
│   └── dev_config.yaml        # Development-specific configuration
│
├── data/
│   └── sample_data.csv        # Static data file example
│
├── scripts/
│   ├── setup.sh               # Script to initialize the project setup
│   └── deploy.sh              # Deployment script
│
├── examples/
│   ├── usage_example.py       # Example usage of main project features
│   └── ...
│
└── assets/
    ├── logo.png               # Project logo
    └── favicon.ico            # Icon for frontend
```

---

## Guidelines for Adding New Files

1. **Check the Existing Structure**  
   - Review the current directory structure before adding a new file to ensure you place it in the most relevant folder.
   - Avoid creating redundant or deeply nested directories without a strong rationale.

2. **Follow Naming Conventions**  
   - Ensure that all files and directories follow the naming conventions specified above, keeping the structure consistent and predictable.

3. **Document New Directories**  
   - When adding a new folder that doesn’t fit existing patterns, include a `README.md` file or entry in the main project documentation within `docs/` to explain its purpose.

4. **Requirements for Documentation**  
   - All documentation files should be placed in the appropriate subdirectory within `docs/`.
   - Markdown or text files should be used for documentation purposes, avoiding other file formats where possible.

