# Documentation Standards

## Table of Contents
1. [Overview](#overview)
2. [Global Documentation Structure](#global-documentation-structure)
3. [Module-Specific Documentation](#module-specific-documentation)
4. [Documentation Format and Naming Conventions](#documentation-format-and-naming-conventions)
5. [File Requirements for Python Modules](#file-requirements-for-python-modules)
6. [Code Documentation Practices](#code-documentation-practices)

---

## Overview

This document defines the standards for documenting business, technical, and code-related affairs within the repository. It ensures that all contributors understand how and where to document essential aspects of the project, including code structure, usage, and business elements.

**Footnote:** While the overarching program is under active development, documentation must be updated with each pull request or addition to maintain consistency and clarity as the project grows.

---

## Global Documentation Structure

All global documentation is stored under the `/docs` directory and is organized into the following key areas:

1. **Business Documentation**:
   - Contains non-technical documents that relate to company operations, legal compliance, project plans, business strategies, and any necessary licenses or contracts.
   - **Directory**: `/docs/business`
   - **Examples**: Business plans, contracts, project outlines, licenses.

2. **Policies and Standards**:
   - Contains files outlining the coding, review, and workflow standards.
   - **Directory**: `/docs/policies-and-standards`
   - **Examples**: `GitAndGitHubStandards.md`, `BranchNamingConventions.md`, `CodingStandards.md`.

3. **Man Pages**:
   - Stores all program-related documentation for the entire codebase.
   - **Directory**: `/docs/manpages`
   - **Examples**: README files detailing how to use, deploy, and modify the overarching program.

Each sub-directory should contain a README.md that outlines its purpose and lists the contents within.

---

## Module-Specific Documentation

All **root modules** within the overarching program should contain their own `README.md` files. These files should act as module-specific documentation, providing:

1. **Overview**:
   - A brief explanation of the module’s functionality and its purpose in the larger program.

2. **Installation and Setup**:
   - Instructions for setting up the module, including any dependencies not covered in the global requirements file.

3. **Usage**:
   - How to run and utilize the module, including examples of inputs, expected outputs, and special configurations.

4. **Development Guidelines**:
   - If applicable, detail any specific guidelines for contributing to the module, such as coding standards, testing guidelines, or branch usage specific to the module.

5. **Version and Dependencies**:
   - Each module should have its own `requirements.txt` file that lists the dependencies required for it to function. This file will be merged into the main program’s `requirements.txt` by the admins.

---

## Documentation Format and Naming Conventions

### Format

- **Markdown** (`.md`) or **Plain Text** (`.txt`) formats are acceptable for all documentation.
- The language should be formal yet accessible, avoiding technical jargon where possible to ensure clarity for all team members and future collaborators.

### Naming Conventions

- **Documentation files** must be in **Capitalized Case** with no spaces, using only letters and alphanumeric characters.
   - **Examples**: `DocumentationStandards.md`, `GitAndGitHubStandards.md`
- **Code files** should follow **lowercase with underscores** for Python, with no spaces or special characters.
   - **Examples**: `data_processing.py`, `config_handler.py`

---

## File Requirements for Python Modules

### Main `requirements.txt`

A main `requirements.txt` file will exist in the root directory of the program. This file aggregates all module-specific dependencies to provide a comprehensive list of requirements for the entire program.

### Module-Specific `requirements.txt`

- Each root module in Python must include its own `requirements.txt` file, specifying only the dependencies necessary for that module.
- Upon review, the admins will incorporate module-specific requirements into the main `requirements.txt`.

---

## Code Documentation Practices

### General Code Documentation Guidelines

Documentation within code should provide clarity and context without redundancy. Focus on documenting the **why** and **how** rather than the **what**.

- **Functions and Methods**:
  - Include docstrings for all public functions and methods. These should follow the [Google Docstring Style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for consistency and readability.

- **Classes**:
  - Each class should have a docstring explaining its purpose and any unique attributes or methods. This docstring should summarize the role of the class in the module.

### Specific Documentation Guidelines

1. **Function-Level Documentation**:
   - Include a brief docstring at the beginning of each function that explains:
     - **Parameters**: List and explain the function’s input parameters.
     - **Returns**: Describe the data returned by the function.
     - **Raises**: Detail any exceptions that the function might raise.
   - **Example**:
     ```python
     def process_data(data: list) -> dict:
         """
         Processes input data and returns a dictionary of results.

         Args:
             data (list): List of data points to be processed.

         Returns:
             dict: Processed results including mean and standard deviation.

         Raises:
             ValueError: If data is not a list of numbers.
         """
     ```

2. **Class-Level Documentation**:
   - Use class-level docstrings to summarize the purpose and usage of the class.
   - **Example**:
     ```python
     class DataProcessor:
         """
         Class for processing and analyzing data.

         This class provides methods for statistical analysis and
         data transformation. Suitable for numerical data inputs.

         Attributes:
             data (list): List of numerical data.
         """
     ```

3. **In-Line Comments**:
   - Use in-line comments sparingly and only to clarify complex logic or non-standard decisions in the code.
   - Avoid obvious comments; instead, focus on why something is done rather than what is being done.

### API Documentation Standards

For code that interfaces with external services or shared modules, follow these standards for documenting APIs:

- **Endpoints**: Clearly list each endpoint and its purpose in the API documentation.
- **Parameters**: Define all required and optional parameters, including data types and default values.
- **Response Format**: Describe the structure and data types returned by each endpoint.
- **Error Handling**: Document possible error codes or messages and recommended solutions for common issues.
- **Usage Examples**: Provide examples showing both requests and responses to illustrate expected usage.

API documentation should be added to the `docs/ManPages/api` directory. Where possible, maintain consistency with other project documentation.

### Documentation Updates with Pull Requests

Whenever a pull request (PR) introduces new features, refactors existing code, or modifies functionality, relevant documentation files must be updated accordingly. Contributors are responsible for ensuring that:

- All impacted `README.md` files, both module-specific and global, reflect the latest changes.
- Necessary updates to docstrings, function comments, and inline comments are made to maintain clarity and usability.

*Footnote: This process is currently noted for future enforcement once the overarching program is complete.*

