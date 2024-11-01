# MiadTechnologies Coding Standards

This document details the coding standards and practices for maintaining consistency, readability, and quality in the codebase for all contributors. These guidelines apply across programming languages and environments used within the project, focusing primarily on Python and any related languages used for interoperability.

---

## Table of Contents
1. [Python Coding Standards](#python-coding-standards)
2. [Virtual Environments and Dependency Management](#virtual-environments-and-dependency-management)
3. [Interfacing with Other Languages](#interfacing-with-other-languages)
4. [Documentation Standards for Code](#documentation-standards-for-code)

---

## Python Coding Standards

Our primary codebase is in Python, and we adhere strictly to **PEP8** guidelines with additional standards to ensure clarity and consistency. 

### PEP8 Guidelines and Best Practices
- **Naming Conventions**:
  - **Variables and functions**: Use `snake_case` for readability.
  - **Classes**: Use `PascalCase`.
  - **Constants**: Use `ALL_CAPS`.
- **Line Length**: Limit lines to **79 characters** for readability.
- **Indentation**: Use **4 spaces per indentation level**.
- **Docstrings**: Follow [PEP 257](https://www.python.org/dev/peps/pep-0257/) conventions.
  - Use docstrings for all public modules, classes, and functions.
  - Structure them using the [Google docstring style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
- **Comments**: Add comments as needed to clarify code functionality, especially around complex logic. Avoid redundant comments.

### Jupyter Notebook Usage
While Jupyter notebooks are valuable for experimentation, they should not be submitted directly to the repository unless absolutely necessary. Instead:
1. **Convert the notebook to a `.py` file** before adding it to the repo. 
   - In Jupyter, navigate to `File > Download as > Python (.py)` to obtain a `.py` version of the notebook.
2. **Document the Python file** properly and integrate it with the existing codebase standards.

### Python Best Practices Quick Reference

Follow these essential Python best practices to ensure code consistency across the project:

- **Use List Comprehensions**: Prefer list comprehensions over traditional loops for concise code.
  ```python
  # Recommended
  squares = [x**2 for x in range(10)]
```
* Avoid Global Variables: Limit the use of global variables, especially in modules intended for import.

* String Formatting: Use f-strings (formatted string literals) for improved readability in Python 3.6 and above.

```python
    name = "Alice"
    print(f"Hello, {name}!")
```

   * Error Handling: Use specific exceptions where possible. Avoid catching all exceptions with except Exception.

   * PEP8 Line Length: Adhere to a line length of 79 characters, and use 4 spaces per indentation level.

   * Docstrings: Use Google-style docstrings to document functions, modules, and classes.

>Refer to PEP8 and Google Python Style Guide for detailed information on Python coding standards.

### Recommended Tools
- **Flake8**: Enforces PEP8 standards.
- **Black**: Automatic formatter for consistent styling.
- **isort**: Automatically organizes imports.

These tools can be added to your development environment for smoother compliance with coding standards.

---

## Virtual Environments and Dependency Management

Using a virtual environment isolates dependencies and keeps the project environment consistent.

### Setting Up a Virtual Environment
1. **Create the virtual environment** in the project root:
   ```bash
   python -m venv venv
   ```
2. **Activate the environment**:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
3. **Install dependencies** from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

### Managing Dependencies with `requirements.txt`
All project dependencies should be added to `requirements.txt`. When new packages are added:
1. Install the package using `pip`.
2. Update `requirements.txt` by freezing the environment:
   ```bash
   pip freeze > requirements.txt
   ```

### Important Note
- **Do not push the `venv` directory** to the repository. The virtual environment is a local development tool and should remain excluded in `.gitignore`.

---

Here's a more detailed and enhanced section for the **Interfacing with Other Languages** area, incorporating standards and industry best practices across each language, file format, and tool.

---

## Interfacing with Other Languages

Python’s flexibility allows it to integrate with other languages, databases, and formats, which is particularly beneficial for high-performance tasks, front-end functionality, data exchange, and efficient file handling. Below are the industry standards, best practices, and interfacing guidelines for each language or format commonly used with Python.

### 1. C (CPython and C Extensions)
C is often used in Python projects for performance-critical modules and low-level operations.
- **Coding Standards**: Follow the [GNU Coding Standards](https://www.gnu.org/prep/standards/).
  - Use `snake_case` for variable and function names.
  - Avoid global variables, and encapsulate functions within modules.
- **Documentation**: Use comments within `.c` and `.h` files, explaining complex logic and including a description of each function.
  - Follow structured comment headers for each function, detailing parameters, expected return values, and purpose.
- **Interfacing with Python**:
  - Use `ctypes` or `cffi` for interfacing.
  - **Example**: Use `ctypes` to call a C function from Python:
    ```c
    // function.c
    int add(int a, int b) {
        return a + b;
    }
    ```
  - Use `setup.py` with `distutils` to compile C extensions for easy distribution.

### 2. TypeScript and JavaScript
TypeScript, a typed superset of JavaScript, should be used for type safety and maintainability on the front-end.
- **Coding Standards**: Follow the [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript).
  - Use `camelCase` for variables and function names.
  - Prefer `const` and `let` over `var`.
- **Documentation**: Comment functions and classes with JSDoc.
  - Use TypeScript's type annotations to document data types explicitly.
- **Interfacing**:
  - Use `pyodide` or REST APIs to communicate between Python and JavaScript.
  - When possible, separate concerns by keeping Python back-end tasks independent of TypeScript front-end logic, communicating only through defined APIs.

### 3. Go
Go is efficient for concurrent tasks, and is often used alongside Python for backend services or tasks needing efficient multithreading.
- **Coding Standards**: Follow [Effective Go](https://golang.org/doc/effective_go.html).
  - Use `PascalCase` for exported (public) identifiers and `camelCase` for private ones.
  - Format code with `go fmt` to maintain consistency.
- **Documentation**:
  - Add a comment above each function explaining its functionality, parameters, and expected return values.
- **Interfacing**:
  - Use `cgo` to call C functions from Go if needed, or implement a REST API to interact with Python.

### 4. Rust
Rust is a high-performance language with strong memory safety guarantees, making it suitable for secure, efficient modules in Python.
- **Standards**: Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).
  - Use `snake_case` for function and variable names, and `CamelCase` for structs and enums.
  - Use `clippy` for linting and `rustfmt` for consistent formatting.
- **Documentation**:
  - Use triple slashes (`///`) for documentation comments on functions and modules.
  - Each public function should explain parameters, return values, and potential errors.
- **Interfacing**:
  - Use `pyo3` for embedding Rust code in Python or `maturin` for building and publishing Python packages containing Rust code.

### 5. JSON
JSON is a lightweight data-interchange format and is frequently used in Python projects for configuration and data exchange.
- **Format**: Follow strict schema validation to ensure data consistency.
  - Use lowercase keys in `snake_case`.
  - Avoid deeply nested structures; keep depth manageable for readability.
- **Best Practices**:
  - Validate JSON against a schema using tools like `jsonschema`.
  - Use consistent encoding (UTF-8) and 4-space indentation.

### 6. CSV
CSV files are widely used for tabular data storage and exchange.
- **Format**: Always include headers as the first row, and use commas (`,`) as delimiters.
- **Best Practices**:
  - Handle missing data by filling with placeholders or removing empty rows if appropriate.
  - Use Python’s `csv` library, and include headers in the file for clarity.
  - Ensure data types are consistent across columns (e.g., dates, numbers).

### 7. SQL (Database Standards)
SQL databases are essential for structured data storage and querying.
- **Naming Conventions**:
  - Use `snake_case` for table and column names.
  - Avoid special characters and reserved keywords in table or column names.
- **Standards**: Follow [SQL Style Guide](https://www.sqlstyle.guide/).
  - Use constraints and foreign keys for data integrity.
  - Normalize data as needed, but consider denormalization for performance in specific cases.
- **Indexing**: Index frequently queried columns for faster retrieval.
- **Transactions**: Wrap changes in transactions to ensure data consistency, and roll back on errors.

### 8. Docker
Docker standardizes development environments and deployment by containerizing applications.
- **Dockerfile Standards**:
  - Use multi-stage builds to optimize image size.
  - Only include production dependencies in the final image.
  - Use environment variables for configuration rather than hardcoding values.
- **Directory Structure**:
  - Place Docker-related files in a `docker/` directory.
  - Use `docker-compose.yml` for multi-container setups.
- **Best Practices**:
  - Use `.dockerignore` to exclude unnecessary files, like logs and local configuration.
  - Always use stable tags for dependencies instead of `latest`.

### 9. Binary Building for Python
For performance-critical Python code, consider building binaries.
- **Tools**:
  - Use `Cython` to compile Python code into C, then build as a binary.
  - Alternatively, use `PyInstaller` to package Python scripts as standalone executables.
- **Best Practices**:
  - Ensure binaries are compatible with target deployment environments.
  - Document the binary build process for reproducibility.

### 10. Makefiles
**Makefiles** provide a standardized way to compile and manage dependencies, particularly for non-Python code.
- **Structure**:
  - Place the `Makefile` in the root of the project.
  - Organize commands for easy readability, grouping related commands together.
- **Common Targets**:
  - `build`: Compile code, if needed.
  - `clean`: Remove compiled files and dependencies.
  - `test`: Run test suites.
  - `install`: Set up dependencies or environment configurations.

---

## Documentation Standards for Code

Clear and concise documentation is essential for long-term project maintainability. 

### Docstring and In-Code Documentation
- **Functions and Classes**: Use [Google Style Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
  - Each docstring should describe what the function does, input parameters, return values, and any exceptions raised.
- **In-Line Comments**: Only add comments where needed to explain non-obvious parts of the code.
- **Module-Level Documentation**: Provide a brief overview at the top of each file explaining its purpose and any key dependencies.

### README.md for Each Module
Each root module should contain a `README.md` file covering:
- **Overview**: High-level description of the module’s functionality.
- **Setup**: Dependencies, pip installs, libraries, and setup instructions.
- **Usage**: Sample commands or code for running the module.
- **API Documentation**: If applicable, list available functions, classes, or endpoints.

