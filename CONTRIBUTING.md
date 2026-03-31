# Contributing to AI Traffic Intelligence System

Thank you for considering contributing! This guide will help you get started.

---

## 🛠️ Development Setup

```bash
# Clone and setup
git clone https://github.com/Karthik2509/AI-Traffic-Intelligence-System.git
cd AI-Traffic-Intelligence-System

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black isort flake8
```

---

## 📐 Code Style

We use **Black** for formatting and **isort** for import sorting:

```bash
black src/ test.py --line-length 100
isort src/ test.py --profile black
```

### Conventions

- **Type hints** on all public function signatures
- **Docstrings** (Google style) on all public classes and functions
- **Dataclasses** for structured data (not plain dicts)
- **Logging** via `src.utils.get_logger(__name__)` — never bare `print()`
- **Constants** in `UPPER_SNAKE_CASE` at module level

---

## 🧪 Testing

Every new module should have corresponding tests in `test.py`:

```bash
# Run tests
pytest test.py -v --tb=short

# With coverage
pytest test.py --cov=src --cov-report=term-missing

# Run specific tests
pytest test.py -k "TestDatabase" -v
```

### Test Requirements

- All new features **must** include tests
- Tests should use `tmp_path` fixture for filesystem operations
- Mock external dependencies (YOLO model, video capture)
- Target: **>80% code coverage**

---

## 🔧 Adding a New Module

### 1. Create the module

Place it in `src/your_module.py` with:
- Module-level docstring explaining the purpose
- Configuration dataclass (e.g., `YourModuleConfig`)
- Main class with clear public API
- Proper logging via `get_logger(__name__)`

### 2. Add configuration

Add a section in `config/settings.yaml`:

```yaml
your_module:
  enabled: true
  key_param: value
```

### 3. Wire into the pipeline

In `src/pipeline.py`, add initialization in `_init_components()` and per-frame calls in `process_frame()`.

### 4. Export from `__init__.py`

```python
from src.your_module import YourMainClass
```

### 5. Add tests

In `test.py`, add a `TestYourModule` class with at least:
- `test_init` — basic construction
- `test_core_functionality` — main feature works
- `test_edge_cases` — empty input, boundary conditions
- `test_reset` — state cleanup

### 6. Update README

Add your module to the feature table and project structure.

---

## 📬 Pull Request Workflow

1. **Fork** the repository
2. Create a **feature branch**: `git checkout -b feature/your-feature`
3. Make changes and **add tests**
4. Run `pytest test.py -v` — all tests must pass
5. Run `black` and `isort` for formatting
6. **Commit** with a descriptive message
7. **Push** and open a Pull Request

### PR Title Format

```
feat: Add vehicle re-identification module
fix: Correct tracker state vector indexing
docs: Update architecture diagram
test: Add edge-case tests for signal optimizer
```

---

## 🐛 Reporting Issues

When reporting bugs, please include:
- Python version (`python --version`)
- OS and architecture
- Steps to reproduce
- Expected vs actual behavior
- Relevant log output from `output/traffic_system.log`

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.
