---
name: python-patterns
description: Pythonic idioms, PEP 8 standards, type hints, and best practices for building robust, efficient, and maintainable Python applications.
origin: ECC
---

# Python Development Patterns

Idiomatic Python patterns and best practices for building robust, efficient, and maintainable applications.

## When to Activate

- Writing new Python code
- Reviewing Python code
- Refactoring existing Python code
- Designing Python packages/modules

## Core Principles

### 1. Readability Counts

```python
# Good: Clear and readable
def get_active_users(users: list[User]) -> list[User]:
    return [user for user in users if user.is_active]

# Bad: Clever but confusing
def get_active_users(u):
    return [x for x in u if x.a]
```

### 2. Explicit is Better Than Implicit

```python
# Good: Explicit configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Bad: Hidden side effects
some_module.setup()
```

### 3. EAFP - Easier to Ask Forgiveness Than Permission

```python
# Good: EAFP style
try:
    return dictionary[key]
except KeyError:
    return default_value
```

## Type Hints

```python
# Modern Type Hints (Python 3.9+)
def process_items(items: list[str]) -> dict[str, int]:
    return {item: len(item) for item in items}

# Protocol-Based Duck Typing
from typing import Protocol

class Renderable(Protocol):
    def render(self) -> str: ...
```

## Error Handling

```python
# Specific exceptions with chaining
def load_config(path: str) -> Config:
    try:
        with open(path) as f:
            return Config.from_json(f.read())
    except FileNotFoundError as e:
        raise ConfigError(f"Config not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON: {path}") from e
```

## Context Managers

```python
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    start = time.perf_counter()
    yield
    print(f"{name} took {time.perf_counter() - start:.4f}s")
```

## Data Classes

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class User:
    id: str
    name: str
    email: str
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
```

## Anti-Patterns to Avoid

```python
# Bad: Mutable default arguments
def append_to(item, items=[]):  # DON'T

# Good: Use None
def append_to(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

# Bad: Bare except
try:
    risky()
except:  # DON'T
    pass

# Good: Specific exception
try:
    risky()
except SpecificError as e:
    logger.error(f"Failed: {e}")
```

## Tooling

```bash
black .          # Code formatting
isort .          # Import sorting
ruff check .     # Linting
mypy .           # Type checking
pytest --cov     # Testing with coverage
bandit -r .      # Security scanning
```

**Remember**: Prioritize clarity over cleverness. When in doubt, be explicit.
