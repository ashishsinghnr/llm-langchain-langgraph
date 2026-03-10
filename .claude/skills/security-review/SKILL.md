---
name: security-review
description: Use this skill when adding authentication, handling user input, working with secrets, creating API endpoints, or implementing payment/sensitive features. Provides comprehensive security checklist and patterns.
origin: ECC
---

# Security Review Skill

This skill ensures all code follows security best practices and identifies potential vulnerabilities.

## When to Activate

- Implementing authentication or authorization
- Handling user input or file uploads
- Creating new API endpoints
- Working with secrets or credentials
- Storing or transmitting sensitive data
- Integrating third-party APIs

## Security Checklist

### 1. Secrets Management

```python
# ❌ NEVER: Hardcoded secrets
api_key = "sk-proj-xxxxx"

# ✅ ALWAYS: Environment variables
import os
api_key = os.environ.get("API_KEY")
if not api_key:
    raise RuntimeError("API_KEY not configured")
```

#### Verification Steps
- [ ] No hardcoded API keys, tokens, or passwords
- [ ] All secrets in environment variables
- [ ] `.env` in .gitignore
- [ ] No secrets in git history

### 2. Input Validation

```python
# ❌ NEVER: eval() on user input
result = eval(user_expression)

# ✅ ALWAYS: Safe alternatives
import ast
result = ast.literal_eval(user_expression)  # For literals only

# Or use a whitelist approach
allowed = {"sqrt": math.sqrt, "pow": pow}
```

### 3. SQL Injection Prevention

```python
# ❌ NEVER: String concatenation
query = f"SELECT * FROM users WHERE email = '{email}'"

# ✅ ALWAYS: Parameterized queries
cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
```

### 4. Sensitive Data Exposure

```python
# ❌ WRONG: Logging sensitive data
print(f"User login: {email}, {password}")

# ✅ CORRECT: Redact sensitive data
print(f"User login: {email}")
logger.info("Login attempt", extra={"user_id": user_id})
```

### 5. Error Handling

```python
# ❌ WRONG: Exposing internals
except Exception as e:
    return {"error": str(e), "traceback": traceback.format_exc()}

# ✅ CORRECT: Generic error messages
except Exception as e:
    logger.exception("Internal error")
    return {"error": "An error occurred. Please try again."}
```

### 6. Dependency Security

```bash
# Check for vulnerabilities
pip-audit
safety check

# Keep dependencies updated
pip list --outdated
```

## Pre-Deployment Checklist

- [ ] No hardcoded secrets
- [ ] All user inputs validated
- [ ] No eval() on untrusted input
- [ ] Error messages don't leak internals
- [ ] Dependencies up to date
- [ ] .env files in .gitignore
- [ ] No secrets in git history

**Remember**: Security is not optional. When in doubt, err on the side of caution.
