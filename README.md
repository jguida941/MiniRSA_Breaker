# 🔐 MiniRSA Breaker

**Category:** Cryptography / Educational Visualization  
**Tech Stack:** Python · PyQt6  
**Author:** [Justin Guida](https://github.com/jguida941)  
**White Hat Verified 🧢**

## Overview

**MiniRSA Breaker** is an interactive desktop application that demonstrates how **RSA encryption and decryption** work through clear, visual, and mathematical explanations.  
It provides **step-by-step feedback** for every stage of the process, prime generation, key computation, and ciphertext decoding, allowing learners to see how real cryptographic logic operates. 

Build understanding through math, logic, and real cryptographic theory. **All students and educators welcome.**


## Use it to:

-  **Teach** RSA key concepts  
-  **Break down** modular exponentiation  
-  **Compare** weak vs. strong keys  

Whether you're a **teacher**, a **CS student**,or just **someone fascinated by how passwords work**, this tool was built for you.


## Features

**RSA Key Generation**  
Generate custom or random small prime keys instantly.

**Modular Exponentiation Engine**  
Performs `c = m^e mod n` and `m = c^d mod n` using Python’s `pow()` function, real, fast RSA behavior.

**Character Mapping System**  
Maps characters **A–Z** to integers **01–26**, with optional symbol support for extended messages.

**Encryption & Decryption Panels**  
Visualize the full message transformation — from plaintext to ciphertext and back,  in real time.

**Step-by-Step Math Breakdown**  
Each stage of RSA math is clearly explained and animated for learning purposes.

**Error Handling + Input Validation**  
Smart checks catch invalid primes, characters, and unsupported input types, with helpful messages.

**100% Offline Application**  
Runs entirely on your device , no web server, no uploads, just fast, secure local computation.

**Debug Mode & Educational Hints**  
Toggle debug mode to show raw values, intermediate results, and contextual explanations at every step.

## Screenshots

### Main Interface RSA Setup: 
Define primes, generate public/private keys, and view entropy ratings in real time.

<img width="1195" alt="Screenshot 2025-06-27 at 10 18 09 PM" src="https://github.com/user-attachments/assets/b5174655-6163-4e40-856f-27b8a8a2759c" />

<br>



### Encryption Panel: 
Watch your message get encrypted character by character using modular exponentiation.

<img width="1197" alt="Screenshot 2025-06-27 at 10 18 49 PM" src="https://github.com/user-attachments/assets/5da211a7-97e6-44e0-899d-0e331e3bacbf" />
<br>



### Decryption Panel: 
Reverse the cipher text and view RSA logic in reverse using your private key

<img width="1198" height="908" alt="Screenshot 2025-10-08 at 8 39 51 AM" src="https://github.com/user-attachments/assets/4038e8ae-68a6-4a9d-8fc9-a6ef96e456c6" />


<br>

### How to Run:.

#### Follow these steps to launch **MiniRSA_Breaker** locally:

 **1. Clone the Repository**
```bash
git clone https://github.com/jguida941/MiniRSA_Breaker.git
cd MiniRSA_Breaker
```

**2. Create a virtual environment (recommended)**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Launch the app**
```bash
python rsa.py
```

## Quality & Testing Pipeline

Run the full local quality stack (matches CI gates):

```bash
pip install -e .[dev,gui]
pip install pre-commit && pre-commit install
make quality
```

Targeted checks (see `docs/ai-assisted-quality-playbook.md` for the full workflow):

- For Codex or fully automated runs without Docker, follow `docs/codex-integration.md` or simply execute `make codex-pipeline` (JSON summary is written to `codex-report.json`).
- `ruff check .` – static analysis that catches unused imports, undefined names, and formatting drift before runtime.
- `mypy src/mini_rsa` – strict typing on the math engine to surface AI-generated signature mismatches or accidental `Any`.
- `pytest -m "gui"` – headless Qt smoke tests (use `xvfb-run` on Linux) to verify signal wiring and background threads.
- `pytest tests/unit/test_imports.py` – import smoke to expose circular imports or optional dependency slips.
- `mutmut run --paths-to-mutate src/mini_rsa` – mutation testing to prove the unit/property tests kill realistic math regressions.
- `bandit -q -r src/mini_rsa rsa.py` – security linting for unsafe randomness, subprocess usage, or crypto misconfigurations.
- `pip-audit -r requirements.txt` – dependency vulnerability scan so nightly AI updates do not introduce known CVEs.
- `make mutation MUTATE_PATHS=src/mini_rsa/core.py` – quick scoped mutation sweep before merging AI-generated core edits.
- `make mutation-clean` – clear cached Hypothesis/Mutmut artefacts if diffs get noisy.
- `pre-commit run --all-files` – mirror the lint/type/security checks that run automatically on every commit.

## License

**Evaluation only - all rights reserved.**

You may **clone and run locally** for personal or hiring evaluation.  
You may **not** redistribute, sublicense, or use this work commercially without my written permission.

See the [LICENSE](LICENSE) file for the exact terms.

**Qt note:** This app uses **PyQt6 (GPLv3)**. Do **not** redistribute the app unless you comply with GPLv3 or have a Qt commercial license.

 <h3> Educational Use Terms</h3>
<ul>
  <li>Use this tool in lessons, demos, or assignments</li>
  <li>Modify and explore it freely for academic learning</li>
  <li>Ensure visible credit appears in repurposed or adapted materials</li>
</ul>

<h3> Learn from the Best:</h3>
<ul>
  <li><a href="https://www.freecodecamp.org/" target="_blank">FreeCodeCamp: Scientific Python</a></li>
  <li><a href="https://cs50.harvard.edu/x/" target="_blank">Harvard CS50x 2024</a></li>
  <li><a href="https://en.wikipedia.org/wiki/Cryptography" target="_blank">Cryptographic Methods</a></li>
</ul>
