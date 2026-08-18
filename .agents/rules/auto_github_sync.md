# Global Mandatory Rule: Automated Study & Code Management

## MANDATORY BEHAVIOR FOR AGENT IN ALL CONVERSATIONS:
Even when the user opens a new chat session, ALWAYS automatically perform the following:

1. **Environment & IDE Auto-Config**:
   - Automatically configure virtualenv (`.venv`), `.vscode/settings.json`, and `pyrightconfig.json` so no linter errors/red lines appear.
   - Maintain alias `python=python3` for clean terminal commands.

2. **Code Quality & Organization**:
   - Keep code clean, modular, and PEP8 compliant with clear docstrings.
   - Maintain chapter-by-chapter structure (e.g. `ch01_...`, `ch02_...`).

3. **Auto GitHub Repository Creation & Sync**:
   - Automatically select a clean, standard GitHub repository name (e.g. `deep-learning-from-scratch-study`).
   - Automatically stage (`git add .`), commit (`git commit`), and push (`git push origin main`) all user work without asking for manual git commands.
   - Update remote names immediately if user requests a rename.

4. **Experiment History Preservation**:
   - Never delete past experiment results, loss logs, or plots; keep them recorded for side-by-side comparison.
