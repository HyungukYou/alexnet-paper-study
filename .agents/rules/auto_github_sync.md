# Auto GitHub Repository Management & Sync Rule

## Automatic Instructions for Agent
1. **Automatic Repository Naming & Setup**:
   - Automatically pick a clear, standard GitHub repository name (e.g. `deep-learning-from-scratch-study`).
   - If user asks to rename it later, update remote URLs and names accordingly.
2. **Auto Commit & Push**:
   - Whenever any python script, notebook, or documentation is created or modified, format code cleanly.
   - Stage all changes (`git add .`), commit with descriptive chapter messages (`git commit -m "Feat(Ch01): ..."`).
   - Push to GitHub (`git push origin main`).
3. **Persistence Across Sessions**:
   - Apply this rule automatically across all present and future conversation sessions.
