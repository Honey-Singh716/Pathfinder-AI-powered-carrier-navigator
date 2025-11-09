# Career Path Finder (Prototype)

A simple CLI and Web AI agent that uses a Decision Tree to suggest a career based on your dataset features. It prints a short roadmap for the predicted career and stores user history.

## Setup

- Recommended: Create a virtual environment
  - Windows (PowerShell):
    - `python -m venv .venv`
    - `.venv\\Scripts\\Activate.ps1`
- Install dependencies:
  - `pip install -r requirements.txt`

## Run (CLI)

- `python career_agent.py`
- Answer the prompts; the model will output a recommended career and a roadmap.

## Data

- The app uses your dataset by default: `career_prediction_dataset_10000.csv` at the project root.
- Column names are normalized (case/spacing-insensitive). If your target is `Career_Path`, it is automatically mapped to `career`.
- The web UI is dataset-driven: categorical fields become dropdowns with unique values from your CSV; numeric fields become number inputs with min/max hints.

## Notes

- This is a minimal prototype. For better results, expand the dataset, tune the model, and consider cross-validation.

---

# Web App (Flask UI)

A simple Flask app with signup/login/logout, dashboard to input details, prediction results with roadmap, and history per user. User accounts and history are stored in `data/users.json` with hashed passwords.

## Run the web app

1. Activate your venv (see above) and install requirements:
   - `pip install -r requirements.txt`
2. (Optional) Set a secret for sessions (PowerShell):
   - `$env:APP_SECRET = "your-long-random-secret"`
3. Start the server:
   - `python app.py`
4. Open: `http://127.0.0.1:5000`

## Auth and Persistence

- On first run, `data/users.json` is created automatically.
- Signup stores: `password_hash`, `profile` (reserved), and `history` (array of predictions).
- Each prediction appends an entry with timestamp, inputs, top matches, and roadmap.

## Shared Model Logic

- Core ML pipeline and roadmaps live in `career_model.py` and are reused by both the CLI and the Flask app.
- The dataset path is set to `career_prediction_dataset_10000.csv`. If it is missing, the app will error and ask you to place your CSV there.
