from typing import Dict, List, Tuple
import os
import json
import time
import hashlib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib

# Shared career roadmaps
ROADMAPS: Dict[str, List[str]] = {
    "Data Scientist": [
        "Master Python & SQL",
        "Statistics & Probability",
        "Data wrangling with pandas",
        "ML algorithms & model evaluation",
        "Projects: EDA, classification, regression",
        "Deploy a model (Flask/FastAPI)"
    ],
    "Software Engineer": [
        "Data structures & algorithms",
        "Language mastery (Python/Java/JS)",
        "Version control & testing",
        "System design basics",
        "Build full-stack or backend projects"
    ],
    "Web Developer": [
        "HTML/CSS/JS fundamentals",
        "Frontend framework (React/Vue)",
        "Backend basics (Node/Express or Django)",
        "Build and deploy 2-3 web apps"
    ],
    "ML Engineer": [
        "Strong Python & NumPy",
        "ML pipelines & MLOps basics",
        "Model optimization & deployment",
        "Cloud (AWS/GCP/Azure)"
    ],
    "UI/UX Designer": [
        "Design fundamentals & typography",
        "Figma & prototyping",
        "User research & usability testing",
        "Design a portfolio"
    ],
    "Embedded Engineer": [
        "C/C++ & microcontrollers",
        "Electronics fundamentals",
        "RTOS & interfacing sensors",
        "Build embedded projects"
    ],
    "Business Analyst": [
        "Excel/SQL & dashboards",
        "Requirements gathering",
        "Visualization (Power BI/Tableau)",
        "Case studies & domain knowledge"
    ],
    "Cybersecurity Analyst": [
        "Networking & OS basics",
        "Security fundamentals",
        "Vulnerability assessment & SIEM",
        "CTFs and lab practice"
    ],
    "Cloud Engineer": [
        "Linux & networking",
        "Cloud provider (AWS/GCP/Azure)",
        "IaC (Terraform) & CI/CD",
        "Deploy scalable apps"
    ],
}

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "Extended_Career_Prediction_Dataset__24_Careers__10_000_rows_.csv")
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "data", ".model_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


def _cache_key() -> str:
    path = os.path.abspath(DATA_PATH)
    try:
        st = os.stat(path)
        meta = f"{path}|{int(st.st_mtime)}|{int(st.st_size)}|v2"  # bump version to invalidate old cache
    except FileNotFoundError:
        meta = f"{path}|missing|v2"
    return hashlib.sha1(meta.encode("utf-8")).hexdigest()


def _cache_path() -> str:
    return os.path.join(MODEL_CACHE_DIR, f"model_{_cache_key()}.joblib")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    # unify target name to 'career'
    if 'career_path' in df.columns and 'career' not in df.columns:
        df = df.rename(columns={'career_path': 'career'})
    return df


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please place your CSV at this path.")
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    # drop rows without target
    if 'career' in df.columns:
        df = df.dropna(subset=['career'])
    return df


def build_pipeline(df: pd.DataFrame) -> Tuple[Pipeline, pd.DataFrame, pd.Series]:
    if 'career' not in df.columns:
        raise ValueError("Dataset must contain a target column 'career' or 'career_path'.")
    X = df.drop(columns=["career"])  # features
    y = df["career"]

    # Dynamically select features
    # Categorical: dtype object or low-cardinality (<30 uniques)
    # Numeric: anything that is number-like
    cat_cols = []
    num_cols = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            num_cols.append(col)
        else:
            # treat as categorical if object or low unique
            if X[col].nunique(dropna=True) <= 50:
                cat_cols.append(col)
            else:
                # high-cardinality text; skip by default
                pass

    categorical = cat_cols
    numeric = num_cols

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", MinMaxScaler(), numeric),
        ]
    )

    clf = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    return pipe, X, y


def train_model() -> Pipeline:
    df = load_data()
    pipe, X, y = build_pipeline(df)
    # FAST_TRAIN mode: skip grid search for speed
    fast = os.environ.get("FAST_TRAIN", "0") == "1"
    if fast:
        pipe.set_params(
            clf__max_depth=8,
            clf__min_samples_split=10,
            clf__min_samples_leaf=2,
            clf__criterion="gini",
        )
        pipe.fit(X, y)
        return pipe

    # Hyperparameter tuning with stratified CV (smaller grid for speed)
    param_grid = {
        "clf__max_depth": [6, 10, None],
        "clf__min_samples_split": [2, 10],
        "clf__min_samples_leaf": [1, 3],
        "clf__criterion": ["gini", "entropy"],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv, n_jobs=-1, scoring="accuracy")
    gs.fit(X, y)
    return gs.best_estimator_


_MODEL: Pipeline | None = None


def get_model() -> Pipeline:
    global _MODEL
    if _MODEL is None:
        # try loading from cache
        cache_fp = _cache_path()
        try:
            if os.path.exists(cache_fp):
                _MODEL = joblib.load(cache_fp)
            else:
                _MODEL = train_model()
                joblib.dump(_MODEL, cache_fp)
        except Exception:
            # fall back to training if cache fails
            _MODEL = train_model()
    return _MODEL


def predict(user_row: Dict) -> Dict:
    import pandas as pd  # local import to avoid polluting namespace

    model = get_model()
    df_row = pd.DataFrame([user_row])
    career = model.predict(df_row)[0]

    top = []
    if hasattr(model["clf"], "predict_proba"):
        try:
            classes = model["clf"].classes_
            probas = model.predict_proba(df_row)[0]
            pairs = sorted(zip(classes, probas), key=lambda x: x[1], reverse=True)[:3]
            top = [(c, float(p)) for c, p in pairs]
        except Exception:
            top = []

    roadmap = ROADMAPS.get(career, ["Explore foundational skills and build 2-3 projects."])
    return {"career": career, "top": top, "roadmap": roadmap}


def dataset_metadata() -> Dict:
    """Return options for categorical fields and ranges for numeric fields from the dataset."""
    df = load_data()
    X = df.drop(columns=[c for c in df.columns if c in ("career",)])

    cat_options: Dict[str, List[str]] = {}
    num_ranges: Dict[str, Dict[str, float]] = {}

    for col in X.columns:
        series = X[col]
        if pd.api.types.is_numeric_dtype(series):
            s = series.dropna()
            if len(s):
                num_ranges[col] = {
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "mean": float(s.mean()),
                }
        else:
            uniques = sorted([str(v) for v in series.dropna().unique().tolist()])
            # limit to a reasonable number to avoid huge selects
            if 1 <= len(uniques) <= 200:
                cat_options[col] = uniques

    # Identify a primary interest column
    interest_col = None
    # Prefer exact 'interest' if present (dataset header 'Interest' normalizes to 'interest')
    if 'interest' in cat_options:
        interest_col = 'interest'
    else:
        for c in cat_options.keys():
            if "interest" in c.lower():
                interest_col = c
                break
    if interest_col is None and cat_options:
        # default to the first categorical column as interest
        interest_col = list(cat_options.keys())[0]

    # Build per-interest filters for other categorical columns
    filters_by_interest: Dict[str, Dict[str, List[str]]] = {}
    if interest_col is not None and interest_col in X.columns:
        for interest_val in sorted([str(v) for v in X[interest_col].dropna().unique().tolist()]):
            mask = X[interest_col].astype(str) == interest_val
            sub = X.loc[mask]
            per_col: Dict[str, List[str]] = {}
            for c in cat_options.keys():
                if c == interest_col:
                    continue
                vals = sorted([str(v) for v in sub[c].dropna().unique().tolist()])
                per_col[c] = vals
            filters_by_interest[interest_val] = per_col

    return {
        "categorical": cat_options,
        "numeric": num_ranges,
        "filters": {
            "interest_col": interest_col,
            "by_interest": filters_by_interest,
        },
    }
