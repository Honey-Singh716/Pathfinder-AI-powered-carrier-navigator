from __future__ import annotations
import os
import json
import datetime as dt
from typing import Dict, Any

from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.security import generate_password_hash, check_password_hash

from career_model import predict, ROADMAPS, dataset_metadata

APP_SECRET = os.environ.get("APP_SECRET", "dev-secret-change-me")
USERS_PATH = os.path.join("data", "users.json")

app = Flask(__name__)
app.secret_key = APP_SECRET

os.makedirs("data", exist_ok=True)
if not os.path.exists(USERS_PATH):
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump({"users": {}}, f, indent=2)


def load_users() -> Dict[str, Any]:
    with open(USERS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(data: Dict[str, Any]):
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def current_user() -> str | None:
    return session.get("user")


def login_required(view):
    from functools import wraps
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not current_user():
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapper


@app.route("/")
def index():
    if current_user():
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            flash("Username and password are required", "error")
            return redirect(url_for("signup"))
        db = load_users()
        if username in db.get("users", {}):
            flash("User already exists. Please login.", "error")
            return redirect(url_for("login"))
        db.setdefault("users", {})[username] = {
            "password_hash": generate_password_hash(password),
            "profile": {},
            "history": []
        }
        save_users(db)
        flash("Signup successful. Please login.", "success")
        return redirect(url_for("login"))
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        db = load_users()
        user = db.get("users", {}).get(username)
        if not user or not check_password_hash(user["password_hash"], password):
            flash("Invalid credentials", "error")
            return redirect(url_for("login"))
        session["user"] = username
        flash("Logged in", "success")
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    session.pop("user", None)
    flash("Logged out", "success")
    return redirect(url_for("index"))


@app.route("/dashboard", methods=["GET"]) 
@login_required
def dashboard():
    meta = dataset_metadata()
    return render_template("dashboard.html", meta=meta)


@app.route("/predict", methods=["POST"]) 
@login_required
def do_predict():
    # Build user_row from dataset metadata so the UI is dataset-driven
    meta = dataset_metadata()
    # Enforce required interest selection
    try:
        interest_col = (meta.get("filters", {}) or {}).get("interest_col")
    except Exception:
        interest_col = None
    if interest_col:
        submitted_interest = request.form.get(interest_col, "").strip()
        if not submitted_interest:
            flash("Please select an Interest before predicting.", "error")
            return redirect(url_for("dashboard"))
    user_row: Dict[str, Any] = {}

    # Categorical fields
    cat_meta = meta.get("categorical", {})
    filters = (meta.get("filters", {}) or {}).get("by_interest", {})
    selected_interest = request.form.get(interest_col, "").strip() if interest_col else None
    for col, options in cat_meta.items():
        val = request.form.get(col, options[0] if options else "")
        # Enforce that dependent categorical values must exist for the chosen interest
        if interest_col and col != interest_col and selected_interest:
            allowed_map = filters.get(selected_interest, {}) or {}
            allowed_vals = set((allowed_map.get(col) or []))
            if allowed_vals and val not in allowed_vals:
                flash(f"Selected value for '{col}' is not valid for Interest '{selected_interest}'.", "error")
                return redirect(url_for("dashboard"))
        user_row[col] = val

    # Numeric fields: UI provides normalized 0..1; map back to dataset scale
    for col, stats in meta.get("numeric", {}).items():
        raw = request.form.get(col, None)
        try:
            v = float(raw) if raw is not None and raw != "" else float(stats.get("mean", 0))
        except (TypeError, ValueError):
            v = float(stats.get("mean", 0))
        mn = float(stats.get("min", 0.0))
        mx = float(stats.get("max", 1.0))
        # clamp and denormalize: original = v*(max-min)+min
        v = max(0.0, min(1.0, v))
        user_row[col] = mn + v * (mx - mn)

    result = predict(user_row)
    username = current_user()
    db = load_users()
    entry = {
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "input": user_row,
        "prediction": result["career"],
        "top": result.get("top", []),
        "roadmap": result.get("roadmap", []),
    }
    db["users"][username]["history"].insert(0, entry)
    save_users(db)

    # store last result in session for print/export
    session["last_result"] = result
    session["last_input"] = user_row

    return render_template("result.html", result=result, user_row=user_row)


@app.route("/history")
@login_required
def history():
    db = load_users()
    user = db["users"][current_user()]
    return render_template("history.html", history=user.get("history", []))


def _roadmap_candidates(career: str):
    import re
    # Preserve case in output name, but normalize for filesystem lookup
    base = career.strip()
    keep_paren = re.sub(r"\s+", "_", base)  # spaces to underscore, keep parentheses
    no_paren = re.sub(r"[()]+", "", keep_paren)  # remove parentheses
    alnum = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")  # collapse to alnum underscores
    # Deduplicate while preserving order
    seen = set()
    variants = []
    for stem in [keep_paren, no_paren, alnum]:
        if stem not in seen:
            seen.add(stem)
            variants.append(stem)
    for stem in variants:
        yield f"{stem}_Roadmap.txt"


@app.route("/roadmap/download")
@login_required
def download_roadmap():
    data = session.get("last_result")
    if not data:
        flash("No recent prediction to download. Please generate a result first.", "error")
        return redirect(url_for("dashboard"))
    career_name = data.get("career") or ""
    folder = os.path.join("data", "Career_Roadmaps_Zipped")
    if not os.path.isdir(folder):
        flash("Roadmap files folder is missing.", "error")
        return redirect(url_for("dashboard"))
    # Try candidate filenames
    target_path = None
    for fname in _roadmap_candidates(str(career_name)):
        fp = os.path.join(folder, fname)
        if os.path.exists(fp):
            target_path = fp
            break
    if target_path is None:
        # As a fallback, try case-insensitive match on stems
        stem = str(career_name).lower().replace(" ", "_")
        try:
            for fn in os.listdir(folder):
                if fn.lower().startswith(stem) and fn.lower().endswith("_roadmap.txt"):
                    target_path = os.path.join(folder, fn)
                    break
        except Exception:
            target_path = None
    if target_path is None:
        flash(f"No roadmap file found for career: {career_name}", "error")
        return redirect(url_for("dashboard"))
    # Provide a friendly download name
    download_name = f"{career_name} - Roadmap.txt"
    return send_file(target_path, as_attachment=True, download_name=download_name)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)
