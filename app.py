# /app.py
import json
import os
import smtplib
import uuid
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import wraps
from pathlib import Path
from typing import Optional

import joblib
from cryptography.fernet import Fernet
from flask import Flask, request, jsonify, render_template, render_template_string, session, redirect, url_for
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from huggingface_hub import login
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from waitress import serve
# ------------------------------------------------------------
# âœ… Local .env support (safe no-op in production)
# ------------------------------------------------------------
def _load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass


_load_dotenv_if_present()


# ------------------------------------------------------------
# âœ… Env helpers
# ------------------------------------------------------------
def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return val


def _require_env(name: str) -> str:
    val = _env(name)
    if val is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def _is_production() -> bool:
    return (_env("APP_ENV", "development") or "development").lower() == "production"


# ------------------------------------------------------------
# ðŸ§  APP CONFIGURATION
# ------------------------------------------------------------
app = Flask(__name__, template_folder="templates")
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Secret key MUST be env var in production
if _is_production():
    app.secret_key = _require_env("APP_SECRET_KEY")
else:
    app.secret_key = _env("APP_SECRET_KEY", "dev-only-insecure-fallback") or "dev-only-insecure-fallback"


# Directories (env override allowed)
UPLOAD_DIR = _env("UPLOAD_DIR", "uploads") or "uploads"
TRANSCRIPT_DIR = _env("TRANSCRIPT_DIR", "transcriptions") or "transcriptions"
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(TRANSCRIPT_DIR).mkdir(parents=True, exist_ok=True)

# Feature flags (text-only launch)
MEDIA_ENABLED = (_env("MEDIA_ENABLED", "false") or "false").lower() == "true"

# Admin config (password hash preferred; plaintext allowed only in dev)
ADMIN_USERNAME = _env("ADMIN_USERNAME", "admin") or "admin"
ADMIN_PASSWORD_HASH = _env("ADMIN_PASSWORD_HASH")
ADMIN_PASSWORD = _env("ADMIN_PASSWORD")  # dev-only convenience; do NOT set in production

# Model config
MODEL_PATH = _env("MODEL_PATH", "FINAL_TRUE_MULTI_LABEL_model.joblib") or "FINAL_TRUE_MULTI_LABEL_model.joblib"
DEFAULT_THRESHOLD = float(_env("DEFAULT_THRESHOLD", "0.15") or "0.15")


# ------------------------------------------------------------
# ðŸ¤— Optional Hugging Face login (only if you use HF features)
# ------------------------------------------------------------
HF_TOKEN = _env("HF_API_TOKEN")
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
    except Exception as e:
        print(f"âš ï¸ Hugging Face login issue: {e}")


# ------------------------------------------------------------
# âš¡ TRUE Multi-Label MODEL LOADING
# ------------------------------------------------------------
print("ðŸ” Loading TRUE Multi-Label Fallacy Model...")
model_package = joblib.load(MODEL_PATH)

# Your exported joblib should contain these keys
ai_model = model_package["model"]
label_binarizer = model_package["label_binarizer"]

print("âœ… Multi-Label Fallacy Model Ready.")


# ------------------------------------------------------------
# ðŸ‘§ Simple 5thâ€“6th grade explanations
# ------------------------------------------------------------
SIMPLE_EXPLANATIONS = {
    "Ad Hominem": "This attacks the person instead of the argument.",
    "Straw Man": "This changes what someone said into something easier to attack.",
    "Bandwagon": "This says something is true because lots of people believe it.",
    "Appeal to Authority": "This says something is true just because an expert said it.",
    "Appeal to Emotion": "This tries to win by making you feel strong emotions instead of using facts.",
    "Appeal to Tradition": "This says something is right because it has always been done that way.",
    "Appeal to Novelty": "This says something is better just because it is new.",
    "False Dilemma": "This acts like there are only two choices when there are more.",
    "Slippery Slope": "This says one small step will lead to a huge disaster without proof.",
    "Hasty Generalization": "This makes a big conclusion from too few examples.",
    "Post Hoc Ergo Propter Hoc": "This says one thing caused another just because it happened first.",
    "Correlation vs Causation": "This assumes two things happening together means one caused the other.",
    "Red Herring": "This changes the topic to distract from the real issue.",
    "Tu Quoque": "This says someone is wrong because they do not follow their own advice.",
    "No True Scotsman": "This changes the definition to avoid being proven wrong.",
    "Begging the Question": "This uses the conclusion as part of the proof.",
    "Circular Reasoning": "This repeats the same idea instead of proving it.",
}


# ------------------------------------------------------------
# ðŸ” ENCRYPTION (env key in production, dev fallback ok)
# ------------------------------------------------------------
FERNET_KEY = _env("TRANSCRIPT_FERNET_KEY")

if _is_production():
    # Production MUST provide this
    if not FERNET_KEY:
        raise RuntimeError("Missing required environment variable: TRANSCRIPT_FERNET_KEY")
    cipher = Fernet(FERNET_KEY.encode("utf-8"))
else:
    # Dev fallback: create/load a local key file
    dev_key_path = Path(_env("DEV_FERNET_KEY_PATH", "secret.key") or "secret.key")
    if dev_key_path.exists():
        dev_key = dev_key_path.read_bytes()
    else:
        dev_key = Fernet.generate_key()
        dev_key_path.write_bytes(dev_key)
    cipher = Fernet(dev_key)


def encrypt_text(text: str) -> bytes:
    return cipher.encrypt(text.encode("utf-8"))


def decrypt_file(path: str) -> str:
    return cipher.decrypt(Path(path).read_bytes()).decode("utf-8")


# --------------------------------------------------------------------
# ðŸ“§ EMAIL ALERT SYSTEM (all env vars; can be disabled)
# --------------------------------------------------------------------
EMAIL_ALERTS_ENABLED = (_env("EMAIL_ALERTS_ENABLED", "false") or "false").lower() == "true"

SMTP_HOST = _env("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(_env("SMTP_PORT", "465") or "465")
SMTP_USE_SSL = (_env("SMTP_USE_SSL", "true") or "true").lower() == "true"

ALERT_FROM = _env("ALERT_FROM")  # e.g. Fallacy01.LogicalApp@gmail.com
ALERT_TO = _env("ALERT_TO")      # e.g. Fallacy01.LogicalApp@gmail.com
SMTP_USERNAME = _env("SMTP_USERNAME")  # usually same as ALERT_FROM
SMTP_PASSWORD = _env("SMTP_PASSWORD")  # app password for Gmail (not your real password)


def send_email_notification(subject: str, msg: str) -> None:
    if not EMAIL_ALERTS_ENABLED:
        return

    # In production, if alerts are enabled, require all SMTP fields.
    if _is_production():
        _ = _require_env("ALERT_FROM")
        _ = _require_env("ALERT_TO")
        _ = _require_env("SMTP_USERNAME")
        _ = _require_env("SMTP_PASSWORD")

    if not (ALERT_FROM and ALERT_TO and SMTP_USERNAME and SMTP_PASSWORD):
        print("âš ï¸ Email alerts enabled but missing SMTP env vars; skipping email.")
        return

    try:
        m = MIMEMultipart()
        m["From"] = ALERT_FROM
        m["To"] = ALERT_TO
        m["Subject"] = subject
        m.attach(MIMEText(msg, "plain"))

        if SMTP_USE_SSL:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as s:
                s.login(SMTP_USERNAME, SMTP_PASSWORD)
                s.send_message(m)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
                s.starttls()
                s.login(SMTP_USERNAME, SMTP_PASSWORD)
                s.send_message(m)

        print(f"ðŸ“§ Email sent: {subject}")
    except Exception as e:
        print(f"âš ï¸ Email failed: {e}")


# --------------------------------------------------------------------
# ðŸ” ADMIN LOGIN (env vars; no hardcoded credentials)
# --------------------------------------------------------------------
ADMIN_USERNAME = _env("ADMIN_USERNAME", "admin") if not _is_production() else _require_env("ADMIN_USERNAME")

# Prefer ADMIN_PASSWORD_HASH in production.
ADMIN_PASSWORD_HASH = _env("ADMIN_PASSWORD_HASH")
ADMIN_PASSWORD = _env("ADMIN_PASSWORD")  # fallback for dev only

if _is_production():
    if not ADMIN_PASSWORD_HASH:
        raise RuntimeError("Missing required environment variable: ADMIN_PASSWORD_HASH")
else:
    if not ADMIN_PASSWORD_HASH:
        if ADMIN_PASSWORD:
            ADMIN_PASSWORD_HASH = generate_password_hash(ADMIN_PASSWORD)
        else:
            # Dev-only convenience default; DO NOT rely on this in production.
            ADMIN_PASSWORD_HASH = generate_password_hash("dev-admin-password-change-me")


def login_required(f):
    @wraps(f)
    def wrap(*a, **kw):
        if not session.get("logged_in"):
            return redirect(url_for("admin_login"))
        return f(*a, **kw)

    return wrap


@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        u = request.form.get("username")
        p = request.form.get("password")

        if u == ADMIN_USERNAME and p and check_password_hash(ADMIN_PASSWORD_HASH, p):
            session["logged_in"] = True
            send_email_notification(
                "âœ… Admin Login",
                f"{u} from {request.remote_addr} at {datetime.now().isoformat()}",
            )
            return redirect(url_for("admin_dashboard"))

        return "Invalid credentials", 401

    return """
    <h2>Admin Login</h2>
    <form method=post>
      Username:<br><input name=username><br>
      Password:<br><input type=password name=password><br><br>
      <button type=submit>Login</button>
    </form>
    """


@app.route("/admin/dashboard")
@login_required
def admin_dashboard():
    files = [f for f in os.listdir(TRANSCRIPT_DIR) if f.endswith(".txt")]
    return render_template_string(
        """
    <h2>Admin Dashboard</h2>
    <ul>
    {% for f in files %}
      <li>{{f}} â€“ <a href='/admin/view/{{f}}'>View</a> | <a href='/admin/delete/{{f}}'>Delete</a></li>
    {% endfor %}
    </ul>
    <a href='/admin/logout'>Logout</a>
    """,
        files=files,
    )


@app.route("/admin/view/<f>")
@login_required
def admin_view(f):
    p = os.path.join(TRANSCRIPT_DIR, secure_filename(f))
    if not os.path.exists(p):
        return "Not found", 404
    return f"<pre>{decrypt_file(p)}</pre>"


@app.route("/admin/delete/<f>")
@login_required
def admin_delete(f):
    p = os.path.join(TRANSCRIPT_DIR, secure_filename(f))
    if os.path.exists(p):
        os.remove(p)
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("admin_login"))


# --------------------------------------------------------------------
# ðŸ§  FALLACY DETECTION (kept as-is; thresholds can be env-configured later)
# --------------------------------------------------------------------
@app.route("/analyze", methods=["POST"])
@limiter.limit("10 per minute")
def analyze():
    from ai_reasoning_engine import analyze_fallacy_json
    import re

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"fallacies": [], "message": "Please enter text to analyze."}), 400

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    if len(sentences) > 10:
        return jsonify({
            "fallacies": [],
            "message": "Maximum 10 sentences per analyze."
        }), 400

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        if sentence_word_count > 35:
            return jsonify({
                "fallacies": [],
                "message": "Maximum 35 words per sentence."
            }), 400

    formatted = []

    for sentence in sentences:
        result = analyze_fallacy_json(sentence)
        analysis = result.get("analysis", [])

        for item in analysis:
            fallacies = item.get("fallacies", [])

            for f in fallacies:
                formatted.append({
                    "sentence": sentence,
                    "label": f.get("name", "Unknown"),
                    "percent": int(f.get("confidence", 0)),
                    "explanation": (f.get("explanation") or "").strip()
                })

    if not formatted:
        return jsonify({"fallacies": [], "message": "No logical fallacies detected."})

    return jsonify({"fallacies": formatted})

    if not formatted:
        return jsonify({"fallacies": [], "message": "No logical fallacies detected."})

    return jsonify({"fallacies": formatted})
    


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")
 
@app.route("/healthz", methods=["GET"])
def healthz():
    try:
        _ = ai_model.predict_proba(["health check"])[0]
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500

if __name__ == "__main__":
    host = _env("HOST", "0.0.0.0") or "0.0.0.0"
    port = int(_env("PORT", "5000") or "5000")
    serve(app, host=host, port=port)

@app.route("/terms")
def terms():
    return render_template("terms.html")


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


@app.route("/disclaimer")
def disclaimer():
    return render_template("disclaimer.html")









