"""
Injini Relay — the one place a Hugging Face write token for the field
dataset actually lives. Phones (and this app's own manual-upload page)
never see it; they present a separate, low-privilege API key instead, and
this backend is the only thing that ever talks to Hugging Face directly.

Every accepted upload opens a pull request on the dataset repo rather than
committing straight to `main` (create_pr=True below) — a bad, garbled, or
malicious upload can never silently become training data. It sits as a PR
until a person reviews and merges it. See README.md for the full rationale
and the required Space secrets.
"""

import csv
import os
import re
import secrets
import tempfile
import time
import zipfile

from fastapi import Depends, FastAPI, File, Form, HTTPException, Header, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

HF_TOKEN = os.environ["HF_TOKEN"]
HF_DATASET_REPO = os.environ["HF_DATASET_REPO"]
UPLOAD_API_KEY = os.environ["UPLOAD_API_KEY"]
ADMIN_USER = os.environ.get("ADMIN_USER", "injini")
ADMIN_PASSWORD = os.environ["ADMIN_PASSWORD"]

# Comfortably above a real field export, still a hard ceiling — nothing
# unbounded ever reaches Hugging Face from here.
MAX_BYTES = 150 * 1024 * 1024

app = FastAPI(title="Injini Relay")
templates = Jinja2Templates(directory="templates")
security = HTTPBasic()
api = HfApi(token=HF_TOKEN)


def check_basic_auth(credentials: HTTPBasicCredentials = Depends(security)) -> None:
    ok_user = secrets.compare_digest(credentials.username, ADMIN_USER)
    ok_pass = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (ok_user and ok_pass):
        raise HTTPException(status_code=401, detail="Bad credentials", headers={"WWW-Authenticate": "Basic"})


def safe_source_id(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", raw.strip())[:64]
    return cleaned or "unknown"


def handle_upload(data: bytes, filename: str, source: str) -> dict:
    """Validate, then stage the zip as a pull request. Shared by both the
    phone's REST call and this page's own manual-upload form, so neither
    path is the "less safe" one."""
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too big ({len(data)} bytes, max {MAX_BYTES}).")
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Expected a .zip file.")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        try:
            with zipfile.ZipFile(tmp_path) as zf:
                names = zf.namelist()
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Not a valid zip file.")
        if not any(n.endswith("manifest.csv") for n in names):
            raise HTTPException(status_code=400, detail="Doesn't look like an Injini export (no manifest.csv inside).")

        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", filename)
        path_in_repo = f"field_exports/{safe_source_id(source)}/{stamp}_{safe_name}"

        try:
            commit_info = api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=path_in_repo,
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                commit_message=f"Field export from {source}",
                create_pr=True,
            )
        except HfHubHTTPError as e:
            raise HTTPException(status_code=502, detail=f"Hugging Face rejected the upload: {e}")

        pr_url = getattr(commit_info, "pr_url", None) or str(commit_info)
        return {"path_in_repo": path_in_repo, "pr_url": pr_url}
    finally:
        os.unlink(tmp_path)


@app.get("/health")
def health():
    return {"status": "ok", "repo": HF_DATASET_REPO}


@app.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    source: str = Form("phone"),
    x_api_key: str = Header(default=""),
):
    if not secrets.compare_digest(x_api_key, UPLOAD_API_KEY):
        raise HTTPException(status_code=401, detail="Bad or missing API key.")
    data = await file.read()
    result = handle_upload(data, file.filename or "upload.zip", source)
    return JSONResponse(result)


def _uploads_summary() -> list[dict]:
    """Where the data is coming from: every field-export PR, newest first — merged, still-pending, or closed."""
    uploads = []
    for d in api.get_repo_discussions(HF_DATASET_REPO, repo_type="dataset", discussion_type="pull_request"):
        uploads.append({
            "source": d.title.replace("Field export from ", "", 1),
            "status": d.status,
            "created_at": d.created_at.strftime("%Y-%m-%d %H:%M UTC"),
            "url": f"https://huggingface.co/datasets/{HF_DATASET_REPO}/discussions/{d.num}",
        })
    uploads.sort(key=lambda u: u["created_at"], reverse=True)
    return uploads


def _data_health() -> dict:
    """What the merged data actually looks like: healthy vs. faulty, per label, per source. Only counts what's been reviewed onto main — a pending PR isn't training data yet."""
    by_label: dict[str, int] = {}
    by_source: dict[str, dict[str, int]] = {}
    try:
        manifest_paths = [p for p in api.list_repo_files(HF_DATASET_REPO, repo_type="dataset") if p.endswith("manifest.csv")]
    except Exception:
        manifest_paths = []

    for path in manifest_paths:
        try:
            local_path = hf_hub_download(HF_DATASET_REPO, path, repo_type="dataset", token=HF_TOKEN)
        except Exception:
            continue
        parts = path.split("/")
        source = parts[1] if len(parts) > 2 and parts[0] == "field_exports" else "unknown"
        bucket = by_source.setdefault(source, {"healthy": 0, "faulty": 0})
        try:
            with open(local_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    label = (row.get("label") or "").strip()
                    if not label:
                        continue
                    by_label[label] = by_label.get(label, 0) + 1
                    bucket["healthy" if label == "Normal" else "faulty"] += 1
        except Exception:
            continue

    healthy = by_label.get("Normal", 0)
    total = sum(by_label.values())
    return {
        "healthy": healthy,
        "faulty": total - healthy,
        "total": total,
        "by_label": sorted(by_label.items(), key=lambda kv: -kv[1]),
        "by_source": sorted(by_source.items()),
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, _auth: None = Depends(check_basic_auth)):
    health = _data_health()
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "repo": HF_DATASET_REPO,
            "uploads": _uploads_summary()[:40],
            **health,
        },
    )


@app.get("/", response_class=HTMLResponse)
def upload_form(request: Request, _auth: None = Depends(check_basic_auth)):
    return templates.TemplateResponse(
        request, "upload.html", {"repo": HF_DATASET_REPO, "result": None, "error": None}
    )


@app.post("/", response_class=HTMLResponse)
async def upload_form_submit(
    request: Request,
    _auth: None = Depends(check_basic_auth),
    file: UploadFile = File(...),
    source: str = Form("manual"),
):
    data = await file.read()
    try:
        result = handle_upload(data, file.filename or "upload.zip", source)
        error = None
    except HTTPException as e:
        result = None
        error = e.detail
    return templates.TemplateResponse(
        request, "upload.html", {"repo": HF_DATASET_REPO, "result": result, "error": error}
    )
