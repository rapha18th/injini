---
title: Injini Relay
emoji: 🛠️
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Injini Relay

The one place a Hugging Face **write** token for the field dataset actually
lives. The Injini Android app never holds it — it can't ship a bundled HF
token, and shouldn't. Phones, and this Space's own manual-upload page,
authenticate with a separate, low-privilege **upload key** instead. If that
key ever leaks out of a decompiled APK, the worst it buys someone is the
ability to open a pull request against the dataset — never to write to it
directly, since every accepted upload becomes a PR, never a direct commit.

## Set the Space to "Protected" visibility

Settings → Visibility → **Protected** (needs a Pro account; you have one).
Protected keeps this repo's source (`app.py`, `Dockerfile`, this README)
hidden from public Hub browsing, while the running app's URL stays publicly
reachable — exactly what a phone calling `/api/upload` over plain HTTP
needs, with no Hugging Face auth token required just to reach the endpoint.
Plain **Public** would work too but exposes the source; **Private** would
require every caller (including the phone) to present an actual HF account
token just to reach the Space at all, which defeats the entire point of this
relay. Protected is the correct middle setting for this use case.

## Required Space secrets (Settings → Variables and secrets)

- `HF_TOKEN` — a Hugging Face **write** token. Scope it to `HF_DATASET_REPO`
  only if you create a fine-grained token; don't use a full-account one.
- `HF_DATASET_REPO` — e.g. `username/injini-field-data`.
- `UPLOAD_API_KEY` — the key phones present as `X-Api-Key`. Generate one
  with `openssl rand -hex 24` (or any long random string) — this is what
  goes into the Injini app's "Sync to Hugging Face" screen, never the HF
  token itself.
- `ADMIN_PASSWORD` — password for this page's manual-upload form. Username
  is `injini` unless you also set `ADMIN_USER`.

None of these go in git. Set them in the Space's Settings tab; they're
injected as environment variables at runtime.

## Endpoints

- `POST /api/upload` — multipart form: `file` (a `.zip`), optional `source`
  text field, header `X-Api-Key: <UPLOAD_API_KEY>`. Returns the pull request
  it opened.
- `GET /` and `POST /` — the same upload, as an HTTP-Basic-protected web
  form, for exports collected manually (WhatsApp, Drive) and moved here from
  a laptop rather than sent directly from a phone.
- `GET /health` — liveness check, no auth, returns the configured repo id.

## Why a pull request, not a direct commit

`api.upload_file(..., create_pr=True)` is the one line that matters most
here. A bad, garbled, or malicious upload — a corrupted zip, a spam attempt
against a leaked API key, a genuinely bad field recording — can never
silently become part of the training data. It sits as a proposed change
until a person actually looks at it and merges. Point any training pipeline
(the Kaggle notebooks in `../notebooks/`) at the reviewed `main` branch, not
at open PRs.

Every upload also lands under its own `field_exports/<source>/<timestamp>_
<file>` path — nothing is ever overwritten, and a bad batch from one device
is always traceable and revertable without touching anyone else's data.

## Deploying

Uses the current `hf` CLI (`pip install -U huggingface_hub`, or the
standalone installer at `hf.co/cli`) — the older `huggingface-cli` name
still works but `hf` is what's documented now.

```bash
# from this directory
hf auth login
hf repos create injini-relay --repo-type space --sdk docker
hf upload <your-username>/injini-relay . --repo-type space
```

Then, in the Space's Settings tab: set the four secrets above, and set
Visibility to **Protected**. The app builds and starts automatically —
`GET /health` should return `{"status": "ok", ...}` once it's up. Re-run the
`hf upload` command any time this directory changes to push an update.
