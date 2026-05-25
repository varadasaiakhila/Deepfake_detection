# Streamlit deployment

## Live app URL

**https://nv6w6xbqkmepvyxlzrcd5y.streamlit.app**

(GitHub repo: `varadasaiakhila/Deepfake_detection`, branch `main`, entry `app.py`)

## Cloud settings (required)

In [share.streamlit.io](https://share.streamlit.io) → your app → **Settings** → **Advanced**:

| Setting | Value |
|---------|--------|
| Python version | **3.11** |
| Requirements file | `requirements-cloud.txt` |

Then click **Reboot app** (or push to `main` to auto-redeploy).

## Models on Cloud

Included in Git: `saved_models/cnn_*.keras` (4 region CNNs).

Not in Git (too large / gitignored): `cvit_face.pth`, `posture_model.keras` — face/posture show ❌ on Cloud unless you add Git LFS or host weights elsewhere.

## Redeploy after git push

```bash
git push origin main
```

Wait 3–10 minutes, then open the URL above.
