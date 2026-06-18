# PointGT — Project Page

Project website for **PointGT: Simultaneous Geometry and Texture Editing for Point-Based Representations** (ECCV 2026).

Live site: https://zvict.github.io/pointgt/

## Structure
- `index.html` — the page (edit content here)
- `static/` — css, js, images, videos, pdfs
- `tools/check_site.py` — sanity checker (asset links, placeholders, file sizes)

## Local preview
```bash
python3 -m http.server 8000
# open http://localhost:8000
```

## Deploy (GitHub Pages)
1. Create an empty repo named `pointgt` under your GitHub account (no README/license).
2. Add the remote and push:
   ```bash
   git remote add origin https://github.com/zvict/pointgt.git
   git push -u origin main
   ```
3. On GitHub: **Settings → Pages → Source: Deploy from a branch → `main` / `/ (root)` → Save**.
4. The site is live at https://zvict.github.io/pointgt/ within a minute or two.

## Manual follow-ups (TODO)
- Upload the paper PDF to `static/pdfs/paper.pdf` (the "Paper" button links here; compress the 192 MB camera-ready first, e.g. with `gs`).
- Add per-author homepage links — replace `href="#"` in `index.html`.
- When available, replace the disabled "arXiv (coming soon)" / "Code (coming soon)" buttons with live links.
- Optionally replace `static/images/favicon.png` and `static/images/social_preview.png` with custom artwork.

## Credit
Built on the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) (CC BY-SA 4.0).
