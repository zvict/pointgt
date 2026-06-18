# PointGT — Project Page

Project website for **PointGT: Simultaneous Geometry and Texture Editing for Point-Based Representations** (ECCV 2026).

Live site: https://zvict.github.io/pointgt/

> This is the **`gh-pages`** branch — it holds only the project website. The PointGT **code** lives on the **`main`** branch of the same `zvict/pointgt` repo. The two branches have independent histories (standard for a GitHub Pages branch).

## Structure
- `index.html` — the page (edit content here)
- `static/` — css, js, images, videos, pdfs
- `tools/check_site.py` — sanity checker (asset links, placeholders, file sizes)

## Local preview
```bash
python3 -m http.server 8000
# open http://localhost:8000
```

## Deploy (GitHub Pages — `gh-pages` branch)
The website is served from the `gh-pages` branch; your code lives on `main` of the same `zvict/pointgt` repo.

1. Point this repo at the combined GitHub repo and push the website branch:
   ```bash
   git remote add origin https://github.com/zvict/pointgt.git   # skip if already set
   git push -u origin gh-pages
   ```
   (Push your code to `main` of the same repo separately, e.g. from the code checkout: `git push origin main`.)
2. On GitHub: **Settings → Pages → Source: Deploy from a branch → `gh-pages` / `/ (root)` → Save**.
3. The site is live at https://zvict.github.io/pointgt/ within a minute or two.

To update the site later, commit to `gh-pages` and `git push origin gh-pages`.

## Manual follow-ups (TODO)
- Upload the paper PDF to `static/pdfs/paper.pdf` (the "Paper" button links here; compress the 192 MB camera-ready first, e.g. with `gs`).
- Add per-author homepage links — replace `href="#"` in `index.html`.
- When available, replace the disabled "arXiv (coming soon)" / "Code (coming soon)" buttons with live links.
- Optionally replace `static/images/favicon.png` and `static/images/social_preview.png` with custom artwork.

## Credit
Built on the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) (CC BY-SA 4.0).
