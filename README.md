# blog

My personal blog site. 

The current theme is based on the [Typora Ursine Theme](https://github.com/noatpad/typora-theme-ursine).

## Notebook posts

Jupyter notebooks can be posts too. [Quarto](https://quarto.org) turns them into markdown.

1. Save the notebook as `_notebooks/YYYY-MM-DD-slug.ipynb`.
2. Make its first cell a **Raw** cell with the usual front matter:
   ```yaml
   ---
   title: My post
   date: 2026-11-26
   permalink: /posts/2026/11/slug/
   tags:
     - nlp
   ---
   ```
3. Run the notebook and save it. Quarto uses the saved outputs and never re-runs code.

The deploy workflow renders notebooks into `_posts/notebooks/` (git-ignored) before the Jekyll build.
To preview locally, run `quarto render _notebooks` first, then `bundle exec jekyll serve`.
