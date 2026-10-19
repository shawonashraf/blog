# blog

My personal blog site. 

The current theme is based on the [Typora Ursine Theme](https://github.com/noatpad/typora-theme-ursine).

## Writing posts with Typora

Drag an image into a post and Typora copies it to `_posts/assets/` and links it as `assets/<name>`. The `_plugins/typora_assets.rb` plugin publishes everything in `_posts/assets/` to `/assets/images/posts/` and rewrites relative image paths (`assets/…`, `./assets/…`, `../assets/images/posts/…`) at build time, so no Liquid `relative_url` wrapping is needed. Avoid reusing a filename that already exists in `assets/images/posts/`, since both end up in the same output directory.
