# frozen_string_literal: true

# Makes Quarto's folded code cells (`#| code-fold: true`) render.
#
# Quarto's gfm output wraps each folded cell in a `<details>` block with a
# markdown code fence inside. Kramdown does not parse markdown inside HTML
# blocks by default, so the fence would show up as raw text. Adding
# `markdown="1"` to the tag tells kramdown to parse its contents. Only these
# blocks are touched, so other raw HTML (e.g. pandas tables) is left alone.
module CodeFold
  DETAILS_TAG = /<details\b(?![^>]*\bmarkdown=)([^>]*)>/i

  Jekyll::Hooks.register :posts, :pre_render do |post|
    next unless post.relative_path.start_with?("_posts/notebooks/")

    post.content = post.content.gsub(DETAILS_TAG, '<details\1 markdown="1">')
  end
end
