# frozen_string_literal: true

# Typora image support.
#
# Typora copies drag-and-dropped images into an `assets/` directory next to
# the markdown file and references them with a relative path, e.g.
# `![x](assets/x.png)` or `<img src="./assets/x.png">`. For posts that means
# `_posts/assets/`, which Jekyll ignores by default.
#
# This plugin:
#   1. publishes every file under `_posts/assets/` to `/assets/images/posts/`
#      (same place as the hand-placed images, so the compression step in the
#      deploy workflow covers them too), and
#   2. rewrites relative image paths in posts to the published URL, resolved
#      against the post's own directory, so `assets/x.png`, `./assets/x.png`
#      and `../assets/images/posts/x.png` all work on the live site.
module TyporaAssets
  SOURCE_DIR = "_posts/assets"
  OUTPUT_DIR = "/assets/images/posts"

  class AssetFile < Jekyll::StaticFile
    def initialize(site, base, dir, name, output_dir)
      super(site, base, dir, name)
      @output_dir = output_dir
    end

    def url
      @url ||= File.join(@output_dir, @name)
    end
  end

  class Generator < Jekyll::Generator
    safe true
    priority :high

    def generate(site)
      root = File.join(site.source, SOURCE_DIR)
      return unless File.directory?(root)

      Dir.glob("**/*", base: root).sort.each do |rel|
        next unless File.file?(File.join(root, rel))

        sub = File.dirname(rel)
        dir = sub == "." ? SOURCE_DIR : File.join(SOURCE_DIR, sub)
        out = sub == "." ? OUTPUT_DIR : File.join(OUTPUT_DIR, sub)
        site.static_files << AssetFile.new(site, site.source, dir, File.basename(rel), out)
      end
    end
  end

  MARKDOWN_IMAGE = /(!\[[^\]]*\]\()\s*([^)\s"']+)(\s+"[^"]*")?(\))/
  HTML_SRC = /(<img\b[^>]*?\bsrc=)(["'])([^"']+)\2/i

  # Rewrites one path as it appears in the markdown. Returns nil when the
  # path should be left alone (absolute, anchor, Liquid, URL with a scheme,
  # or a relative path that does not point at anything in the site source).
  def self.resolve(path, post_dir, site)
    return nil if path.empty?
    return nil if path.start_with?("/", "#", "{") || path =~ %r{\A[a-z][a-z0-9+.-]*:}i

    source = Pathname.new(site.source)
    abs = Pathname.new(File.expand_path(path, File.join(site.source, post_dir)))
    rel = abs.relative_path_from(source).to_s
    return nil if rel.start_with?("..")

    target =
      if rel.start_with?("#{SOURCE_DIR}/")
        rel.sub(SOURCE_DIR, OUTPUT_DIR)
      elsif File.file?(Jekyll::URL.unescape_path(abs.to_s))
        "/#{rel}"
      end
    return nil unless target

    File.join(site.config["baseurl"].to_s, target).gsub(%r{/{2,}}, "/")
  end

  def self.rewrite(content, post_dir, site)
    content
      .gsub(MARKDOWN_IMAGE) do
        pre, path, title, close = Regexp.last_match.captures
        "#{pre}#{resolve(path, post_dir, site) || path}#{title}#{close}"
      end
      .gsub(HTML_SRC) do
        pre, quote, path = Regexp.last_match.captures
        "#{pre}#{quote}#{resolve(path, post_dir, site) || path}#{quote}"
      end
  end

  # Runs after reading, not at pre-render, because the excerpt shown on the
  # index pages is cut from the content at read time and never goes through
  # the per-post render hooks.
  Jekyll::Hooks.register :site, :post_read do |site|
    site.posts.docs.each do |post|
      rewritten = rewrite(post.content.to_s, File.dirname(post.relative_path), site)
      next if rewritten == post.content

      post.content = rewritten
      post.data["excerpt"] = Jekyll::Excerpt.new(post) if post.data["excerpt"].is_a?(Jekyll::Excerpt)
    end
  end
end
