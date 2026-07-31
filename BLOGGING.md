# Haoyan's Blog authoring

Posts are plain Markdown. The `./blogctl` helper handles repetitive file setup and
validation, but it never commits, pushes, or changes Git branches.

## Start a draft

```sh
./blogctl new "Why Institutions Forget"
```

This creates:

```text
_drafts/why-institutions-forget.md
assets/blog/why-institutions-forget/
```

The smallest useful front matter is:

```yaml
---
title: "Why Institutions Forget"
slug: "why-institutions-forget"
tags:
  - institutions
  - incentives
toc: true
---
```

The opening paragraph becomes the 30-word post preview. Add `<!--more-->` after
the intended opening when you want an explicit excerpt boundary.

## Preview and check

```sh
./blogctl preview
./blogctl check
```

Drafts are visible locally but excluded from an ordinary production build.

## Publish locally

```sh
./blogctl publish why-institutions-forget
```

This validates the draft, adds the current date, moves it into `_posts/`, and
regenerates tag and pagination pages. Review the resulting diff and perform all
Git operations manually.

If posts are added or edited without the helper, run:

```sh
./blogctl sync
```

## Optional metadata

```yaml
last_modified_at: 2026-08-12
image: /assets/blog/why-institutions-forget/social-preview.png
toc: false
```

Use `last_modified_at` only for material changes. Minor typo corrections do not
need an update date.

## Citations

Use readable author-year links in the article and a final numbered References
section for research-heavy posts. Insert the reusable citation block immediately
before References:

```liquid
{% include blog_citation.html %}
```

An optional `citation_key` front-matter value customizes the BibTeX key.
