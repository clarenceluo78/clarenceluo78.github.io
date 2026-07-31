---
title: "A General Style Showcase"
slug: "style-showcase"
tags:
  - patterns
  - systems
  - methods
toc: true
citation_key: styleshowcase
---

Good explanations make structure visible. This unpublished demonstration explores how Haoyan's Blog treats an opening argument, evidence, technical notation, and references without pretending to be a real essay.

<!--more-->

<p class="update-note"><strong>Demonstration:</strong> This draft exists only to evaluate the visual and editorial system of Haoyan's Blog. It is not intended for publication.</p>

## Begin with a useful question

A finished post should begin with the question or observation that makes it
worth reading. The opening does not need to announce every section. It should
give the reader a reason to continue and enough orientation to understand what
is at stake.

> A pattern becomes useful when it compresses several observations without
> hiding the differences that matter.

This sentence is a block quotation. Ordinary links can point to a
[primary source](https://jekyllrb.com/docs/posts/), while a footnote can hold a
helpful aside without interrupting the argument.[^aside]

### Move from observation to mechanism

Suppose an outcome $y$ depends on a set of interacting factors:

$$
y = f(x_1, x_2, \ldots, x_n).
$$

The notation should sit comfortably beside prose. Inline mathematics such as
$P(y \mid x)$ should also preserve the line’s rhythm.

## Use structure to carry the argument

Long posts benefit from hierarchy, but headings should express conceptual
movement rather than merely divide text. A compact comparison can clarify the
role of each section:

| Stage | Reader’s question | Writer’s task |
| --- | --- | --- |
| Observation | What happened? | Establish the phenomenon |
| Mechanism | Why might it happen? | Identify a causal or explanatory structure |
| Test | What would change our mind? | Expose evidence and uncertainty |

Lists are useful when the items genuinely form a set:

1. State the pattern precisely.
2. Separate observation from interpretation.
3. Look for cases that should break the explanation.

### Code should remain part of the reading flow

Technical posts can include compact, copyable code with a visible language
label:

```python
def shared_patterns(observations):
    """Return features that recur without erasing exceptions."""
    candidates = intersect(observations)
    return [item for item in candidates if survives_counterexample(item)]
```

The copy action is intentionally quiet. It appears as part of the code frame
rather than as a large interface element.

## Let figures support, not decorate

<figure class="demo-figure">
  <div class="demo-figure__plot" role="img" aria-label="Four bars increasing in height from left to right">
    <span class="demo-figure__bar" style="height: 28%"></span>
    <span class="demo-figure__bar" style="height: 47%"></span>
    <span class="demo-figure__bar" style="height: 68%"></span>
    <span class="demo-figure__bar" style="height: 91%"></span>
  </div>
  <figcaption>Figure 1. A deliberately generic figure demonstrating responsive sizing and a concise caption.</figcaption>
</figure>

Figures should earn their space by making a relationship easier to see. Alt
text describes the information carried by the visual, while the caption tells
the reader why it matters in context.

## End with what the reader should retain

A conclusion should not simply replay the introduction. It can state the
strongest supported claim, identify an unresolved uncertainty, or show what
the explanation changes. For this demonstration, the intended impression is a
calm reading surface with enough structure for long technical and nontechnical
posts.

{% include blog_citation.html %}

## References

1. Jekyll contributors. [“Posts.”](https://jekyllrb.com/docs/posts/) *Jekyll Documentation*.
2. Weng, Lilian. [“Reward Hacking in Reinforcement Learning.”](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) *Lil’Log*, 2024.

[^aside]: Footnotes are best reserved for qualifications, background, or
    tangents that are useful but not necessary to follow the main line.
