(function () {
  "use strict";

  var searchRoot = document.querySelector(".search");
  if (!searchRoot) return;

  var input = document.getElementById("blog-search-input");
  var tagContainer = document.getElementById("search-tag-filters");
  var resultsContainer = document.getElementById("search-results");
  var status = document.getElementById("search-status");
  var selectedTags = new Set();
  var posts = [];

  function escapeHtml(value) {
    var div = document.createElement("div");
    div.textContent = value == null ? "" : String(value);
    return div.innerHTML;
  }

  function normalized(value) {
    return String(value || "").toLowerCase();
  }

  function updateAddress() {
    var params = new URLSearchParams();
    var query = input.value.trim();
    if (query) params.set("q", query);
    Array.from(selectedTags).sort().forEach(function (tag) { params.append("tag", tag); });
    var suffix = params.toString();
    history.replaceState(null, "", window.location.pathname + (suffix ? "?" + suffix : ""));
  }

  function renderResults() {
    var terms = normalized(input.value).split(/\s+/).filter(Boolean);
    var tags = Array.from(selectedTags);

    var matches = posts.filter(function (post) {
      var searchable = normalized([
        post.title,
        post.excerpt,
        post.content,
        (post.tags || []).join(" ")
      ].join(" "));
      var matchesWords = terms.every(function (term) { return searchable.indexOf(term) !== -1; });
      var postTags = (post.tags || []).map(normalized);
      var matchesTags = tags.every(function (tag) { return postTags.indexOf(normalized(tag)) !== -1; });
      return matchesWords && matchesTags;
    });

    matches.sort(function (a, b) {
      if (terms.length) {
        var aTitle = normalized(a.title);
        var bTitle = normalized(b.title);
        var aScore = terms.reduce(function (score, term) { return score + (aTitle.indexOf(term) !== -1 ? 1 : 0); }, 0);
        var bScore = terms.reduce(function (score, term) { return score + (bTitle.indexOf(term) !== -1 ? 1 : 0); }, 0);
        if (aScore !== bScore) return bScore - aScore;
      }
      return b.date.localeCompare(a.date);
    });

    status.textContent = matches.length + (matches.length === 1 ? " post" : " posts");
    resultsContainer.innerHTML = matches.map(function (post) {
      var tagText = (post.tags || []).join(" · ");
      return [
        '<article class="search-result">',
        '<h2><a href="' + escapeHtml(post.url) + '">' + escapeHtml(post.title) + "</a></h2>",
        '<div class="search-result__meta">' + escapeHtml(post.dateLabel) + (tagText ? " · " + escapeHtml(tagText) : "") + "</div>",
        "<p>" + escapeHtml(post.excerpt) + "</p>",
        "</article>"
      ].join("");
    }).join("");

    if (!matches.length) {
      resultsContainer.innerHTML = '<section class="empty-state"><p>No matching posts.</p></section>';
    }
    updateAddress();
  }

  function renderTags() {
    var tags = Array.from(new Set(posts.reduce(function (all, post) {
      return all.concat(post.tags || []);
    }, []))).sort(function (a, b) { return a.localeCompare(b); });

    tagContainer.innerHTML = "";
    tags.forEach(function (tagName) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "tag";
      button.textContent = tagName;
      button.setAttribute("aria-pressed", selectedTags.has(tagName) ? "true" : "false");
      button.addEventListener("click", function () {
        if (selectedTags.has(tagName)) {
          selectedTags.delete(tagName);
        } else {
          selectedTags.add(tagName);
        }
        button.setAttribute("aria-pressed", selectedTags.has(tagName) ? "true" : "false");
        renderResults();
      });
      tagContainer.appendChild(button);
    });
  }

  var params = new URLSearchParams(window.location.search);
  input.value = params.get("q") || "";
  params.getAll("tag").forEach(function (tag) { selectedTags.add(tag); });

  input.addEventListener("input", renderResults);

  fetch(searchRoot.dataset.searchIndex)
    .then(function (response) {
      if (!response.ok) throw new Error("Search index unavailable");
      return response.json();
    })
    .then(function (data) {
      posts = data;
      renderTags();
      renderResults();
    })
    .catch(function () {
      status.textContent = "Search is temporarily unavailable.";
    });
}());
