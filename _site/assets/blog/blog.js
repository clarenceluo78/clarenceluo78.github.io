(function () {
  "use strict";

  var root = document.documentElement;
  var themeButton = document.querySelector(".theme-toggle");

  function currentTheme() {
    return root.dataset.theme === "dark" ? "dark" : "light";
  }

  function updateThemeLabel() {
    if (!themeButton) return;
    var next = currentTheme() === "dark" ? "light" : "dark";
    themeButton.setAttribute("aria-label", "Switch to " + next + " theme");
    themeButton.setAttribute("title", "Switch to " + next + " theme");
  }

  if (themeButton) {
    updateThemeLabel();
    themeButton.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      localStorage.setItem("blog-theme", next);
      updateThemeLabel();
    });
  }

  function slugify(value) {
    return value
      .toLowerCase()
      .trim()
      .replace(/<[^>]*>/g, "")
      .replace(/[^\w\s-]/g, "")
      .replace(/[\s_-]+/g, "-")
      .replace(/^-+|-+$/g, "") || "section";
  }

  function uniqueHeadingId(base, used) {
    var id = base;
    var suffix = 2;
    while (used[id]) {
      id = base + "-" + suffix;
      suffix += 1;
    }
    used[id] = true;
    return id;
  }

  function buildTableOfContents() {
    var content = document.querySelector(".post-content");
    var target = document.getElementById("post-toc-list");
    if (!content || !target) return;

    var headings = Array.prototype.slice.call(content.querySelectorAll("h2, h3"));
    if (headings.length < 3) {
      var aside = document.querySelector(".post-toc");
      if (aside) aside.hidden = true;
      return;
    }

    var used = {};
    var list = document.createElement("ol");
    var links = [];

    headings.forEach(function (heading) {
      var id = heading.id || uniqueHeadingId(slugify(heading.textContent), used);
      heading.id = id;

      var anchor = document.createElement("a");
      anchor.className = "heading-anchor";
      anchor.href = "#" + id;
      anchor.setAttribute("aria-label", "Link to " + heading.textContent.trim());
      anchor.textContent = "#";
      heading.prepend(anchor);

      var item = document.createElement("li");
      if (heading.tagName === "H3") item.className = "toc-depth-3";
      var link = document.createElement("a");
      link.href = "#" + id;
      link.textContent = heading.textContent.replace(/^#/, "").trim();
      item.appendChild(link);
      list.appendChild(item);
      links.push({heading: heading, link: link});
    });

    target.appendChild(list);

    if ("IntersectionObserver" in window) {
      var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          links.forEach(function (pair) {
            pair.link.classList.toggle("is-active", pair.heading === entry.target);
          });
        });
      }, {rootMargin: "-15% 0px -70% 0px", threshold: 0});
      headings.forEach(function (heading) { observer.observe(heading); });
    }

    var tocDetails = document.querySelector(".post-toc__details");
    if (tocDetails && window.matchMedia("(max-width: 860px)").matches) {
      tocDetails.removeAttribute("open");
    }
  }

  function languageLabel(pre) {
    var code = pre.querySelector("code");
    var classes = ((code && code.className) || "") + " " + ((pre.parentElement && pre.parentElement.className) || "");
    var match = classes.match(/language-([a-z0-9_+-]+)/i);
    return match ? match[1] : "Code";
  }

  function copyText(value, button) {
    function confirmCopy() {
      var original = button.textContent;
      button.textContent = "Copied";
      window.setTimeout(function () { button.textContent = original; }, 1400);
    }

    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(value).then(confirmCopy);
      return;
    }

    var helper = document.createElement("textarea");
    helper.value = value;
    helper.style.position = "fixed";
    helper.style.opacity = "0";
    document.body.appendChild(helper);
    helper.select();
    document.execCommand("copy");
    helper.remove();
    confirmCopy();
  }

  function enhanceCodeBlocks() {
    document.querySelectorAll(".post-content pre").forEach(function (pre) {
      if (pre.closest(".code-frame") || pre.dataset.enhanced === "true") return;
      pre.dataset.enhanced = "true";

      var wrapper = pre.parentElement;
      if (!wrapper.classList.contains("highlight")) {
        wrapper = document.createElement("div");
        wrapper.className = "highlight";
        pre.parentNode.insertBefore(wrapper, pre);
        wrapper.appendChild(pre);
      }

      var toolbar = document.createElement("div");
      toolbar.className = "code-toolbar";
      var label = document.createElement("span");
      label.textContent = languageLabel(pre);
      var button = document.createElement("button");
      button.className = "code-copy";
      button.type = "button";
      button.textContent = "Copy";
      button.setAttribute("aria-label", "Copy code");
      button.addEventListener("click", function () {
        copyText(pre.textContent, button);
      });
      toolbar.appendChild(label);
      toolbar.appendChild(button);
      wrapper.insertBefore(toolbar, wrapper.firstChild);
    });
  }

  function enhanceCitationCopy() {
    document.querySelectorAll(".code-frame .code-copy").forEach(function (button) {
      button.addEventListener("click", function () {
        var pre = button.closest(".code-frame").querySelector("pre");
        if (pre) copyText(pre.textContent, button);
      });
    });
  }

  var copyLinkButton = document.querySelector(".copy-link");
  if (copyLinkButton) {
    copyLinkButton.addEventListener("click", function () {
      copyText(copyLinkButton.dataset.url || window.location.href, copyLinkButton.querySelector(".copy-link__label"));
    });
  }

  buildTableOfContents();
  enhanceCodeBlocks();
  enhanceCitationCopy();
}());
