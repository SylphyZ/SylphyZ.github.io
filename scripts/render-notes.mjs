import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const { marked } = require("marked");
const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(scriptDir, "..");
const notebookDir = path.join(rootDir, "notebook");

marked.setOptions({ breaks: false, gfm: true });

function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const fullPath = path.join(dir, entry.name);
    return entry.isDirectory() ? walk(fullPath) : [fullPath];
  });
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function stripMarkdown(value) {
  return value
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/[*_`>#]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeFootnotes(markdown) {
  return markdown
    .replace(/^\[\^([^\]]+)\]:\s*/gm, "\n#### 注 $1\n\n")
    .replace(/\[\^([^\]]+)\]/g, "<sup>注 $1</sup>");
}

function renderMarkdown(markdown) {
  const mathSegments = [];
  const stashMath = (value) => {
    const token = `MATHSEGMENT${mathSegments.length}PLACEHOLDER`;
    mathSegments.push({ token, value });
    return token;
  };

  const protectedMarkdown = normalizeFootnotes(markdown)
    .replace(/\$\$[\s\S]*?\$\$/g, stashMath)
    .replace(/\$(?!\$)(?:\\.|[^$\n])+\$/g, stashMath);

  let html = marked.parse(protectedMarkdown);
  for (const { token, value } of mathSegments) {
    html = html.replaceAll(token, () => escapeHtml(value));
  }
  return html;
}

function getImageDimensions(filePath) {
  const buffer = fs.readFileSync(filePath);

  if (
    buffer.length >= 24 &&
    buffer.subarray(0, 8).equals(
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    )
  ) {
    return {
      width: buffer.readUInt32BE(16),
      height: buffer.readUInt32BE(20),
    };
  }

  if (buffer.length >= 4 && buffer[0] === 0xff && buffer[1] === 0xd8) {
    let offset = 2;
    while (offset + 8 < buffer.length) {
      if (buffer[offset] !== 0xff) {
        offset += 1;
        continue;
      }
      const marker = buffer[offset + 1];
      if (marker === 0xd8 || marker === 0xd9) {
        offset += 2;
        continue;
      }
      const segmentLength = buffer.readUInt16BE(offset + 2);
      const isStartOfFrame =
        (marker >= 0xc0 && marker <= 0xc3) ||
        (marker >= 0xc5 && marker <= 0xc7) ||
        (marker >= 0xc9 && marker <= 0xcb) ||
        (marker >= 0xcd && marker <= 0xcf);
      if (isStartOfFrame) {
        return {
          width: buffer.readUInt16BE(offset + 7),
          height: buffer.readUInt16BE(offset + 5),
        };
      }
      if (segmentLength < 2) break;
      offset += segmentLength + 2;
    }
  }

  return null;
}

function enrichImages(html, pageDir) {
  return html.replace(/<img\b[^>]*>/g, (tag) => {
    const srcMatch = tag.match(/\bsrc="([^"]+)"/);
    if (!srcMatch || /^(?:https?:|data:|\/\/)/.test(srcMatch[1])) {
      return tag;
    }

    const decodedSrc = decodeURIComponent(srcMatch[1]);
    const imagePath = path.resolve(pageDir, decodedSrc);
    if (!fs.existsSync(imagePath)) return tag;

    const dimensions = getImageDimensions(imagePath);
    const attributes = [
      'loading="lazy"',
      'decoding="async"',
      dimensions ? `width="${dimensions.width}"` : "",
      dimensions ? `height="${dimensions.height}"` : "",
    ]
      .filter(Boolean)
      .join(" ");

    return tag.replace("<img", `<img ${attributes}`);
  });
}

function canonicalUrl(filePath) {
  const relative = path.relative(rootDir, filePath).split(path.sep).join("/");
  return encodeURI(`https://sylphyz.github.io/${relative}`);
}

function buildMetadata(title, description, url) {
  const safeTitle = escapeHtml(`${title} · SylphyZ`);
  const safeDescription = escapeHtml(description);
  return [
    "  <!-- NOTE_META_START -->",
    `  <meta name="description" content="${safeDescription}">`,
    '  <meta name="author" content="SylphyZ">',
    '  <meta name="theme-color" content="#f5f0e8">',
    `  <link rel="canonical" href="${url}">`,
    '  <link rel="icon" href="/favicon.png" type="image/png">',
    '  <meta property="og:type" content="article">',
    `  <meta property="og:title" content="${safeTitle}">`,
    `  <meta property="og:description" content="${safeDescription}">`,
    `  <meta property="og:url" content="${url}">`,
    '  <meta property="og:image" content="https://sylphyz.github.io/og.png">',
    '  <meta property="og:image:width" content="1200">',
    '  <meta property="og:image:height" content="630">',
    '  <meta property="og:image:alt" content="SYLPHYZ · NOTES · VISION · MODELS">',
    '  <meta name="twitter:card" content="summary_large_image">',
    "  <!-- NOTE_META_END -->",
  ].join("\n");
}

const excludedNoteBasenames = new Set(["第6章_因子研究现状_学习笔记"]);
const isExcludedNote = (filePath) =>
  excludedNoteBasenames.has(path.basename(filePath, path.extname(filePath)));

const markdownFiles = walk(notebookDir).filter(
  (filePath) => filePath.endsWith(".md") && !isExcludedNote(filePath),
);
const templatePath = path.join(
  notebookDir,
  "Bodie《Investments》",
  "Chapter5_风险收益与历史记录_总结.html",
);

for (const markdownPath of markdownFiles) {
  const htmlPath = markdownPath.replace(/\.md$/i, ".html");
  if (fs.existsSync(htmlPath)) continue;

  if (!fs.existsSync(templatePath)) {
    throw new Error(`Note template not found: ${templatePath}`);
  }

  const template = fs
    .readFileSync(templatePath, "utf8")
    .replace(
      /^[ \t]*<article id="content">[\s\S]*?<\/article>/m,
      '  <article id="content"></article>',
    );
  fs.writeFileSync(htmlPath, template, "utf8");
}

const articleFiles = walk(notebookDir).filter(
  (filePath) =>
    filePath.endsWith(".html") &&
    path.basename(filePath) !== "index.html" &&
    !isExcludedNote(filePath),
);

for (const htmlPath of articleFiles) {
  let html = fs.readFileSync(htmlPath, "utf8");
  const markdownPath = htmlPath.replace(/\.html$/i, ".md");
  const embedded = html.match(
    /<script type="text\/markdown" id="md-source">\s*([\s\S]*?)\s*<\/script>/,
  );

  if (!fs.existsSync(markdownPath)) {
    if (!embedded) {
      throw new Error(`No Markdown source found for ${htmlPath}`);
    }
    fs.writeFileSync(markdownPath, `${embedded[1].trim()}\n`, "utf8");
  }

  const markdown = fs.readFileSync(markdownPath, "utf8");
  const heading = markdown.match(/^#\s+(.+)$/m)?.[1]?.trim() ?? "学习笔记";
  const descriptionSource =
    markdown.match(/^>\s*(.+)$/m)?.[1] ??
    markdown.match(/^(?!#|\s*$)(.+)$/m)?.[1] ??
    "SylphyZ 的学习笔记";
  const description = stripMarkdown(descriptionSource).slice(0, 150);
  const rendered = enrichImages(
    renderMarkdown(markdown),
    path.dirname(htmlPath),
  );
  const noteBlock = [
    '  <article id="content">',
    "    <!-- NOTE_CONTENT_START -->",
    rendered
      .trim()
      .split("\n")
      .map((line) => `    ${line}`)
      .join("\n"),
    "    <!-- NOTE_CONTENT_END -->",
    "  </article>",
  ].join("\n");

  html = html
    .replace(
      /\s*<!-- NOTE_META_START -->[\s\S]*?<!-- NOTE_META_END -->\s*/g,
      "\n",
    )
    .replace(
      /\s*<meta name="description"[\s\S]*?<meta name="twitter:card" content="summary_large_image">\s*/g,
      "\n",
    )
    .replace(
      /\s*<script src="https:\/\/cdn\.jsdelivr\.net\/npm\/marked\/marked\.min\.js"><\/script>\s*/g,
      "\n",
    )
    .replace(
      /<script src="https:\/\/cdn\.jsdelivr\.net\/npm\/mathjax@3\/es5\/tex-svg\.js"><\/script>/g,
      '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script>',
    )
    .replace(
      /\s*<link href="https:\/\/fonts\.googleapis\.com\/css2[^"]+" rel="stylesheet">\s*/g,
      "\n",
    )
    .replace(
      /<title>[\s\S]*?<\/title>/,
      () => `<title>${escapeHtml(`${heading} · SylphyZ`)}</title>`,
    )
    .replace(
      /(<title>[\s\S]*?<\/title>)/,
      (_match, titleTag) =>
        `${titleTag}\n${buildMetadata(heading, description, canonicalUrl(htmlPath))}`,
    )
    .replace(
      /^[ \t]*<article id="content">[\s\S]*?<\/article>/m,
      () => noteBlock,
    )
    .replace(
      /\s*<script type="text\/markdown" id="md-source">[\s\S]*?<\/script>\s*/g,
      "\n",
    )
    .replace(
      /\s*<script>\s*\(function\(\) \{[\s\S]*?marked\.setOptions[\s\S]*?<\/script>\s*/g,
      "\n",
    )
    .replaceAll("--accent: #8b6f47;", "--accent: #765a35;")
    .replaceAll("--accent-light: #c4a882;", "--accent-light: #786241;")
    .replaceAll("--muted: #9a9080;", "--muted: #6f675b;")
    .replaceAll(
      'font-family: "Noto Serif SC", serif;',
      'font-family: ui-serif, "Noto Serif SC", "Songti SC", STSong, serif;',
    )
    .replaceAll(
      'font-family: "Cormorant Garamond", serif;',
      'font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;',
    );

  if (!html.includes("ACCESSIBILITY_STYLES")) {
    html = html.replace(
      "</style>",
      `    /* ACCESSIBILITY_STYLES */\n    a:focus-visible, button:focus-visible, input:focus-visible {\n      outline: 2px solid var(--accent);\n      outline-offset: 4px;\n    }\n    @media (prefers-reduced-motion: reduce) {\n      *, *::before, *::after {\n        scroll-behavior: auto !important;\n        transition-duration: 0.01ms !important;\n        animation-duration: 0.01ms !important;\n        animation-iteration-count: 1 !important;\n      }\n    }\n  </style>`,
    );
  }

  html = html
    .replace(
      /(?:<nav aria-label="面包屑">)*<a class="back-link" href="\.\.\/index\.html">&larr; 返回笔记目录<\/a>(?:<\/nav>)*/g,
      () =>
        '<nav aria-label="面包屑"><a class="back-link" href="../index.html">&larr; 返回笔记目录</a></nav>',
    )
    .replace(
      '<div class="wrap">\n  <nav aria-label="面包屑">',
      '<main class="wrap">\n  <nav aria-label="面包屑">',
    )
    .replace("\n</div>\n</body>", "\n</main>\n</body>")
    .replaceAll("\t", "  ")
    .replace(/[ ]+$/gm, "")
    .replace(/\n{3,}/g, "\n\n");

  fs.writeFileSync(htmlPath, html, "utf8");
}

const sitemapPages = walk(rootDir)
  .filter((filePath) => filePath.endsWith(".html"))
  .filter((filePath) => path.basename(filePath) !== "404.html")
  .map((filePath) => {
    const relative = path.relative(rootDir, filePath).split(path.sep).join("/");
    if (relative === "index.html") return "https://sylphyz.github.io/";
    if (relative.endsWith("/index.html")) {
      return encodeURI(
        `https://sylphyz.github.io/${relative.slice(0, -"index.html".length)}`,
      );
    }
    return encodeURI(`https://sylphyz.github.io/${relative}`);
  })
  .sort();

const sitemap = [
  '<?xml version="1.0" encoding="UTF-8"?>',
  '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
  ...sitemapPages.map((url) => `  <url><loc>${url}</loc></url>`),
  "</urlset>",
  "",
].join("\n");
fs.writeFileSync(path.join(rootDir, "sitemap.xml"), sitemap, "utf8");

console.log(
  `Rendered ${articleFiles.length} note pages and ${sitemapPages.length} sitemap entries.`,
);
