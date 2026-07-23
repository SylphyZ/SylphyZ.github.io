import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(scriptDir, "..");
const ignoredDirs = new Set([".git", "node_modules"]);

function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    if (entry.isDirectory() && ignoredDirs.has(entry.name)) return [];
    const fullPath = path.join(dir, entry.name);
    return entry.isDirectory() ? walk(fullPath) : [fullPath];
  });
}

const htmlFiles = walk(rootDir).filter((filePath) => filePath.endsWith(".html"));
const errors = [];

for (const filePath of htmlFiles) {
  const html = fs.readFileSync(filePath, "utf8");
  const relativePage = path.relative(rootDir, filePath);

  if (!/<meta name="viewport"/.test(html)) {
    errors.push(`${relativePage}: missing viewport metadata`);
  }
  if (!/<meta name="description"/.test(html)) {
    errors.push(`${relativePage}: missing description metadata`);
  }
  if (!/<h1[\s>]/.test(html)) {
    errors.push(`${relativePage}: missing h1`);
  }
  if ((html.match(/<h1[\s>]/g) ?? []).length !== 1) {
    errors.push(`${relativePage}: expected exactly one h1`);
  }
  if ((html.match(/<article[\s>]/g) ?? []).length > 1) {
    errors.push(`${relativePage}: repeated article element`);
  }
  if (/MARKDOWN_CONTENT|marked\.min\.js|fonts\.googleapis\.com/.test(html)) {
    errors.push(`${relativePage}: contains a stale placeholder or runtime dependency`);
  }

  for (const image of html.matchAll(/<img\b[^>]*>/g)) {
    if (!/\balt="[^"]+"/.test(image[0])) {
      errors.push(`${relativePage}: image is missing useful alt text`);
    }
    if (
      relativePage.startsWith(`notebook${path.sep}`) &&
      (!/\bloading="lazy"/.test(image[0]) ||
        !/\bwidth="\d+"/.test(image[0]) ||
        !/\bheight="\d+"/.test(image[0]))
    ) {
      errors.push(`${relativePage}: note image is missing lazy loading or dimensions`);
    }
  }

  for (const match of html.matchAll(
    /(?:href|src)\s*=\s*["']([^"'#?]+)["']/g,
  )) {
    const reference = match[1];
    if (/^(?:https?:|mailto:|tel:|data:|javascript:|\/\/)/.test(reference)) {
      continue;
    }

    const target = reference.startsWith("/")
      ? path.join(rootDir, decodeURIComponent(reference.slice(1)))
      : path.resolve(path.dirname(filePath), decodeURIComponent(reference));
    if (!fs.existsSync(target)) {
      errors.push(`${relativePage}: missing local reference ${reference}`);
    }
  }
}

for (const filePath of walk(rootDir).filter((file) =>
  [".html", ".md"].includes(path.extname(file)),
)) {
  const text = fs.readFileSync(filePath, "utf8");
  if (text.includes("MARKDOWN_CONTENT")) {
    errors.push(`${path.relative(rootDir, filePath)}: unresolved Markdown placeholder`);
  }
}

if (errors.length) {
  console.error(errors.join("\n"));
  process.exit(1);
}

console.log(`Checked ${htmlFiles.length} HTML pages; all local references resolve.`);
