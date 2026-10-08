// Renders scripter's math blocks to PNG: AsciiMath2 -> LaTeX with
// asciimath-parser (configured like asciimath-markdown, the renderer
// mark-vector uses, so a formula looks the same in both), then LaTeX -> SVG
// with MathJax, then SVG -> PNG with resvg. No browser involved.
//
// stdin:  {"items": ["x ^ 2 + 5", ...], "fontSize": 80, "color": "#f9fafb"}
// stdout: {"images": ["<base64 PNG>", ...]} in the same order
import { createRequire } from "node:module";
import { AsciiMath, TokenTypes } from "asciimath-parser";
import { Resvg } from "@resvg/resvg-js";

const require = createRequire(import.meta.url);
const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { SVG } = require("mathjax-full/js/output/svg.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { AllPackages } = require("mathjax-full/js/input/tex/AllPackages.js");

// Same overrides as asciimath-markdown: "<=" / ">=" are plain \le / \ge.
const asciiMath = new AsciiMath({
  display: false,
  symbols: [
    ["<=", { type: TokenTypes.Const, tex: "\\le" }],
    [">=", { type: TokenTypes.Const, tex: "\\ge" }],
  ],
});

// One row per line, like a $$ block in mark-vector: rows go to the parser
// separated by blank lines, and rows with nothing to align on (&) are each
// centered (gathered) rather than right-aligned to a shared edge (aligned).
function toTex(source) {
  const rows = source.split("\n").map((line) => line.trim()).filter(Boolean);
  if (!rows.length) return "";
  let tex = asciiMath.toTex(rows.join("\n\n"), { display: true });
  if (!source.includes("&")) {
    tex = tex.replace(/\\begin\{aligned\}/g, "\\begin{gathered}").replace(/\\end\{aligned\}/g, "\\end{gathered}");
  }
  return tex;
}

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const doc = mathjax.document("", {
  InputJax: new TeX({ packages: AllPackages }),
  OutputJax: new SVG({ fontCache: "none" }),
});

function render(source, fontSize, color) {
  let tex;
  try {
    tex = toTex(source);
  } catch (err) {
    tex = `\\text{${source.replace(/[\\{}]/g, "")}}`;
  }
  if (!tex) return null;
  const node = doc.convert(tex, { display: true, em: fontSize, ex: fontSize / 2 });
  let svg = adaptor.innerHTML(node);
  // MathJax sizes the SVG in ex; resvg needs pixels.
  const ex = fontSize / 2;
  svg = svg
    .replace(/(<svg[^>]*?\s)width="([\d.]+)ex"/, (_, head, w) => `${head}width="${(parseFloat(w) * ex).toFixed(2)}"`)
    .replace(/(<svg[^>]*?\s)height="([\d.]+)ex"/, (_, head, h) => `${head}height="${(parseFloat(h) * ex).toFixed(2)}"`)
    .replace(/currentColor/g, color);
  const png = new Resvg(svg, { fitTo: { mode: "original" }, font: { loadSystemFonts: false } }).render().asPng();
  return Buffer.from(png).toString("base64");
}

let input = "";
for await (const chunk of process.stdin) input += chunk;
const { items, fontSize = 80, color = "#000000" } = JSON.parse(input);
process.stdout.write(JSON.stringify({ images: items.map((item) => render(item, fontSize, color)) }));
