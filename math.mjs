// scripter's math, in Node so it shares code with mark-vector:
//
// - spoken Korean math -> AsciiMath2 with asciimath-markdown's spoken-math
//   converter, the very one behind mark-vector's math dictation
//     {"op": "commands", "vocabulary": csv, "text": t}  -> {"spans": [[start, end], ...]}
//     {"op": "convert", "vocabulary": csv, "items": [...]} -> {"items": [...]}
// - AsciiMath2 -> PNG: the LaTeX asciimath-markdown renders in mark-vector
//   (asciiMathBlockToTex - same symbols, row layout and bracket sizing),
//   then MathJax SVG, then resvg. No browser involved.
//     {"op": "render", "items": [...], "fontSize": 80, "color": "#f9fafb"}
//       -> {"images": ["<base64 PNG>", ...]}
// One JSON request on stdin, one response on stdout.
import { createRequire } from "node:module";
import { Resvg } from "@resvg/resvg-js";
import {
  parseSpokenMathCsv,
  setSpokenMathVocabulary,
  spokenMathToAsciiMath,
  findMathBlockCommands,
} from "asciimath-markdown/spoken-math";
import { asciiMathBlockToTex } from "asciimath-markdown";

const require = createRequire(import.meta.url);
const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { SVG } = require("mathjax-full/js/output/svg.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { AllPackages } = require("mathjax-full/js/input/tex/AllPackages.js");

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const doc = mathjax.document("", {
  InputJax: new TeX({ packages: AllPackages }),
  OutputJax: new SVG({ fontCache: "none" }),
});

function render(source, fontSize, color) {
  let tex;
  try {
    tex = asciiMathBlockToTex(source);
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
const request = JSON.parse(input);
if (request.vocabulary !== undefined) setSpokenMathVocabulary(parseSpokenMathCsv(request.vocabulary));
let response;
if (request.op === "commands") {
  response = { spans: findMathBlockCommands(request.text) };
} else if (request.op === "convert") {
  response = { items: request.items.map(spokenMathToAsciiMath) };
} else {
  const { items, fontSize = 80, color = "#000000" } = request;
  response = { images: items.map((item) => render(item, fontSize, color)) };
}
process.stdout.write(JSON.stringify(response));
