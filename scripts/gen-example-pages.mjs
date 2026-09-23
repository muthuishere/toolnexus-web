// Render the demo's examples into docs pages, one per example plus an overview.
//
// The tools code on those pages is the demo's own PACKS, read out of
// examples/index.html — so the code a reader copies from the docs is the code
// the live demo runs, and the two cannot drift. test/examples-docs.test.mjs
// fails when the committed pages are stale. Re-run: npm run docs:examples
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';

const root = new URL('../', import.meta.url);
const html = readFileSync(new URL('examples/index.html', root), 'utf8');

/** The demo's PACKS array, evaluated from the page source. */
export function readPacks(source = html) {
  const start = source.indexOf('const PACKS = [');
  const end = source.indexOf('\n];', start) + 3;
  if (start < 0 || end < 3) throw new Error('PACKS not found in examples/index.html');
  return new Function(source.slice(start, end) + '\nreturn PACKS;')();
}

// Prose per example: what it shows, what to expect (only what was measured),
// and how to take it into a real app. Keyed by pack id; every pack needs one.
const DOCS = {
  basics: {
    lead: 'Three tools whose answers a model cannot guess — so a correct answer proves a real call happened.',
    why: `Weather, the time in a city, and exact multiplication. The values are deliberately fixed and
checkable: *31C, humid, light haze* in Chennai, *1096637* for 4831 × 227. A model that answers
correctly had to call the tool; a model that answers anything else guessed. That makes this the
example to run first on any new model.`,
    expect: `Measured by the repo's harness (\`npm run test:models\`, real weights, CPU): **Qwen3-0.6B and
Qwen2.5-0.5B both call all three tools and answer from the result at q4** (3/3 each). Below
~0.5B it stops working — SmolLM2-360M scored 0/3 at every quantization. See
[Which models actually work](/toolnexus-web/verified-models/).`,
    further: `Replace the fixed tables with \`fetch()\` calls to your own API. The handler can be async
and return any JSON; the model reads it back and answers from it.`,
  },
  js: {
    lead: 'A code interpreter in the tab: the model writes JavaScript, a Web Worker runs it.',
    why: `Small models are bad at arithmetic, counting and date maths — and good at writing a line of code
that does it. \`run_js\` hands them that escape hatch. The code runs in a **Web Worker built from a
Blob**: it cannot touch the page, its \`console.log\` output is captured, and it is **terminated
after 3 seconds**, so an infinite loop costs nothing.`,
    expect: `Not measured by the harness yet. One real run on Qwen2.5-0.5B (WebGPU, q4): the model called
\`run_js\` and the worker ran its code — but the code only *defined* a Fibonacci function and never
called it, so the value came back empty. The mechanism works; a 0.5B model's code is the weak link.
Phrase questions so the code is short ("Use JavaScript to…"), and use a bigger model for real work.`,
    further: `The worker is the security boundary — keep it. For heavier sandboxing, run the code in a
sandboxed \`<iframe>\` instead, or in a worker with a stricter timeout.`,
  },
  tab: {
    lead: 'Tools no server-side agent can have: they read and change the page you are looking at.',
    why: `\`page_info\` reads the tab (language, timezone, screen, cores), \`set_accent_color\` restyles the
page you are on, \`count_elements\` queries the DOM. This is the category of tool that only exists
because the model runs where the user is — a server can call your API, but it cannot see the
DOM or change it.`,
    expect: `Not measured by the harness yet. The change is visible, so you can tell at a glance whether the
call happened.`,
    further: `Anything a page can do is a tool: fill a form, scroll to a section, read the selected text,
toggle a setting. Validate inputs in the handler — \`set_accent_color\` checks
\`CSS.supports('color', …)\` before touching the page.`,
  },
  crypto: {
    lead: 'SHA-256, base64 and UUIDs from Web Crypto — answers no model can guess, only compute.',
    why: `A hash is the purest test of whether a tool was really called: there is no way to produce
\`b94d27b9…\` for "hello world" from memory. \`sha256\` uses \`crypto.subtle.digest\`, \`base64\`
encodes and decodes UTF-8 safely, \`uuid\` uses \`crypto.randomUUID()\`.`,
    expect: `Not measured by the harness yet. Compare the hash in the answer with the one in the tool log —
a small model sometimes truncates or retypes long hex strings when it reports them.`,
    further: `Web Crypto also signs, verifies, encrypts and derives keys — all in the tab, all without a
server. Web Crypto needs a secure context (HTTPS or localhost).`,
  },
  notes: {
    lead: 'Memory that survives a reload: three tools over localStorage.',
    why: `\`save_note\`, \`list_notes\` and \`read_note\` give the model a small persistent store. Save a note,
reload the page, and ask for it back — the model forgot the conversation, the tools did not.`,
    expect: `Not measured by the harness yet. Saving works best with the title and text spelled out in the
question ("Save a note titled groceries: milk, eggs, rice").`,
    further: `Swap localStorage for IndexedDB for larger data, or for your own API to sync across devices.
Notes stay in this browser only; "Delete everything" in the demo removes them.`,
  },
  dates: {
    lead: 'Day counts, weekdays and unit conversions — the arithmetic small models get subtly wrong.',
    why: `\`days_between\`, \`weekday\` and \`convert_units\` do in code what a small model does from memory
and gets off by one: leap years, the weekday of a date in 1969, 26.2 miles in kilometres.
\`convert_units\` accepts names as well as symbols ("miles", "fahrenheit") because models use both.`,
    expect: `Not measured by the harness yet. The answers are exact and easy to check against the tool log.`,
    further: `Add time zones with \`Intl.DateTimeFormat\`, or currency with a rates API. Keep one tool per
kind of question — a small model picks well from three names, not from thirty.`,
  },
  web: {
    lead: "Live data from GitHub, Hacker News and Wikipedia — the model is local, only each tool's request leaves the tab.",
    why: `\`github_repo\`, \`hacker_news_top\` and \`wikipedia_summary\` call public APIs that allow
cross-origin requests. The split is the point: **the model and your question never leave the
page**; the only network traffic is the one request each tool makes, to the host it names.`,
    expect: `Not measured by the harness yet. Answers change with the live data, so check them against the
tool log rather than a fixed value.`,
    further: `Any API that sends CORS headers works the same way. For one that does not, put a small proxy
in front of it — the tool is still just a \`fetch()\`.`,
  },
  plain: {
    lead: 'No tools: an ordinary chat with the model running in the tab.',
    why: `With no tools registered, \`chat()\` injects no system prompt and runs a single round — a plain
local chat. Add one \`tool(...)\` line to the file and it becomes a tool-calling chat again.`,
    expect: `A 0.5–0.6B model writes short, fluent text and makes things up freely. Use tools, or
[retrieval](/toolnexus-web/embeddings-rag/), whenever the answer has to be right.`,
    further: `\`chat.on('token', …)\` streams tokens as they arrive; \`chat.reset()\` starts a new
conversation; \`chat.messages\` is the history, if you want to save or restore it.`,
  },
};

const mdx = (t) => String(t).replace(/[{}<>]/g, (c) => `\\${c}`);

function page(p, order) {
  const d = DOCS[p.id];
  if (!d) throw new Error(`no docs prose for example "${p.id}" — add it to DOCS in scripts/gen-example-pages.mjs`);
  const loader = p.id === 'plain'
    ? `const chat = await NexusChat.load({ hub: 'onnx-community/Qwen3-0.6B-ONNX' });`
    : `// loadForTools checks the model can really call a tool before handing it over.
const chat = await NexusChat.loadForTools({ hub: 'onnx-community/Qwen3-0.6B-ONNX' });
await chat.loadTools('./tools.js');   // the file above, saved as tools.js`;
  return `---
title: ${JSON.stringify(p.title)}
description: ${JSON.stringify(d.lead)}
sidebar:
  order: ${order}
---

{/* GENERATED from the PACKS in examples/index.html by scripts/gen-example-pages.mjs — do not edit. */}

${d.lead}

**[Try it live →](/toolnexus-web/demo/#${p.id})** — load a model once in the demo, then ask one of
the questions below.

## What it shows

${d.why}

## The tools file

${mdx(p.note)}

\`\`\`js
${p.code.trimEnd()}
\`\`\`

Each \`tool(name, description, params, handler)\` is one callable. The file is plain JavaScript,
not a module — no imports, no build step, and the demo lets you edit it and apply it live.

## Wire it up

\`\`\`ts
import { NexusChat } from 'toolnexus-web';

${loader}

chat.on('token', (t) => render(t));
${p.id === 'plain' ? '' : "chat.on('toolCall', (call, result) => console.log(call.name, result));\n"}
const answer = await chat.chat(${JSON.stringify(p.questions[0][1])});
\`\`\`

## Questions to try

${p.questions.map(([, q]) => `- ${mdx(q)}`).join('\n')}

## What to expect

${d.expect}

## Going further

${d.further}
`;
}

function overview(packs) {
  return `---
title: Examples
description: Every example in the live demo, with its complete tools file, the questions to try, and what to expect from a small model.
sidebar:
  order: 0
---

{/* GENERATED from the PACKS in examples/index.html by scripts/gen-example-pages.mjs — do not edit. */}

Every example below runs in the [live demo](/toolnexus-web/demo/), which loads a model **once**
and lets you switch between them. Each page has the complete tools file — the same code the demo
runs — plus the questions to try and an honest note on how small models handle it.

| Example | What it shows |
|---|---|
${packs.map((p) => `| [${p.title}](/toolnexus-web/examples/${p.id}/) | ${mdx(DOCS[p.id].lead)} |`).join('\n')}
| [Decisions, Jev-style](/toolnexus-web/decisions/) | Probabilities over your options from one forward pass — and a game of Snake run by a decision model. |
| [Embeddings & RAG](/toolnexus-web/embeddings-rag/) | Answer from your own documents, retrieved in the tab. |

Every tools file follows one shape:

\`\`\`js
tool('name', 'What it does — the model reads this to decide when to call it.',
  { arg: 'string' },                 // shorthand JSON schema
  async ({ arg }) => ({ result }));  // any JSON back; the model answers from it
\`\`\`

Keep each example to a handful of tools: a 0.5B model picks reliably from three names, not from
thirty.
`;
}

export function render(packs = readPacks()) {
  const files = { 'index.mdx': overview(packs) };
  packs.forEach((p, i) => { files[`${p.id}.mdx`] = page(p, i + 1); });
  return files;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const dir = new URL('site/src/content/docs/examples/', root);
  mkdirSync(dir, { recursive: true });
  for (const [name, text] of Object.entries(render())) writeFileSync(new URL(name, dir), text);
  console.log(`wrote ${Object.keys(render()).length} pages to site/src/content/docs/examples/`);
}
