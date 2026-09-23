// Put the live demo into the site, running the library built from this commit.
//
// examples/ imports toolnexus-web from the CDN so each page stays a standalone
// file you can copy. The deployed copy must not: the site deploys on every push
// to main, a release reaches npm later (or waits for approval when staged), and
// until then a CDN import of the new version is a 404 and the demo is dead.
// So the site ships dist/ beside the demo and points both pages at it.
import { copyFileSync, mkdirSync, readdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';

const examples = new URL('../../examples/', import.meta.url);
const dist = new URL('../../dist/', import.meta.url);
const out = new URL('../public/demo/', import.meta.url);
const lib = new URL('lib/', out);

if (!existsSync(new URL('index.js', dist))) {
  throw new Error('dist/index.js is missing — run `npm run build` at the repo root before building the site');
}
mkdirSync(lib, { recursive: true });
for (const f of readdirSync(dist).filter((f) => f.endsWith('.js'))) copyFileSync(new URL(f, dist), new URL(f, lib));

const CDN = /https:\/\/cdn\.jsdelivr\.net\/npm\/toolnexus-web@[^/'"]+\/dist\/index\.js/g;
for (const page of ['index.html', 'jev-snake.html']) {
  const html = readFileSync(new URL(page, examples), 'utf8');
  const local = html.replace(CDN, './lib/index.js');
  if (local === html) throw new Error(`${page}: no toolnexus-web CDN import found to repoint`);
  writeFileSync(new URL(page, out), local);
}
copyFileSync(new URL('coi.js', examples), new URL('coi.js', out));
console.log('copied examples/{index.html,jev-snake.html,coi.js} + dist/*.js -> site/public/demo/');
