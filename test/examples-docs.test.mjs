// The example docs pages are generated from the demo's PACKS. Committed pages
// that no longer match the demo are a docs bug, so catch it here.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { render, readPacks } from '../scripts/gen-example-pages.mjs';

test('every example has a docs page, and the committed pages are current', () => {
  const files = render();
  assert.equal(Object.keys(files).length, readPacks().length + 1, 'one page per example plus the overview');
  for (const [name, text] of Object.entries(files)) {
    const committed = readFileSync(new URL(`../site/src/content/docs/examples/${name}`, import.meta.url), 'utf8');
    assert.equal(committed, text, `${name} is stale — run: npm run docs:examples`);
  }
});

test('each page carries its example\'s exact tools code and a live link to it', () => {
  const files = render();
  for (const p of readPacks()) {
    assert.ok(files[`${p.id}.mdx`].includes(p.code.trimEnd()), `${p.id}: tools code`);
    assert.ok(files[`${p.id}.mdx`].includes(`/toolnexus-web/demo/#${p.id}`), `${p.id}: live link`);
  }
});
