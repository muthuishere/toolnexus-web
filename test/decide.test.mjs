// decide(): Jev-style direct readout — one forward pass, probabilities over
// the options you allowed, no text generated.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { NexusChat } from '../dist/chat.js';
import { mockLLM } from './helpers/mock-llm.mjs';

async function chatWith(opts = {}) {
  const llm = mockLLM(opts);
  const chat = await NexusChat.load(
    { base: 'https://host/models/', id: 'stub/model' },
    { transformers: llm.transformers, device: 'wasm', dtype: 'q4' },
  );
  return { chat, llm };
}

const EMAIL = 'Payroll asks for your password on a non-company sign-in page.';
const CHOICES = ['Legitimate', 'Spam', 'Phishing'];

// The logits and probabilities published in "Jev in 25 lines of Python"
// (Qwen3-0.6B Q8_0), so the maths is pinned to a result someone else printed.
const ARTICLE = { A: 26.254, B: 27.262, C: 29.614 };

test('reproduces the published Jev readout', async () => {
  const { chat } = await chatWith({ logits: ARTICLE });
  const d = await chat.decide(`Email: ${EMAIL}`, CHOICES);

  assert.equal(d.choice, 'Phishing');
  assert.deepEqual(d.options.map((o) => o.choice), CHOICES, 'options keep the caller order');
  assert.deepEqual(d.options.map((o) => +o.probability.toFixed(3)), [0.031, 0.084, 0.885]);
  assert.deepEqual(d.options.map((o) => +o.logprob.toFixed(3)), [-3.482, -2.474, -0.122]);
  assert.ok(Math.abs(d.options.reduce((s, o) => s + o.probability, 0) - 1) < 1e-9);
  assert.equal(d.probability, d.options[2].probability);
});

test('reads the last position only', async () => {
  // The mock puts a huge logit for "A" on every earlier position.
  const { chat } = await chatWith({ logits: { A: 0, B: 5 } });
  const d = await chat.decide('q', ['first', 'second']);
  assert.equal(d.choice, 'second');
});

test('the prompt lists labelled options, thinking off, and history untouched', async () => {
  const { chat, llm } = await chatWith({ logits: ARTICLE });
  chat.messages = [{ role: 'user', content: 'earlier turn' }];
  await chat.decide(`Email: ${EMAIL}`, CHOICES);

  const r = llm.rendered.at(-1);
  assert.equal(r.opts.add_generation_prompt, true);
  assert.equal(r.opts.enable_thinking, false);
  assert.equal(r.messages[0].role, 'system');
  assert.equal(r.messages[1].content, `Email: ${EMAIL}\n\nA. Legitimate\nB. Spam\nC. Phishing`);
  assert.equal(r.messages.length, 2, 'the conversation is not part of a decision');
  assert.deepEqual(chat.messages, [{ role: 'user', content: 'earlier turn' }]);
  assert.equal(llm.rounds, 0, 'nothing was generated');
  assert.equal(llm.forwards.length, 1, 'exactly one forward pass');
});

test('fp16 logits stored as raw half bits are decoded, not copied', async () => {
  const { chat } = await chatWith({ logits: { A: 1.5, B: 3.25 }, logitsType: 'float16' });
  const d = await chat.decide('q', ['x', 'y']);
  assert.deepEqual(d.options.map((o) => o.logit), [1.5, 3.25]);
  assert.equal(d.choice, 'y');
});

test('a custom system instruction replaces the default', async () => {
  const { chat, llm } = await chatWith({ logits: ARTICLE });
  await chat.decide('q', CHOICES, { system: 'You are a spam filter.' });
  assert.equal(llm.systemAt(llm.rendered.length - 1), 'You are a spam filter.');
});

test('needs between 2 and 26 options', async () => {
  const { chat } = await chatWith({ logits: ARTICLE });
  await assert.rejects(chat.decide('q', ['only']), /2.26 choices/);
  await assert.rejects(chat.decide('q', Array.from({ length: 27 }, (_, i) => `o${i}`)), /2.26 choices/);
});

test('metrics count decisions and time them', async () => {
  const { chat } = await chatWith({ logits: ARTICLE });
  await chat.decide('q', CHOICES);
  await chat.decide('q', CHOICES);
  assert.equal(chat.metrics.counters.get('decisions'), 2);
  assert.ok('decide' in chat.metrics.summary() || Object.keys(chat.metrics.summary()).some((k) => k.startsWith('decide')));
});
