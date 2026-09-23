// NexusDecider: a trained System One encoder (open-jev). One state, any number
// of typed questions, one forward pass.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { NexusDecider, OPEN_JEV } from '../dist/decider.js';
import { mockDecider } from './helpers/mock-decider.mjs';

async function deciderWith(opts = {}, loadOpts = {}) {
  const m = mockDecider(opts);
  const decider = await NexusDecider.load(undefined, { transformers: m.transformers, device: 'wasm', ...loadOpts });
  return { decider, m };
}

const softmax = (xs, t = 1.05) => {
  const mx = Math.max(...xs), e = xs.map((x) => Math.exp((x - mx) / t)), s = e.reduce((a, b) => a + b);
  return e.map((x) => x / s);
};

test('loads the open-jev model by default, tokenizer and model from the same repo', async () => {
  const { decider, m } = await deciderWith();
  assert.equal(decider.modelId, OPEN_JEV);
  assert.deepEqual(m.loaded.map((l) => [l.what, l.repo]), [['tokenizer', OPEN_JEV], ['model', OPEN_JEV]]);
});

test('dtype follows the device unless given: q4f16 on WebGPU, q4 on CPU', async () => {
  const gpu = await deciderWith({}, { device: 'webgpu' });
  const cpu = await deciderWith({}, { device: 'wasm' });
  const pinned = await deciderWith({}, { device: 'webgpu', dtype: 'fp16' });
  assert.equal(gpu.m.loaded[1].opts.dtype, 'q4f16');
  assert.equal(cpu.m.loaded[1].opts.dtype, 'q4');
  assert.equal(pinned.m.loaded[1].opts.dtype, 'fp16');
});

test('the feed follows the published layout exactly', async () => {
  const { decider, m } = await deciderWith();
  await decider.systemOne('charged twice', {
    area: { type: 'choice', instructions: 'which area', options: ['fees', 'refund'] },
    refund: { type: 'noul', instructions: 'wants refund' },
  });

  const [feed] = m.calls;
  assert.deepEqual(m.decode(feed.input_ids.data), [
    '[CLS]', '[STATE]', 'charged', 'twice',
    '[Q]', 'which', 'area', '[OPT]', 'fees', '[OPT]', 'refund',
    '[Q]', 'wants', 'refund', '[OPT]', 'no', '[OPT]', 'yes',
    '[SEP]',
  ]);
  // option tokens → their pair index; question tokens → pairs + question index
  assert.deepEqual(Array.from(feed.seg.data, Number), [-1, -1, -1, -1, -1, 4, 4, -1, 0, -1, 1, -1, 5, 5, -1, 2, -1, 3, -1]);
  assert.deepEqual(Array.from(feed.pair_q.data, Number), [4, 4, 5, 5]);
  assert.deepEqual(Array.from(feed.pair_opt.data, Number), [0, 1, 2, 3]);
  assert.deepEqual(Array.from(feed.attention_mask.data, Number), Array(19).fill(1));
  assert.deepEqual(feed.input_ids.dims, [1, 19]);
  for (const k of ['input_ids', 'attention_mask', 'seg', 'pair_q', 'pair_opt']) assert.equal(feed[k].type, 'int64', k);
});

test('choice: calibrated softmax over the options, in the order given', async () => {
  const scores = { fees: 0.2, refund: 2.1, card: -0.4 };
  const { decider } = await deciderWith({ score: ({ option }) => scores[option] });
  const { answers } = await decider.systemOne('charged twice', {
    area: { type: 'choice', instructions: 'which area', options: ['fees', 'refund', 'card'] },
  });
  const want = softmax([0.2, 2.1, -0.4]);
  assert.equal(answers.area.choice, 'refund');
  assert.deepEqual(Object.keys(answers.area.probabilities), ['fees', 'refund', 'card']);
  Object.values(answers.area.probabilities).forEach((p, i) => assert.ok(Math.abs(p - want[i]) < 1e-6));
  assert.ok(Math.abs(answers.area.probability - want[1]) < 1e-6);
});

test('noul: P(true) is the "yes" slot', async () => {
  const { decider } = await deciderWith({ score: ({ option }) => (option === 'yes' ? 1.5 : -0.5) });
  const { answers } = await decider.systemOne('s', { refund: { type: 'noul', instructions: 'wants refund' } });
  const p = softmax([-0.5, 1.5])[1];
  assert.ok(Math.abs(answers.refund.noul - p) < 1e-6);
  assert.equal(answers.refund.value, true);
});

test('score: expected level over ordered options, plus the likeliest level', async () => {
  const s = { low: 0, mid: 0, high: 0 };
  const { decider } = await deciderWith({ score: ({ option }) => s[option] });
  const { answers } = await decider.systemOne('s', { urgency: { type: 'score', instructions: 'how urgent', options: ['low', 'mid', 'high'] } });
  assert.ok(Math.abs(answers.urgency.score - 1) < 1e-9, 'uniform over 0,1,2 has mean 1');
});

test('each question is normalized on its own, all in one forward pass', async () => {
  const { decider, m } = await deciderWith({ score: ({ question, option }) => (question === 'q one' ? (option === 'b' ? 3 : 0) : option === 'x' ? 3 : 0) });
  const { answers } = await decider.systemOne('s', {
    one: { type: 'choice', instructions: 'q one', options: ['a', 'b'] },
    two: { type: 'choice', instructions: 'q two', options: ['x', 'y', 'z'] },
  });
  assert.equal(m.calls.length, 1);
  assert.equal(answers.one.choice, 'b');
  assert.equal(answers.two.choice, 'x');
  const sum = (o) => Object.values(o).reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum(answers.one.probabilities) - 1) < 1e-9);
  assert.ok(Math.abs(sum(answers.two.probabilities) - 1) < 1e-9);
});

test('a JSON state is serialized, and the state is cut to 256 tokens', async () => {
  const { decider, m } = await deciderWith();
  await decider.systemOne({ order: 42 }, { q: { type: 'noul', instructions: 'ok' } });
  assert.deepEqual(m.decode(m.calls[0].input_ids.data).slice(2, 3), ['{"order":42}']);

  const long = Array.from({ length: 400 }, (_, i) => `w${i}`).join(' ');
  await decider.systemOne(long, { q: { type: 'noul', instructions: 'ok' } });
  const ids = m.decode(m.calls[1].input_ids.data);
  assert.equal(ids.indexOf('[Q]') - 2, 256);
});

test('fp16 logits stored as raw half bits are decoded', async () => {
  const { decider } = await deciderWith({ logitsType: 'float16', score: ({ option }) => (option === 'b' ? 2.5 : 0.5) });
  const { answers } = await decider.systemOne('s', { q: { type: 'choice', instructions: 'pick', options: ['a', 'b'] } });
  const want = softmax([0.5, 2.5]);
  assert.ok(Math.abs(answers.q.probabilities.b - want[1]) < 1e-6);
});

test('a model without the open-jev markers is refused at load', async () => {
  await assert.rejects(deciderWith({ markers: false }), /\[STATE\].*open-jev/);
});

test('bad questions are refused before anything runs', async () => {
  const { decider, m } = await deciderWith();
  await assert.rejects(decider.systemOne('s', {}), /at least one question/);
  await assert.rejects(decider.systemOne('s', { q: { type: 'choice', instructions: 'x', options: ['one'] } }), /2 or more options/);
  await assert.rejects(decider.systemOne('s', { q: { type: 'guess', instructions: 'x' } }), /unknown question type/);
  assert.equal(m.calls.length, 0);
});

test('an input past 512 tokens is refused rather than silently cut', async () => {
  const { decider } = await deciderWith();
  const many = Array.from({ length: 300 }, (_, i) => `o${i}`);
  await assert.rejects(decider.systemOne('s', { q: { type: 'choice', instructions: 'x', options: many } }), /512/);
});

test('a raw runtime exception becomes an error that names the fix', async () => {
  const m = mockDecider();
  m.transformers.env.version = '3.8.1';
  m.transformers.AutoModel.from_pretrained = async () => { throw 20968528; };
  await assert.rejects(
    NexusDecider.load(undefined, { transformers: m.transformers, device: 'webgpu' }),
    /raw exception \(20968528\).*Transformers\.js 4\.x.*3\.8\.1/,
  );
});

test('metrics time each decision', async () => {
  const { decider } = await deciderWith();
  await decider.systemOne('s', { q: { type: 'noul', instructions: 'ok' } });
  assert.equal(decider.metrics.counters.get('decisions'), 1);
});
