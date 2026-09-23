// A stand-in for an open-jev encoder: a word-level tokenizer with the three
// markers, and a model that DECODES its own feed — so the scores it returns
// depend on the option and question text the feed actually carries. A wrong
// seg / pair_q / pair_opt layout then produces wrong answers, not just
// different tensors.

export function mockDecider({ score = () => 0, logitsType = 'float32', markers = true } = {}) {
  const vocab = new Map([['[CLS]', 1], ['[SEP]', 2]]);
  if (markers) { vocab.set('[STATE]', 3); vocab.set('[Q]', 4); vocab.set('[OPT]', 5); }
  const words = new Map();
  const id = (w) => {
    if (vocab.has(w)) return vocab.get(w);
    if (!words.has(w)) words.set(w, 100 + words.size);
    return words.get(w);
  };
  const back = (i) => [...vocab, ...words].find(([, v]) => v === i)?.[0];

  const calls = [];
  const loaded = [];

  class Tensor {
    constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
  }

  const tokenizer = (text) => {
    // A real tokenizer splits an unregistered "[STATE]" into pieces.
    const ids = String(text).split(/\s+/).filter(Boolean)
      .flatMap((w) => (/^\[.+\]$/.test(w) && !vocab.has(w) ? ['[', w.slice(1, -1), ']'] : [w]))
      .map(id);
    return { input_ids: new Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]) };
  };

  const model = async (feed) => {
    calls.push(feed);
    const ids = Array.from(feed.input_ids.data, Number);
    const seg = Array.from(feed.seg.data, Number);
    const pairQ = Array.from(feed.pair_q.data, Number);
    const pairOpt = Array.from(feed.pair_opt.data, Number);
    const text = (slot) => ids.filter((_, i) => seg[i] === slot).map(back).join(' ');
    const f32 = Float32Array.from(pairOpt, (slot, p) => score({ option: text(slot), question: text(pairQ[p]) }));
    const data = logitsType === 'float16' ? Uint16Array.from(f32, toHalf) : f32;
    return { logits: new Tensor(logitsType, data, [1, f32.length]) };
  };
  model.dispose = async () => { model.disposed = true; };

  const transformers = {
    env: {},
    Tensor,
    AutoTokenizer: { from_pretrained: async (repo, opts) => { loaded.push({ what: 'tokenizer', repo, opts }); return tokenizer; } },
    AutoModel: { from_pretrained: async (repo, opts) => { loaded.push({ what: 'model', repo, opts }); return model; } },
    pipeline: async () => { throw new Error('a decider never builds a pipeline'); },
  };

  return { transformers, calls, loaded, model, decode: (ids) => Array.from(ids, (i) => back(Number(i))) };
}

function toHalf(x) {
  const f = new Float32Array([x]), u = new Uint32Array(f.buffer)[0];
  const sign = (u >>> 16) & 0x8000, exp = ((u >>> 23) & 0xff) - 127 + 15, man = (u >>> 13) & 0x3ff;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7c00;
  return sign | (exp << 10) | man;
}
