import { Metrics } from './metrics.ts';
import { resolveTransformers, detectDevice, readLogit, type Device, type RuntimeOptions, type TransformersLike } from './runtime.ts';
import { resolveSource, type ModelSource } from './source.ts';

/** A Transformers.js export of an open System One encoder (DeBERTa-v3-large,
 *  trained on typed decisions). Measured by its authors on public gold labels:
 *  0.854 accuracy in-domain, 0.690 on unseen instructions and option sets. */
export const OPEN_JEV = 'onnx-community/open-jev-deberta-v3-large-ONNX';

export type DecisionQuestion =
  /** Pick one option. */
  | { type: 'choice'; instructions: string; options: string[] }
  /** Ordered levels, listed low to high. */
  | { type: 'score'; instructions: string; options: string[] }
  /** A statement; the answer is P(true). */
  | { type: 'noul'; instructions: string };

export interface ChoiceAnswer {
  type: 'choice';
  choice: string;
  probability: number;
  /** Every option, in the order given. */
  probabilities: Record<string, number>;
}

export interface ScoreAnswer {
  type: 'score';
  /** Expected level index, 0 for the first option to K-1 for the last. */
  score: number;
  /** The single likeliest level. */
  level: string;
  probabilities: Record<string, number>;
}

export interface NoulAnswer {
  type: 'noul';
  /** P(true). */
  noul: number;
  value: boolean;
}

export type Answer<Q> = Q extends { type: 'noul' } ? NoulAnswer : Q extends { type: 'score' } ? ScoreAnswer : ChoiceAnswer;

export interface SystemOneResult<Q extends Record<string, DecisionQuestion>> {
  answers: { [K in keyof Q]: Answer<Q[K]> };
  ms: number;
  inputTokens: number;
}

export interface DeciderOptions extends RuntimeOptions {
  device?: Device;
  /** Default: q4f16 on WebGPU (348 MB), q4 on CPU. */
  dtype?: string;
  onProgress?: (p: unknown) => void;
  /** The source model's post-hoc calibration. */
  temperature?: number;
}

const MAX_LEN = 512;
const MAX_STATE_TOKENS = 256;

/** The open-jev export needs a current ONNX Runtime. Transformers.js 3.8.1's
 *  runtime (1.22-dev) rejects the graph by throwing a bare wasm exception
 *  pointer — a number, no message — on both WebGPU and CPU; 4.3.0 loads it. */
export const DECIDER_TRANSFORMERS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0';

/** Typed decisions from a trained System One encoder — Jev's request shape,
 *  in the browser. One state, any number of questions, ONE forward pass that
 *  scores every (question, option) pair; nothing is generated.
 *
 *      const decider = await NexusDecider.load();
 *      const { answers } = await decider.systemOne(ticket, {
 *        team:   { type: 'choice', instructions: 'Which team handles this?', options: ['billing', 'support'] },
 *        urgent: { type: 'noul',   instructions: 'The customer needs help right now.' },
 *      });
 *      answers.team.choice;   // 'billing'
 *      answers.urgent.noul;   // 0.12
 *
 *  Unlike {@link NexusChat.decide}, which reads letter logits off a chat model,
 *  this model was trained for the job and calibrated — and it cannot chat. */
export class NexusDecider {
  readonly metrics = new Metrics();
  temperature: number;

  private constructor(
    private tokenizer: any,
    private model: any,
    private tjs: { Tensor: new (type: string, data: unknown, dims: number[]) => unknown },
    private markers: { CLS: number; SEP: number; STATE: number; Q: number; OPT: number },
    readonly modelId: string,
    readonly device: string,
    readonly dtype: string,
    temperature: number,
  ) {
    this.temperature = temperature;
  }

  static async load(source: ModelSource = { hub: OPEN_JEV }, opts: DeciderOptions = {}): Promise<NexusDecider> {
    const tjs = await resolveTransformers({
      transformers: opts.transformers ?? ((await import(/* @vite-ignore */ DECIDER_TRANSFORMERS)) as TransformersLike),
    });
    if (!tjs.AutoTokenizer || !tjs.AutoModel || !tjs.Tensor) {
      throw new Error('NexusDecider needs AutoTokenizer, AutoModel and Tensor from transformers.js');
    }
    const modelId = await resolveSource(tjs, source);
    const device = await detectDevice(opts.device ?? 'auto');
    const dtype = opts.dtype ?? (device === 'webgpu' ? 'q4f16' : 'q4');
    const t0 = Date.now();
    const tokenizer = await tjs.AutoTokenizer.from_pretrained(modelId, { progress_callback: opts.onProgress });

    const enc = (t: string) => Array.from(tokenizer(t, { add_special_tokens: false }).input_ids.data as ArrayLike<bigint>, Number);
    const one = (m: string) => {
      const ids = enc(m);
      if (ids.length !== 1) throw new Error(`${modelId} has no ${m} token — not an open-jev decision model`);
      return ids[0]!;
    };
    const markers = { CLS: one('[CLS]'), SEP: one('[SEP]'), STATE: one('[STATE]'), Q: one('[Q]'), OPT: one('[OPT]') };

    let model;
    try {
      model = await tjs.AutoModel.from_pretrained(modelId, { dtype, device, progress_callback: opts.onProgress });
    } catch (e) {
      if (e instanceof Error) throw e;
      const version = tjs.env?.version ?? 'unknown';
      throw new Error(
        `ONNX Runtime rejected ${modelId} (${device}/${dtype}) with a raw exception (${String(e)}). ` +
          `This model needs Transformers.js 4.x; the one passed in is ${version}.`,
      );
    }
    const decider = new NexusDecider(tokenizer, model, tjs as never, markers, modelId, device, dtype, opts.temperature ?? 1.05);
    decider.metrics.time('load', Date.now() - t0);
    return decider;
  }

  private enc(text: string): number[] {
    return Array.from(this.tokenizer(text, { add_special_tokens: false }).input_ids.data as ArrayLike<bigint>, Number);
  }

  async systemOne<Q extends Record<string, DecisionQuestion>>(state: string | object, questions: Q): Promise<SystemOneResult<Q>> {
    const entries = Object.entries(questions);
    if (!entries.length) throw new Error('systemOne needs at least one question');
    const options = entries.map(([name, q]) => {
      if (q.type === 'noul') return ['no', 'yes'];
      if (q.type !== 'choice' && q.type !== 'score') throw new Error(`${name}: unknown question type "${(q as { type: string }).type}"`);
      if (!Array.isArray(q.options) || q.options.length < 2) throw new Error(`${name}: a ${q.type} question needs 2 or more options`);
      return q.options;
    });

    // The layout the model was trained on: option tokens carry their pair
    // index, question tokens carry (pairs + question index), the rest -1.
    const { CLS, SEP, STATE, Q: QM, OPT } = this.markers;
    const text = typeof state === 'string' ? state : JSON.stringify(state);
    const tokens = [CLS, STATE, ...this.enc(text).slice(0, MAX_STATE_TOKENS)];
    const seg: number[] = tokens.map(() => -1);
    const pairQ: number[] = [], pairOpt: number[] = [];
    const totalPairs = options.reduce((n, o) => n + o.length, 0);
    const groups = entries.map(([, q], qi) => {
      const ins = this.enc(q.instructions);
      tokens.push(QM, ...ins);
      seg.push(-1, ...ins.map(() => totalPairs + qi));
      return options[qi]!.map((option) => {
        const o = this.enc(option);
        tokens.push(OPT, ...o);
        seg.push(-1, ...o.map(() => pairOpt.length));
        pairQ.push(totalPairs + qi);
        pairOpt.push(pairOpt.length);
        return pairOpt.length - 1;
      });
    });
    tokens.push(SEP);
    seg.push(-1);
    if (tokens.length > MAX_LEN) {
      throw new Error(`decision input is ${tokens.length} tokens; the model takes ${MAX_LEN} — ask fewer or shorter questions`);
    }

    const i64 = (v: number[], dims: number[]) => new this.tjs.Tensor('int64', BigInt64Array.from(v, BigInt), dims);
    const t0 = Date.now();
    const { logits } = await this.model({
      input_ids: i64(tokens, [1, tokens.length]),
      attention_mask: i64(tokens.map(() => 1), [1, tokens.length]),
      seg: i64(seg, [1, seg.length]),
      pair_q: i64(pairQ, [1, pairQ.length]),
      pair_opt: i64(pairOpt, [1, pairOpt.length]),
    });
    const ms = Date.now() - t0;
    this.metrics.time('decide', ms);
    this.metrics.count('decisions');

    const answers: Record<string, ChoiceAnswer | ScoreAnswer | NoulAnswer> = {};
    entries.forEach(([name, q], qi) => {
      const p = this.softmax(groups[qi]!.map((pair) => readLogit(logits, pair)));
      if (q.type === 'noul') {
        answers[name] = { type: 'noul', noul: p[1]!, value: p[1]! >= 0.5 };
        return;
      }
      const probabilities = Object.fromEntries(q.options.map((o, i) => [o, p[i]!]));
      const best = p.indexOf(Math.max(...p));
      answers[name] = q.type === 'score'
        ? { type: 'score', score: p.reduce((s, v, i) => s + v * i, 0), level: q.options[best]!, probabilities }
        : { type: 'choice', choice: q.options[best]!, probability: p[best]!, probabilities };
    });
    return { answers: answers as SystemOneResult<Q>['answers'], ms, inputTokens: tokens.length };
  }

  private softmax(xs: number[]): number[] {
    const max = Math.max(...xs);
    const e = xs.map((x) => Math.exp((x - max) / this.temperature));
    const sum = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / sum);
  }

  dispose(): Promise<void> {
    return this.model.dispose?.() ?? Promise.resolve();
  }
}
