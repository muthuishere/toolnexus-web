// A mock LLM: real chat templates, real call dialects, no weights and no network.
//
// The point is to exercise the actual loop in chat.ts — template rendering,
// parsing, dispatch, feed-back, phase switching — against the formats real
// models emit, without downloading a model to find out something broke.

/** Format a tool call the way each model family actually emits one. */
export const dialect = {
  qwen: (name, args) => `<tool_call>\n${JSON.stringify({ name, arguments: args })}\n</tool_call>`,
  hermes: (name, args) => `<tool_call>${JSON.stringify({ name, arguments: args })}</tool_call>`,
  mistral: (name, args) => `[TOOL_CALLS] ${JSON.stringify([{ name, arguments: args }])}`,
  llama: (name, args) => JSON.stringify({ name, parameters: args }),
  fenced: (name, args) => '```json\n' + JSON.stringify({ name, arguments: args }) + '\n```',
  // Arguments as a JSON *string* — extremely common from quantized models.
  openai: (name, args) =>
    JSON.stringify({ function: { name, arguments: JSON.stringify(args) } }),
  thinking: (name, args) =>
    `<think>The user wants ${name}. I should call it.</think>\n<tool_call>\n${JSON.stringify({ name, arguments: args })}\n</tool_call>`,
};

/** float32 → IEEE half bits (normal range only — enough for test logits). */
function toHalf(x) {
  const f = new Float32Array([x]), u = new Uint32Array(f.buffer)[0];
  const sign = (u >>> 16) & 0x8000, exp = ((u >>> 23) & 0xff) - 127 + 15, man = (u >>> 13) & 0x3ff;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7c00;
  return sign | (exp << 10) | man;
}

/** Chat templates, close enough to the real ones to assert against. */
const templates = {
  qwen(messages, tools) {
    let out = '';
    for (const m of messages) {
      if (m.role === 'system') {
        out += `<|im_start|>system\n${m.content}`;
        if (tools?.length) {
          out += `\n\n# Tools\n<tools>\n${tools.map((t) => JSON.stringify(t)).join('\n')}\n</tools>`;
        }
        out += '<|im_end|>\n';
      } else if (m.role === 'tool') {
        out += `<|im_start|>user\n<tool_response>\n${m.content}\n</tool_response><|im_end|>\n`;
      } else {
        out += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
      }
    }
    return out + '<|im_start|>assistant\n';
  },
  mistral(messages, tools) {
    let out = tools?.length ? `[AVAILABLE_TOOLS] ${JSON.stringify(tools)}[/AVAILABLE_TOOLS]` : '';
    for (const m of messages) {
      if (m.role === 'user') out += `[INST] ${m.content}[/INST]`;
      else if (m.role === 'tool') out += `[TOOL_RESULTS] ${m.content}[/TOOL_RESULTS]`;
      else out += m.content ?? '';
    }
    return out;
  },
};

/**
 * Build a mock transformers runtime.
 *
 * `script` drives generation: each entry is the raw text the "model" returns
 * for that round — a string, or a function given { messages, tools, round }.
 * The last entry repeats if the loop runs longer.
 */
export function mockLLM({
  family = 'qwen', template = 'qwen', script = [], streamer = true,
  // Forward pass for decide(): next-token logits at the LAST position, keyed by
  // the single character each token spells. Token id = char code, vocab 128.
  logits = {}, logitsType = 'float32',
} = {}) {
  const rendered = [];   // every apply_chat_template call
  const generated = [];  // every generation's options
  const forwards = [];   // every forward pass's decoded prompt
  let round = 0;

  const tokenizer = (text) => {
    const ids = [...String(text)].map((c) => Math.min(c.charCodeAt(0), 127));
    return { input_ids: { dims: [1, ids.length], ids, text }, attention_mask: { dims: [1, ids.length] } };
  };
  tokenizer.encode = (text) => [...String(text)].map((c) => c.charCodeAt(0));
  tokenizer.apply_chat_template = (messages, opts = {}) => {
    rendered.push({ messages: structuredClone(messages), tools: opts.tools, opts });
    return (templates[template] ?? templates.qwen)(messages, opts.tools);
  };

  const generator = async (prompt, opts = {}) => {
    const ctx = { ...rendered[rendered.length - 1], round, prompt, dialect: dialect[family] };
    const entry = script[Math.min(round, script.length - 1)];
    round++;
    let text = typeof entry === 'function' ? await entry(ctx) : (entry ?? '');
    generated.push({ prompt, opts });
    // Drive the streamer the way a real pipeline does, so 'token' hooks fire.
    if (opts.streamer?.callback_function) {
      for (const chunk of String(text).match(/.{1,8}/gs) ?? []) opts.streamer.callback_function(chunk);
    }
    return [{ generated_text: text }];
  };
  generator.tokenizer = tokenizer;
  generator.dispose = async () => { generator.disposed = true; };

  const VOCAB = 128;
  generator.model = async ({ input_ids }) => {
    forwards.push(input_ids.text);
    const seq = input_ids.dims[1];
    const f32 = new Float32Array(seq * VOCAB).fill(-50);
    // Earlier positions carry a decoy, so reading any row but the last is caught.
    for (let p = 0; p < seq - 1; p++) f32[p * VOCAB + 'A'.charCodeAt(0)] = 99;
    const last = (seq - 1) * VOCAB;
    for (const [ch, v] of Object.entries(logits)) f32[last + ch.charCodeAt(0)] = v;
    const data = logitsType === 'float16' ? Uint16Array.from(f32, toHalf) : f32;
    return { logits: { type: logitsType, dims: [1, seq, VOCAB], data } };
  };

  const transformers = {
    env: {},
    pipeline: async (task, modelId, opts) => {
      transformers.lastPipeline = { task, modelId, opts };
      return generator;
    },
    ...(streamer
      ? {
          TextStreamer: class {
            constructor(tok, o) { this.callback_function = o.callback_function; }
          },
        }
      : {}),
  };

  return {
    transformers,
    generator,
    rendered,
    generated,
    forwards,
    get rounds() { return round; },
    /** The system message content as the model saw it in round `i`. */
    systemAt: (i) => rendered[i]?.messages.find((m) => m.role === 'system')?.content,
    /** Tool-result messages the model saw in round `i`. */
    toolMsgsAt: (i) => rendered[i]?.messages.filter((m) => m.role === 'tool') ?? [],
  };
}
