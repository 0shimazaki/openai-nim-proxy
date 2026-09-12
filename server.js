// server.js - OpenAI to NVIDIA NIM Proxy (Optimized for Chub AI & SillyTavern)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json({ limit: '50mb' }));

// ---- CORS ----
// If you set ALLOWED_ORIGINS (comma-separated) in your env vars, only those
// origins can call this proxy from a browser. If you leave it unset, all
// origins are allowed (same as before) so nothing breaks by default.
const allowedOrigins = (process.env.ALLOWED_ORIGINS || '')
  .split(',')
  .map(o => o.trim())
  .filter(Boolean);

app.use(cors(
  allowedOrigins.length > 0
    ? { origin: allowedOrigins }
    : {}
));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';

// Support multiple NIM API keys, round-robin + automatic fallback on failure
const NIM_API_KEYS = [process.env.NIM_API_KEY, process.env.NIM_API_KEY2, process.env.NIM_API_KEY3].filter(Boolean);

if (NIM_API_KEYS.length === 0) {
  console.error('FATAL: No NIM API keys configured! Set NIM_API_KEY and/or NIM_API_KEY2/3.');
}

let keyIndex = 0;
function getNextKey() {
  const key = NIM_API_KEYS[keyIndex % NIM_API_KEYS.length];
  keyIndex = (keyIndex + 1) % NIM_API_KEYS.length;
  return key;
}

// Display thinking tags in Chub AI chat UI
// Can now be flipped via env var without redeploying code: SHOW_REASONING=true
const SHOW_REASONING = process.env.SHOW_REASONING === 'true';

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-4': 'deepseek-ai/deepseek-v4-flash-0731',
  'gpt-4-turbo': 'z-ai/glm-5.2',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro-0813',
  'claude-3-opus': 'google/gemma-4-31b-it',
  'claude-3-sonnet': 'minimaxai/minimax-m3',
  'gemini-pro': 'moonshotai/kimi-k3'
};

// Explicit list of models that need the special reasoning/thinking handling.
// Safer than checking `.includes('kimi-k3')` on the model string, since that
// substring check could accidentally match future/unrelated model names.
const REASONING_MODELS = new Set(['moonshotai/kimi-k3']);

// Axios instance (per-key retry logic removed in favor of cross-key fallback below)
const nimClient = axios.create({
  baseURL: NIM_API_BASE,
  timeout: 300000 // 5-minute timeout for deep reasoning
});

// ---- Shared secret required from clients (Chub AI / SillyTavern "API key" box) ----
// No more hardcoded fallback secret. If PROXY_SECRET isn't set in your Vercel
// env vars, every request is rejected with a clear error instead of silently
// accepting a publicly-known default password.
const PROXY_SECRET = process.env.PROXY_SECRET;

function checkAuth(req, res) {
  if (!PROXY_SECRET) {
    res.status(500).json({
      error: {
        message: 'Server misconfigured: PROXY_SECRET is not set.',
        type: 'server_error'
      }
    });
    return false;
  }

  const authHeader = req.headers['authorization'] || '';
  const token = authHeader.replace(/^Bearer\s+/i, '');

  if (token !== PROXY_SECRET) {
    res.status(401).json({ error: { message: 'Invalid API key.', type: 'invalid_request_error' } });
    return false;
  }
  return true;
}

// Tries each configured key in turn; rotates on 429/401/403, bails immediately on other errors
async function postWithKeyFallback(payload, config) {
  if (NIM_API_KEYS.length === 0) {
    const err = new Error('No NIM API keys configured on the server.');
    err.response = { status: 500, data: { error: err.message } };
    throw err;
  }

  let lastError;
  for (let i = 0; i < NIM_API_KEYS.length; i++) {
    const key = getNextKey();
    try {
      return await nimClient.post('/chat/completions', payload, {
        ...config,
        headers: { ...config.headers, 'Authorization': `Bearer ${key}` }
      });
    } catch (err) {
      lastError = err;
      const status = err.response?.status;
      if (status !== 429 && status !== 401 && status !== 403) throw err;
      console.log(`[Key Fallback] Key failed with status ${status}, trying next key...`);
    }
  }
  throw lastError;
}

// Some models (DeepSeek V4 Flash/Pro) don't send reasoning in a separate
// `reasoning_content` field like Kimi K3 does — they embed <think>...</think>
// tags directly inside the normal `content` text. The reasoning_content-based
// suppression above never touches that, so this strips it too whenever
// SHOW_REASONING is off, regardless of which model produced it.

// For non-streamed responses we have the full string at once, so a simple
// regex removal is enough.
function stripThinkTagsFromText(text) {
  if (!text) return text;
  return text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
}

// For streamed responses, the <think>/</think> markers can be split across
// two or more separate network chunks (e.g. one chunk ends in "<thi" and the
// next starts with "nk>"). This factory returns a stateful function that
// keeps a tiny buffer of "could still become a tag" text between calls, so a
// split tag is still caught and removed correctly.
function makeThinkStripper() {
  let insideThink = false;
  let carry = '';
  const OPEN = '<think>';
  const CLOSE = '</think>';

  return function strip(chunkText) {
    let text = carry + (chunkText || '');
    carry = '';
    let output = '';

    while (text.length) {
      if (!insideThink) {
        const idx = text.indexOf('<');
        if (idx === -1) {
          output += text;
          text = '';
        } else {
          output += text.slice(0, idx);
          const remainder = text.slice(idx);
          if (remainder.length < OPEN.length && OPEN.startsWith(remainder)) {
            carry = remainder; // possible partial "<think>" split across chunks
            text = '';
          } else if (remainder.startsWith(OPEN)) {
            insideThink = true;
            text = remainder.slice(OPEN.length);
          } else {
            output += remainder[0];
            text = remainder.slice(1);
          }
        }
      } else {
        const idx = text.indexOf('<');
        if (idx === -1) {
          text = ''; // still inside <think>, discard this piece
        } else {
          const remainder = text.slice(idx);
          if (remainder.length < CLOSE.length && CLOSE.startsWith(remainder)) {
            carry = remainder; // possible partial "</think>" split across chunks
            text = '';
          } else if (remainder.startsWith(CLOSE)) {
            insideThink = false;
            text = remainder.slice(CLOSE.length);
          } else {
            text = remainder.slice(1); // still inside <think>, discard
          }
        }
      }
    }

    return output;
  };
}

// Strips anything that looks like a key/token/secret out of error bodies
// before they're sent back to the client, so a NIM error response can never
// accidentally leak credentials to whoever is calling your proxy.
function sanitizeError(errorDetails) {
  let text = typeof errorDetails === 'object' ? JSON.stringify(errorDetails) : String(errorDetails);
  text = text.replace(/(sk-|nvapi-)[a-zA-Z0-9\-_]+/g, '[REDACTED]');
  return text;
}

app.get('/health', (req, res) => res.json({
  status: NIM_API_KEYS.length > 0 ? 'ok' : 'misconfigured',
  proxy: 'Chub AI to NVIDIA NIM',
  keysConfigured: NIM_API_KEYS.length
}));

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  if (!checkAuth(req, res)) return;
  if (req.socket) req.socket.setTimeout(600000);

  try {
    const { model, messages, stream, reasoning_effort } = req.body;

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: { message: "Missing 'messages' array.", type: "invalid_request_error" } });
    }

    let nimModel = model ? (MODEL_MAPPING[model] || model) : 'moonshotai/kimi-k3';
    const isReasoningModel = REASONING_MODELS.has(nimModel);

    // Reconstruct message history to preserve reasoning content for Kimi K3
    const finalMessages = messages.map(msg => {
      const cleanMsg = { role: msg.role, content: msg.content || '' };

      try {
        if (msg.reasoning_content) {
          cleanMsg.reasoning_content = msg.reasoning_content;
        } else if (msg.role === 'assistant' && typeof msg.content === 'string' && msg.content.includes('<think>')) {
          const match = msg.content.match(/<think>([\s\S]*?)<\/think>/);
          if (match) {
            cleanMsg.reasoning_content = match[1].trim();
            cleanMsg.content = msg.content.replace(/<think>[\s\S]*?<\/think>/, '').trim();
          }
        }
      } catch (parseErr) {
        // If a malformed <think> tag ever shows up, fall back to the raw
        // message instead of letting one bad entry break the whole request.
        console.error('Message reasoning parse error, using raw content:', parseErr.message);
      }
      return cleanMsg;
    });

    // Build payload strict to NIM requirements
    const nimRequest = {
      model: nimModel,
      messages: finalMessages,
      stream: Boolean(stream)
    };

    if (isReasoningModel) {
      // Force token budget high enough to handle thinking + response output
      nimRequest.max_tokens = 16384;
      nimRequest.temperature = 1.0;
      nimRequest.reasoning_effort = reasoning_effort || 'low';
      // Strictly omit frequency_penalty, presence_penalty, repetition_penalty, min_p, top_k
    } else {
      nimRequest.max_tokens = req.body.max_tokens || 4096;
      if (req.body.temperature !== undefined) nimRequest.temperature = req.body.temperature;
      if (req.body.top_p !== undefined) nimRequest.top_p = req.body.top_p;
    }

    const response = await postWithKeyFallback(nimRequest, {
      headers: {
        'Content-Type': 'application/json',
        'Accept': stream ? 'text/event-stream' : 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;
      // One stripper instance per response, so its partial-tag memory
      // persists correctly across all chunks of this single reply.
      const stripInlineThink = makeThinkStripper();

      response.data.on('data', (chunk) => {
        try {
          buffer += chunk.toString();
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          lines.forEach(line => {
            if (line.startsWith('data: ')) {
              if (line.includes('[DONE]')) {
                res.write('data: [DONE]\n\n');
                return;
              }

              try {
                const data = JSON.parse(line.slice(6));
                if (data.choices?.[0]?.delta) {
                  const reasoning = data.choices[0].delta.reasoning_content;
                  const content = data.choices[0].delta.content;

                  if (SHOW_REASONING) {
                    let combinedContent = '';
                    if (reasoning && !reasoningStarted) {
                      combinedContent = '<think>\n' + reasoning;
                      reasoningStarted = true;
                    } else if (reasoning) {
                      combinedContent = reasoning;
                    }

                    if (content && reasoningStarted) {
                      combinedContent += '\n</think>\n\n' + content;
                      reasoningStarted = false;
                    } else if (content) {
                      combinedContent += content;
                    }

                    data.choices[0].delta.content = combinedContent;
                  } else {
                    // Handles both cases: models that never send inline
                    // <think> tags (stripInlineThink passes text through
                    // unchanged) and models like DeepSeek V4 that embed them
                    // directly in `content` (stripInlineThink removes them,
                    // even if a tag is split across this chunk and the next).
                    data.choices[0].delta.content = stripInlineThink(content || '');
                  }
                  delete data.choices[0].delta.reasoning_content;
                }
                res.write(`data: ${JSON.stringify(data)}\n\n`);
              } catch (e) {
                res.write(line + '\n\n');
              }
            }
          });
        } catch (streamError) {
          console.error('Stream processing error:', streamError.message);
        }
      });

      response.data.on('end', () => res.end());
      response.data.on('error', () => { if (!res.headersSent) res.end(); });
    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model || 'unspecified-model',
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = `<think>\n${choice.message.reasoning_content}\n</think>\n\n${fullContent}`;
          } else if (!SHOW_REASONING) {
            // Some models (DeepSeek V4 Flash/Pro) embed <think>...</think>
            // directly in the content string instead of a separate
            // reasoning_content field. Strip it here too.
            fullContent = stripThinkTagsFromText(fullContent);
          }
          return {
            index: choice.index,
            message: { role: choice.message.role, content: fullContent },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      res.json(openaiResponse);
    }

  } catch (error) {
    const status = error.response?.status || 500;
    const errorDetails = error.response?.data || error.message;
    console.error(`NIM Intercept Error [${status}]:`, JSON.stringify(errorDetails));

    if (!res.headersSent) {
      res.status(status).json({
        error: {
          message: sanitizeError(errorDetails),
          type: 'invalid_request_error',
          code: status
        }
      });
    }
  }
});

app.all('*', (req, res) => res.status(404).json({ error: { message: 'Endpoint not found', code: 404 } }));

if (!process.env.VERCEL) {
  const server = app.listen(PORT, () => console.log(`Proxy listening on port ${PORT}`));
  server.timeout = 600000;
}

module.exports = app;
