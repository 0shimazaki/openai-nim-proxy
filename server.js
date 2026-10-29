// server.js - OpenAI to NVIDIA NIM Proxy (Optimized for Chub AI & SillyTavern)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json({ limit: '50mb' }));

// ---- CORS ----
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
const SHOW_REASONING = process.env.SHOW_REASONING === 'true';

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'minimaxai/minimax-m3.1-flash',
  'gpt-4': 'deepseek-ai/deepseek-v4.1-flash',
  'gpt-4-turbo': 'z-ai/glm-5.3',
  'gpt-4o': 'stepfun-ai/step-5-preview',
  'claude-3-opus': 'google/gemma-4-31b-it',
  'claude-3-sonnet': 'z-ai/glm-5.3-flash',
  'gemini-pro': 'mistralai/mistral-large-4-1t-a52b'
};

// Explicit list of models that use the standard `reasoning_effort` field.
// Added Mistral Large 4 here so it uses low reasoning effort by default.
const REASONING_MODELS = new Set([
  'moonshotai/kimi-k3',
  'z-ai/glm-5.3',
  'z-ai/glm-5.3-flash',
  'mistralai/mistral-large-4-1t-a52b'
]);

const CHAT_TEMPLATE_REASONING_OFF = {
  'deepseek-ai/deepseek-v4-flash-0731': { thinking: false },
  'deepseek-ai/deepseek-v4.1-flash': { thinking: false },
  'deepseek-ai/deepseek-v4-pro-0813': { thinking: false },
  'google/gemma-4-31b-it': { enable_thinking: false }
};

const nimClient = axios.create({
  baseURL: NIM_API_BASE,
  timeout: 300000 // 5-minute timeout for deep reasoning
});

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

function stripThinkTagsFromText(text) {
  if (!text) return text;
  let result = text.replace(/<think>[\s\S]*?<\/think>/g, '');
  const closeIdx = result.indexOf('</think>');
  if (closeIdx !== -1) {
    result = result.slice(closeIdx + '</think>'.length);
  }
  return result.trim();
}

function makeThinkStripper(leadBufferLimit = 500) {
  let insideThink = false;
  let carry = '';
  let leadResolved = false;
  let leadBuffer = '';
  const OPEN = '<think>';
  const CLOSE = '</think>';

  function processTagged(chunkText) {
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
            carry = remainder;
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
          text = '';
        } else {
          const remainder = text.slice(idx);
          if (remainder.length < CLOSE.length && CLOSE.startsWith(remainder)) {
            carry = remainder;
            text = '';
          } else if (remainder.startsWith(CLOSE)) {
            insideThink = false;
            text = remainder.slice(CLOSE.length);
          } else {
            text = remainder.slice(1);
          }
        }
      }
    }

    return output;
  }

  function flush() {
    let output = '';
    if (!leadResolved) {
      leadResolved = true;
      const buffered = leadBuffer;
      leadBuffer = '';
      output += processTagged(buffered);
    }
    if (carry) {
      output += carry;
      carry = '';
    }
    return output;
  }

  return {
    strip(chunkText) {
      if (leadResolved) {
        return processTagged(chunkText);
      }

      leadBuffer += (chunkText || '');
      const closeIdx = leadBuffer.indexOf(CLOSE);
      const openIdx = leadBuffer.indexOf(OPEN);

      if (closeIdx !== -1 && (openIdx === -1 || closeIdx < openIdx)) {
        const rest = leadBuffer.slice(closeIdx + CLOSE.length);
        leadResolved = true;
        leadBuffer = '';
        return processTagged(rest);
      }

      if (openIdx !== -1 || leadBuffer.length >= leadBufferLimit) {
        const buffered = leadBuffer;
        leadResolved = true;
        leadBuffer = '';
        return processTagged(buffered);
      }

      return '';
    },
    flush
  };
}

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
        console.error('Message reasoning parse error, using raw content:', parseErr.message);
      }
      return cleanMsg;
    });

    const nimRequest = {
      model: nimModel,
      messages: finalMessages,
      stream: Boolean(stream)
    };

    if (isReasoningModel) {
      nimRequest.max_tokens = 16384;
      nimRequest.temperature = 1.0;
      // Forces reasoning effort to 'low' by default unless explicitly overridden,
      // making it much faster for roleplay response outputs.
      nimRequest.reasoning_effort = reasoning_effort || 'low';
    } else {
      nimRequest.max_tokens = req.body.max_tokens || 4096;
      if (req.body.temperature !== undefined) nimRequest.temperature = req.body.temperature;
      if (req.body.top_p !== undefined) nimRequest.top_p = req.body.top_p;

      if (CHAT_TEMPLATE_REASONING_OFF[nimModel]) {
        nimRequest.chat_template_kwargs = {
          ...(nimRequest.chat_template_kwargs || {}),
          ...CHAT_TEMPLATE_REASONING_OFF[nimModel]
        };
      }
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
      res.setHeader('X-Accel-Buffering', 'no');
      res.flushHeaders();

      const keepAlive = setInterval(() => {
        if (!res.writableEnded) res.write(': keepalive\n\n');
      }, 15000);
      res.on('close', () => clearInterval(keepAlive));

      let buffer = '';
      let reasoningStarted = false;
      const stripInlineThink = makeThinkStripper();

      response.data.on('data', (chunk) => {
        try {
          buffer += chunk.toString();
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          lines.forEach(line => {
            if (line.startsWith('data: ')) {
              if (line.includes('[DONE]')) {
                const trailing = stripInlineThink.flush();
                if (trailing) {
                  res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: trailing }, finish_reason: null }] })}\n\n`);
                }
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
                    data.choices[0].delta.content = stripInlineThink.strip(content || '');
                  }
                  delete data.choices[0].delta.reasoning_content;
                }

                if (Array.isArray(data.choices) && data.choices.length === 0) {
                  return;
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

      response.data.on('end', () => {
        clearInterval(keepAlive);
        const trailing = stripInlineThink.flush();
        if (trailing && !res.writableEnded) {
          res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: trailing }, finish_reason: null }] })}\n\n`);
        }
        res.end();
      });
      response.data.on('error', () => {
        clearInterval(keepAlive);
        if (!res.headersSent) res.end();
      });
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
    if (error.response?.data?.on) {
      const chunks = [];
      for await (const c of error.response.data) chunks.push(c);
      error.response.data = Buffer.concat(chunks).toString();
    }

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
