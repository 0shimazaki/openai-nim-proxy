// server.js - OpenAI to NVIDIA NIM Proxy (Optimized for Chub AI & SillyTavern)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Display thinking tags in Chub AI chat UI
const SHOW_REASONING = false; 

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-4': 'deepseek-ai/deepseek-v4-flash-0731',
  'gpt-4-turbo': 'z-ai/glm-5.2',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro-0813',
  'claude-3-opus': 'meta/muse-glimmer-30b',
  'claude-3-sonnet': 'minimaxai/minimax-m3',
  'gemini-pro': 'moonshotai/kimi-k3' 
};

// Axios instance with 429 rate limit retry logic
const nimClient = axios.create({
  baseURL: NIM_API_BASE,
  timeout: 300000 // 5-minute timeout for deep reasoning
});

nimClient.interceptors.response.use(null, async (error) => {
  const { config, response } = error;
  if (response && response.status === 429 && (!config._retryCount || config._retryCount < 3)) {
    config._retryCount = (config._retryCount || 0) + 1;
    const delay = config._retryCount * 2000;
    console.log(`[429 Rate Limit Hit] Retrying request in ${delay}ms (Attempt ${config._retryCount}/3)...`);
    await new Promise(resolve => setTimeout(resolve, delay));
    return nimClient(config);
  }
  return Promise.reject(error);
});

app.get('/health', (req, res) => res.json({ status: 'ok', proxy: 'Chub AI to NVIDIA NIM' }));

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
  if (req.socket) req.socket.setTimeout(600000);

  try {
    const { model, messages, stream, reasoning_effort } = req.body;

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: { message: "Missing 'messages' array.", type: "invalid_request_error" } });
    }

    let nimModel = model ? (MODEL_MAPPING[model] || model) : 'moonshotai/kimi-k3';
    const isKimiModel = nimModel.includes('kimi-k3');

    // Reconstruct message history to preserve reasoning content for Kimi K3
    const finalMessages = messages.map(msg => {
      const cleanMsg = { role: msg.role, content: msg.content || '' };
      
      if (msg.reasoning_content) {
        cleanMsg.reasoning_content = msg.reasoning_content;
      } else if (msg.role === 'assistant' && typeof msg.content === 'string' && msg.content.includes('<think>')) {
        const match = msg.content.match(/<think>([\s\S]*?)<\/think>/);
        if (match) {
          cleanMsg.reasoning_content = match[1].trim();
          cleanMsg.content = msg.content.replace(/<think>[\s\S]*?<\/think>/, '').trim();
        }
      }
      return cleanMsg;
    });

    // Build payload strict to NIM requirements
    const nimRequest = {
      model: nimModel,
      messages: finalMessages,
      stream: Boolean(stream)
    };

    if (isKimiModel) {
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

    const response = await nimClient.post('/chat/completions', nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
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
                    data.choices[0].delta.content = content || '';
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
          message: typeof errorDetails === 'object' ? JSON.stringify(errorDetails) : errorDetails,
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
