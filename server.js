// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE
const SHOW_REASONING = false; // Set to true to format reasoning with <think> tags

// Model mapping
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-4': 'deepseek-ai/deepseek-v4-flash-0731',
  'gpt-4-turbo': 'z-ai/glm-5.2',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro-0813',
  'claude-3-opus': 'meta/muse-glimmer-30b',
  'claude-3-sonnet': 'minimaxai/minimax-m3',
  'gemini-pro': 'moonshotai/kimi-k3' 
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING
  });
});

// List models endpoint
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  // Prevent Node/Express socket drops during long reasoning phases
  if (req.socket) req.socket.setTimeout(600000);

  try {
    const { 
      model, 
      messages, 
      temperature, 
      max_tokens, 
      stream,
      top_p,
      frequency_penalty,
      presence_penalty,
      reasoning_effort
    } = req.body;

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({
        error: { message: "Invalid request body or missing 'messages' array.", type: "invalid_request_error", code: 400 }
      });
    }
    
    // Resolve NIM model identifier
    let nimModel = model ? MODEL_MAPPING[model] : undefined;
    if (!nimModel && model) {
      nimModel = model; // Direct pass-through if custom model string provided
    }

    if (!nimModel) {
      nimModel = 'meta/llama-3.1-8b-instruct';
    }
    
    const isKimiModel = nimModel.includes('kimi-k3');

    // Clean and normalize incoming message array
    const finalMessages = messages.map(msg => {
      const cleanMsg = { role: msg.role, content: msg.content || '' };
      
      // Preserve or extract reasoning_content required by Kimi K3 for multi-turn
      if (msg.reasoning_content) {
        cleanMsg.reasoning_content = msg.reasoning_content;
      } else if (msg.role === 'assistant' && typeof msg.content === 'string' && msg.content.includes('<think>')) {
        const match = msg.content.match(/<think>([\s\S]*?)<\/think>/);
        if (match) {
          cleanMsg.reasoning_content = match[1].trim();
        }
      }
      
      return cleanMsg;
    });

    // Kimi K3 requires larger max_tokens so reasoning doesn't consume all output tokens
    const nimRequest = {
      model: nimModel,
      messages: finalMessages,
      max_tokens: max_tokens || (isKimiModel ? 16384 : 4096),
      stream: stream || false
    };

    // Sanitize parameters for Kimi K3 vs Standard Models
    if (isKimiModel) {
      nimRequest.temperature = 1.0;
      nimRequest.top_p = top_p !== undefined ? top_p : 0.95;
      nimRequest.reasoning_effort = reasoning_effort || 'low';
    } else {
      nimRequest.temperature = temperature !== undefined ? temperature : 0.6;
      if (top_p !== undefined) nimRequest.top_p = top_p;
      if (frequency_penalty !== undefined) nimRequest.frequency_penalty = frequency_penalty;
      if (presence_penalty !== undefined) nimRequest.presence_penalty = presence_penalty;
      if (reasoning_effort !== undefined) nimRequest.reasoning_effort = reasoning_effort;
    }
    
    // Dispatch request with explicit Accept headers for NIM streaming
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json',
        'Accept': stream ? 'text/event-stream' : 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 300000 // 5-minute timeout
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

                  // Retain reasoning_content in delta if present so frontend can pass it back
                  if (!SHOW_REASONING && reasoning) {
                    data.choices[0].delta.reasoning_content = reasoning;
                  }
                }
                res.write(`data: ${JSON.stringify(data)}\n\n`);
              } catch (e) {
                res.write(line + '\n\n');
              }
            }
          });
        } catch (streamError) {
          console.error('Runtime error parsing active stream chunk:', streamError.message);
        }
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream pipeline hardware error:', err);
        if (!res.headersSent) res.end();
      });
    } else {
      // Non-streaming response formatting
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
          
          const messageObj = {
            role: choice.message.role,
            content: fullContent
          };

          if (choice.message?.reasoning_content) {
            messageObj.reasoning_content = choice.message.reasoning_content;
          }

          return {
            index: choice.index,
            message: messageObj,
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
    console.error(`Proxy Intercept Error [${status}]:`, JSON.stringify(errorDetails));

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

// Catch-all route
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found. Ensure your client app hits /v1/chat/completions with a POST request.`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

let server;
if (!process.env.VERCEL) {
  server = app.listen(PORT, () => {
    console.log(`OpenAI to NVIDIA NIM Proxy running locally on port ${PORT}`);
  });
  server.timeout = 600000;
}

module.exports = app;
