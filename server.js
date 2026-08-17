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

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false; // Set to true to enable chat_template_kwargs thinking parameter

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-4': 'deepseek-ai/deepseek-v4-flash-0731',
  'gpt-4-turbo': 'z-ai/glm-5.2',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro',
  'claude-3-opus': 'meta/muse-glimmer-30b',
  'claude-3-sonnet': 'minimaxai/minimax-m3',
  'gemini-pro': 'google/gemma-4-31b-it' 
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint (OpenAI compatible)
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

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { 
      model, 
      messages, 
      temperature, 
      max_tokens, 
      stream,
      top_p,
      frequency_penalty,
      presence_penalty 
    } = req.body;

    // 1. Defend against malformed pings or missing arrays
    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({
        error: { message: "Invalid request body or missing 'messages' array.", type: "invalid_request_error", code: 400 }
      });
    }
    
    // Smart model selection with fallback
    let nimModel = model ? MODEL_MAPPING[model] : undefined;
    if (!nimModel && model) {
      try {
        // Changed to use a separate variable to avoid shadowing the route's 'res' object
        const verificationCheck = await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        });
        
        if (verificationCheck.status >= 200 && verificationCheck.status < 300) {
          nimModel = model;
        }
      } catch (e) {}
    }

    // Secondary fallback routine if the model isn't mapped or verified
    if (!nimModel) {
      const modelLower = (model || '').toLowerCase();
      if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
        nimModel = 'meta/llama-3.1-405b-instruct';
      } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
        nimModel = 'meta/llama-3.1-70b-instruct';
      } else {
        nimModel = 'meta/llama-3.1-8b-instruct';
      }
    }
    
    // Prepare parameter overrides and dynamic configurations
    let finalMessages = [...messages];
    let finalTemperature = temperature;
    let finalTopP = top_p;
    let finalFrequencyPenalty = frequency_penalty;
    let finalPresencePenalty = presence_penalty;
    let finalRepetitionPenalty = undefined;

    if (finalTemperature === undefined) finalTemperature = 0.6;
    
    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: finalMessages,
      temperature: finalTemperature,
      top_p: finalTopP,
      frequency_penalty: finalFrequencyPenalty,
      presence_penalty: finalPresencePenalty,
      max_tokens: max_tokens || 9024,
      stream: stream || false,
      ...(finalRepetitionPenalty && { repetition_penalty: finalRepetitionPenalty })
    };
    
    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
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
        // 2. Encapsulate stream processing in its own try/catch to protect the core Node process
        try {
          buffer += chunk.toString();
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';
          
          lines.forEach(line => {
            if (line.startsWith('data: ')) {
              if (line.includes('[DONE]')) {
                res.write(line + '\n');
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
                      combinedContent += '</think>\n\n' + content;
                      reasoningStarted = false;
                    } else if (content) {
                      combinedContent += content;
                    }
                    
                    if (combinedContent) {
                      data.choices[0].delta.content = combinedContent;
                      delete data.choices[0].delta.reasoning_content;
                    }
                  } else {
                    if (content) {
                      data.choices[0].delta.content = content;
                    } else {
                      data.choices[0].delta.content = '';
                    }
                    delete data.choices[0].delta.reasoning_content;
                  }
                }
                res.write(`data: ${JSON.stringify(data)}\n\n`);
              } catch (e) {
                res.write(line + '\n');
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
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model || 'unspecified-model',
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy intercept error:', error.message);
    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: {
          message: error.message || 'Internal proxy execution error',
          type: 'invalid_request_error',
          code: error.response?.status || 500
        }
      });
    }
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found. Ensure your client app hits /v1/chat/completions with a POST request.`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// 3. Prevent listener execution on Vercel deployment infrastructures
if (!process.env.VERCEL) {
  app.listen(PORT, () => {
    console.log(`OpenAI to NVIDIA NIM Proxy running locally on port ${PORT}`);
  });
}

// Export the module app configuration natively for serverless environments
module.exports = app;
