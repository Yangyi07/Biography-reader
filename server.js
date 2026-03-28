import dotenv from 'dotenv';
dotenv.config();
import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import OpenAI from 'openai';
import path from 'path';
import { fileURLToPath } from 'url';

// --- ESM Helper for __dirname ---
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// --- LangChain Imports ---
import { ChatOpenAI } from '@langchain/openai';
import { TavilySearch } from "@langchain/tavily";
import { createOpenAIFunctionsAgent, AgentExecutor } from '@langchain/classic/agents';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';

const app = express();
const port = process.env.PORT || 3000;

// Configure Middleware
app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));
app.use((req, res, next) => {
    console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
    next();
});

// Serve Static Files
app.use(express.static(path.join(__dirname, 'public'))); 
app.use('/Photos', express.static(path.join(__dirname, 'Photos')));

// Serve the Main HTML Application
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'code.html'));
});

// Configure OpenAI (Direct SDK for simple tasks)
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1';
const openai = OPENAI_API_KEY
    ? new OpenAI({
        apiKey: OPENAI_API_KEY,
        baseURL: OPENAI_BASE_URL
    })
    : null;

const MODEL_NAME = process.env.MODEL_NAME || 'gpt-4o';

// --- Startup Check ---
async function checkLLMConnection() {
    console.log("--- Checking LLM Connection ---");
    console.log(`Model: ${MODEL_NAME}`);
    console.log(`Base URL: ${process.env.OPENAI_BASE_URL || 'Default'}`);
    try {
        const testLLM = new ChatOpenAI({
            modelName: MODEL_NAME,
            openAIApiKey: process.env.OPENAI_API_KEY,
            configuration: { baseURL: process.env.OPENAI_BASE_URL },
            maxRetries: 1,
            timeout: 30000 // 30s for check
        });
        const res = await testLLM.invoke("Hello");
        console.log("LLM Connection Success! Response:", res.content);
    } catch (e) {
        console.error("!!! LLM Connection Failed !!!");
        console.error("Error details:", e.message || e);
        if (e.message && e.message.includes("429")) {
             console.error("Hint: Check your API quota/balance.");
        }
        if (e.message && e.message.includes("timeout")) {
             console.error("Hint: API response is slow, consider checking your network or proxy.");
        }
        console.error("Please check your API Key and Base URL in .env");
    }
    console.log("-------------------------------");
}

// --- LangChain Agent Setup ---
const searchTool = process.env.TAVILY_API_KEY
    ? new TavilySearch({
        maxResults: 3,
        apiKey: process.env.TAVILY_API_KEY
    })
    : null;
const tools = searchTool ? [searchTool] : [];

const llm = OPENAI_API_KEY
    ? new ChatOpenAI({
        modelName: MODEL_NAME,
        openAIApiKey: OPENAI_API_KEY,
        configuration: { baseURL: process.env.OPENAI_BASE_URL },
        temperature: 0.7,
        maxRetries: 2,
        timeout: 120000
    })
    : null;

// --- Helper: Safe JSON Parse ---
function parseJSONFromLLM(content) {
    try {
        return JSON.parse(content);
    } catch (e) {
        const jsonMatch = content.match(/```json\n([\s\S]*?)\n```/) || content.match(/```\n([\s\S]*?)\n```/);
        if (jsonMatch && jsonMatch[1]) {
            try { return JSON.parse(jsonMatch[1]); } catch (e2) {}
        }
        // Fallback extraction logic...
        const firstOpen = content.indexOf('{');
        const lastClose = content.lastIndexOf('}');
        if (firstOpen !== -1 && lastClose !== -1) {
             try { return JSON.parse(content.substring(firstOpen, lastClose + 1)); } catch (e3) {}
        }
        throw new Error("Could not parse JSON from LLM response");
    }
}

// --- Routes ---

// Health Check
app.get('/health', (req, res) => {
    res.json({ status: 'ok', model: MODEL_NAME });
});

// Task 2: Character Profile (Reliable Sequential Workflow)
app.post('/api/generate-profile', async (req, res) => {
    try {
        const { text, name } = req.body;
        if (!text || !name || text.trim().length === 0) {
            return res.status(400).json({ error: '请提供有效的文本或人物名称' });
        }
        if (!llm) {
            return res.status(500).json({ error: 'OPENAI_API_KEY 未配置' });
        }

        console.log(`Starting Sequential Profile Generation for: ${name}`);

        // --- Step 1: Decision & Search ---
        let enrichedContext = text;
        const decisionPrompt = `你是一位传记研究员。请判断以下关于 {name} 的文本是否足以构建深度心理画像（包含童年、性格、核心冲突、生平总结）。
        
        文本内容：
        ${text.substring(0, 2000)}
        
        如果文本不足以构建画像，请输出一个针对 {name} 的搜索查询语句（Search Query），以便查找其真实生平。
        如果文本已经足够，请输出 "ENOUGH"。
        
        你的回复格式必须是：
        SEARCH_QUERY: [查询语句]
        或者
        DECISION: ENOUGH`;

        const decisionRes = await llm.invoke(decisionPrompt.replace(/{name}/g, name));
        const decisionContent = decisionRes.content;
        console.log("Decision Step:", decisionContent);

        if (decisionContent.includes("SEARCH_QUERY:")) {
            const query = decisionContent.split("SEARCH_QUERY:")[1].trim().replace(/[\[\]]/g, "");
            console.log(`Searching for more info: ${query}`);
            try {
                if (searchTool) {
                    const searchResults = await searchTool.invoke({ input: query });
                    enrichedContext += "\n\n--- 补充搜索信息 ---\n" + searchResults;
                }
            } catch (searchErr) {
                console.warn("Search failed, continuing with original text:", searchErr.message);
            }
        }

        // --- Step 2: Final Generation ---
        const finalPrompt = ChatPromptTemplate.fromMessages([
            ["system", "你是一位专业的传记作家和心理侧写师。你的任务是为主角 {name} 建立深度心理画像。"],
            ["user", `请基于以下整合的信息（包含书籍片段和可能的搜索补充），输出一个JSON格式的心理画像。
            
            整合信息：
            ${enrichedContext.substring(0, 10000)}
            
            输出格式要求（必须是严格的JSON，不要有任何多余文字）：
            {{
                "name": "{name}",
                "core_traits": ["性格关键词1", "性格关键词2"],
                "speaking_style": "描述说话风格",
                "inner_conflict": "核心冲突",
                "emotional_triggers": ["雷点1", "雷点2"],
                "background_summary": "200字以内的人生背景总结（必须包含补充背景）"
            }}`]
        ]);

        const chain = finalPrompt.pipe(llm);
        const result = await chain.invoke({ name: name });
        
        const jsonResult = parseJSONFromLLM(result.content);
        console.log("Generation Success!");
        res.json(jsonResult);

    } catch (error) {
        console.error('Profile generation error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Task 1: Memory Index (Standard LLM)
app.post('/api/generate-memory-index', async (req, res) => {
    try {
        const { text, name, promptTemplate } = req.body;
        if (!text || !name) return res.status(400).json({ error: 'Missing text or name' });
        if (!openai) return res.status(500).json({ error: 'OPENAI_API_KEY 未配置' });

        const prompt = promptTemplate.replace(/{name}/g, name);
        const truncatedText = text.substring(0, 100000);

        const completion = await openai.chat.completions.create({
            model: MODEL_NAME,
            messages: [
                { role: "system", content: "You are an expert biographer." },
                { role: "user", content: `Context:\n${truncatedText}\n\nTask:\n${prompt}` }
            ],
            response_format: { type: "json_object" },
            temperature: 0.8
        });

        const result = parseJSONFromLLM(completion.choices[0].message.content);
        // Ensure it's an array
        const finalResult = Array.isArray(result) ? result : (result.events || result.nodes || []);
        
        res.json(finalResult);
    } catch (error) {
        console.error('Memory index generation error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Task 3: Path Deduction
app.post('/api/deduce-path', async (req, res) => {
    try {
        const { node, profile, promptTemplate } = req.body;
        if (!node || !profile) return res.status(400).json({ error: 'Missing node or profile' });
        if (!openai) return res.status(500).json({ error: 'OPENAI_API_KEY 未配置' });

        const name = profile.name;
        const nodeTitle = node.title;
        let prompt = promptTemplate.replace(/{name}/g, name).replace(/{nodeTitle}/g, nodeTitle);
        
        // Add context from profile and node description
        const context = `Character Profile: ${JSON.stringify(profile)}\n\nCurrent Event Node: ${JSON.stringify(node)}`;

        const completion = await openai.chat.completions.create({
            model: MODEL_NAME,
            messages: [
                { role: "system", content: "You are a creative writer specializing in parallel universe stories." },
                { role: "user", content: `Context:\n${context}\n\nTask:\n${prompt}` }
            ],
            response_format: { type: "json_object" },
            temperature: 0.8
        });

        const result = parseJSONFromLLM(completion.choices[0].message.content);
        res.json(result);
    } catch (error) {
        console.error('Path deduction error:', error);
        res.status(500).json({ error: error.message });
    }
});

// --- Image Generation (CogView-4) ---
async function generateImage(prompt) {
    const url = 'https://api.z.ai/api/paas/v4/images/generations';
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`, // Reuse Zhipu Key
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: "cogView-4-250304",
                prompt: prompt,
                size: "1024x1024"
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`CogView API Error: ${response.status} - ${errorText}`);
        }

        const data = await response.json();
        // Zhipu CogView usually returns { data: [{ url: "..." }] }
        if (data.data && data.data.length > 0) {
            return data.data[0].url;
        }
        throw new Error("No image URL in response");
    } catch (error) {
        console.error("Image Generation Failed:", error);
        throw error;
    }
}

// Task 4: Generate Scene Image with Character Consistency
app.post('/api/generate-scene-image', async (req, res) => {
    try {
        const { description, profile } = req.body;
        if (!description) return res.status(400).json({ error: 'Missing description' });
        if (!llm || !OPENAI_API_KEY) return res.status(500).json({ error: 'OPENAI_API_KEY 未配置' });

        console.log(`Generating image for scene: ${description.substring(0, 30)}...`);

        // Extract or construct character appearance string
        let characterAppearance = "";
        if (profile) {
            // Construct a consistent character prompt based on profile
            const traits = profile.core_traits ? profile.core_traits.join(", ") : "";
            characterAppearance = `主角特征：${profile.name}，${traits}。${profile.background_summary || ""}`;
        }

        // Step 1: Optimize Prompt with Consistency
        const promptGenPrompt = ChatPromptTemplate.fromMessages([
            ["system", "你是一位专业的 AI 绘画提示词工程师。你的任务是将中文场景描述转化为高质量的英文绘画提示词。"],
            ["user", `请基于以下信息生成英文提示词：
            
            【角色设定】(必须在画面中保持一致):
            ${characterAppearance}
            
            【当前场景】:
            ${description}
            
            要求：
            1. **核心要求**：必须严格遵循【角色设定】中的外貌特征（如年龄、发型、衣着风格），确保人物一致性。如果角色是 "${profile ? profile.name : '主角'}"，请确保画面中的人物符合其年龄段和身份特征。
            2. **风格要求**：Vintage film photography style, soft focus, slight film grain, nostalgic atmosphere, emotional storytelling, cinematic composition. Avoid overly sharp digital look.
            3. **构图要求**：Focus on mood and emotion, artistic composition.
            4. 只输出英文提示词，不要包含任何前缀、后缀或解释。`]
        ]);

        const chain = promptGenPrompt.pipe(llm);
        const promptResult = await chain.invoke({});
        const englishPrompt = promptResult.content.trim();
        console.log(`Generated Consistent Prompt: ${englishPrompt}`);

        // Step 2: Call CogView-4
        const imageUrl = await generateImage(englishPrompt);
        
        res.json({ imageUrl, prompt: englishPrompt });

    } catch (error) {
        console.error('Scene image generation error:', error);
        res.status(500).json({ error: error.message });
    }
});

export default app;

if (!process.env.VERCEL) {
    app.listen(port, () => {
        console.log(`LLM Backend Service running on http://localhost:${port}`);
        checkLLMConnection().catch(err => console.error("Background check failed:", err));
    });
}
