import { PineconeStore } from "@langchain/pinecone";
import { TaskType } from "@google/generative-ai";
import { PromptTemplate } from "@langchain/core/prompts";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { Pinecone } from '@pinecone-database/pinecone';
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";

const pineconeApiKey = process.env.REACT_APP_PINECONE_API_KEY;
const googleApiKey = process.env.REACT_APP_GOOGLE_API_KEY;
const pineconeIndex_key = process.env.REACT_APP_PINECONE_INDEX;

const pinecone = new Pinecone({
    apiKey: pineconeApiKey,
});

const pineconeIndex = pinecone.Index(pineconeIndex_key);

const embeddings = new GoogleGenerativeAIEmbeddings({
    modelName: "embedding-001", // 768 dimensions
    taskType: TaskType.RETRIEVAL_DOCUMENT,
    title: "Document title",
    apiKey: googleApiKey
});

const model = new ChatGoogleGenerativeAI({
    apiKey: googleApiKey,
    modelName: "gemini-pro",
    maxOutputTokens: 2048,
    temperature: 0.2,
    top_k: 3,
});

const vectorStore = await PineconeStore.fromExistingIndex(
    embeddings,
    { pineconeIndex }
);

const prompt_template = `
    In your role as a fire safety officer, respond to the user's query regarding fire safety regulations. Consider the laws and regulations applicable in your jurisdiction, ensuring your answer aligns with the standards set forth for fire safety compliance.

    Question: {question}
    Context: {context}

    If not match found on the given context return null, i.e return empty text. Means if context is empty return null

    Helpful response:
`;

const PROMPT = new PromptTemplate({
    inputVariables: ["context", "question"],
    template: prompt_template,
});

const chain_type_kwargs = { "prompt": PROMPT };

const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorStore.asRetriever(),
    {
        maxOutputTokens: 2048,
        returnSourceDocuments: true,
        questionGeneratorChainOptions: chain_type_kwargs,
    }
);

const fireKeywords = ["fire", "nfpa", "safety", "endanger", "standards", "agni","ul standards", "ul 94", "spark", "fire-safety", "extinguisher", "smoke", "extinguish", "burn", "flame", "emergency", "emergency-safety", "regulation", "emergency", "fire", "safety", "regulation", "prevention", "emergency", "evacuation", "alarm", "extinguisher", "sprinkler", "hazard", "code", "inspection", "compliance", "training", "drill", "protocol", "risk", "response", "plan", "protection", "equipment", "evacuation plan", "fire drill", "fire marshal", "firefighter", "fireproof", "flame", "ignition", "smoke", "detector", "safety officer", "escape route", "fire alarm system", "fire blanket", "fire brigade", "fire code", "fire department", "fire door", "fire escape", "fire hazard", "fire hydrant", "fire insurance", "fire investigation", "fire marshal", "fire prevention", "fire protection", "fire risk assessment", "fire safety training", "fire suppression system", "fire warden", "flammable", "heat", "hot work", "life safety", "passive fire protection", "fire safety sign", "smoke detector", "sprinkler system", "structural fire protection", "active fire protection", "burn", "combustible", "emergency evacuation plan", "emergency lighting", "emergency response plan", "evacuation chair", "exit sign", "fire alarm call point", "fire blanket", "fire compartment", "fire control room", "fire damper", "fire door", "fire escape", "fire exit", "fire hydrant", "fire investigation", "fire marshal", "fire point", "fire resistant", "fire risk assessment", "fire safety audit", "fire safety certificate", "fire safety equipment", "fire safety inspection", "fire safety management", "fire safety plan", "fire safety training", "fire service", "fire suppression", "fire warden", "fireproofing", "flame retardant", "hazardous materials", "passive fire protection", "portable fire extinguisher", "smoke alarm", "smoke detector", "sprinkler head", "workplace safety"];
const harmfulKeywords = ["bomb", "kill", "explosive", "firefox","free fire","fire fly","attack"];

export async function searchSimilarQuestions(question) {
    const containsHarmfulKeyword = harmfulKeywords.some(keyword =>
        question.toLowerCase().includes(keyword)
    );

    if (containsHarmfulKeyword) {
        return "Your query contains harmful and dangerous content. Please provide a different prompt.";
    }

    const containsFireKeyword = fireKeywords.some(keyword =>
        question.toLowerCase().includes(keyword)
    );

    if (!containsFireKeyword) {
        return "Sorry, I am trained on fire-related data only.";
    }

    try {
        const res = await chain.invoke({ question, chat_history: "" });
        if (res.text.includes('answer this question') || res.text === '') {
            const result = await model.invoke(question);
            return result.content;
        } else {
            return res.text;
        }
    } catch (error) {
        if (error.response && error.response.status === 404) {
            return "I don't know, it's not in my knowledge.";
        } else {
            console.error("Error occurred during similarity search:", error);
            return null;
        }
    }
}
