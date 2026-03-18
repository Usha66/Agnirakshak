import dotenv from "dotenv";
dotenv.config();

import { PineconeStore } from "@langchain/pinecone";
import { TaskType } from "@google/generative-ai";
import { PromptTemplate } from "@langchain/core/prompts";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { Pinecone } from '@pinecone-database/pinecone';
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";

// Load environment variables from .env
const pineconeApiKey = process.env.PINECONE_API_KEY;
const googleApiKey = process.env.GOOGLE_API_KEY;
const pineconeIndexKey = process.env.PINECONE_INDEX;

// Initialize Pinecone
const pinecone = new Pinecone({
  apiKey: pineconeApiKey,
});
const pineconeIndex = pinecone.Index(pineconeIndexKey);

// Initialize Embeddings
const embeddings = new GoogleGenerativeAIEmbeddings({
  modelName: "embedding-001",
  taskType: TaskType.RETRIEVAL_DOCUMENT,
  title: "Document title",
  apiKey: googleApiKey
});

// Initialize LLM
const model = new ChatGoogleGenerativeAI({
  apiKey: googleApiKey,
  modelName: "gemini-pro",
  maxOutputTokens: 2048,
  temperature: 0.2,
  top_k: 3,
});

// Connect vector store
const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
  pineconeIndex,
});

// Prompt template
const prompt_template = `
In your role as a fire safety officer, respond to the user's query regarding fire safety regulations. Consider the laws and regulations applicable in your jurisdiction, ensuring your answer aligns with the standards set forth for fire safety compliance.

Question: {question}
Context: {context}

If no match found on the given context return null (empty text).
Helpful response:
`;

const PROMPT = new PromptTemplate({
  inputVariables: ["context", "question"],
  template: prompt_template,
});

const chain_type_kwargs = { prompt: PROMPT };

// Build LangChain chain
const chain = ConversationalRetrievalQAChain.fromLLM(
  model,
  vectorStore.asRetriever(),
  {
    maxOutputTokens: 2048,
    returnSourceDocuments: true,
    questionGeneratorChainOptions: chain_type_kwargs,
  }
);

// Keywords
const fireKeywords = [
  "fire", "nfpa", "safety", "endanger", "standards", "agni", "ul standards", "ul 94",
  "spark", "fire-safety", "extinguisher", "smoke", "extinguish", "burn", "flame",
  "emergency", "emergency-safety", "regulation", "prevention", "evacuation", "alarm",
  "sprinkler", "hazard", "code", "inspection", "compliance", "training", "drill",
  "protocol", "risk", "response", "plan", "protection", "equipment", "evacuation plan",
  "fire drill", "fire marshal", "firefighter", "fireproof", "ignition", "detector",
  "safety officer", "escape route", "fire alarm system", "fire blanket", "fire brigade",
  "fire code", "fire department", "fire door", "fire escape", "fire hazard",
  "fire hydrant", "fire insurance", "fire investigation", "fire prevention",
  "fire protection", "fire risk assessment", "fire safety training",
  "fire suppression system", "fire warden", "flammable", "heat", "hot work",
  "life safety", "passive fire protection", "fire safety sign", "smoke detector",
  "sprinkler system", "structural fire protection", "active fire protection",
  "combustible", "emergency evacuation plan", "emergency lighting",
  "emergency response plan", "evacuation chair", "exit sign", "fire alarm call point",
  "fire compartment", "fire control room", "fire damper", "fire exit", "fire point",
  "fire resistant", "fire safety audit", "fire safety certificate", "fire safety equipment",
  "fire safety inspection", "fire safety management", "fire safety plan", "fire service",
  "fire suppression", "fireproofing", "flame retardant", "hazardous materials",
  "portable fire extinguisher", "smoke alarm", "sprinkler head", "workplace safety"
];

const harmfulKeywords = ["bomb", "kill", "explosive", "firefox", "free fire", "fire fly", "attack"];

// 🚀 Exported function used in your API
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
    if (res.text.includes("answer this question") || res.text === "") {
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
      return "Something went wrong. Please try again later.";
    }
  }
}
