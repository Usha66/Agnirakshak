import { searchSimilarQuestions } from "./lang.mjs";
const API = {
  GetChatbotResponse: async message => {
    try {
      const botMessage = await searchSimilarQuestions(message);
      return botMessage;
    } catch (error) {
      console.error("Error fetching Bot response:", error);
      return "Error: Failed to fetch response";
    }
  },
};

export default API;