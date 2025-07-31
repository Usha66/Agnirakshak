const API = {
  GetChatbotResponse: async message => {
    try {
      const res = await fetch("http://localhost:5000/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ question: message })
      });
      const data = await res.json();
      return data.response;
    } catch (error) {
      console.error("Error fetching Bot response:", error);
      return "Error: Failed to fetch response";
    }
  }
};

export default API;
