import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import { searchSimilarQuestions } from "./langchain.js";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

app.post("/api/ask", async (req, res) => {
  const { question } = req.body;
  try {
    const response = await searchSimilarQuestions(question);
    res.json({ response });
  } catch (err) {
    console.error("Error:", err);
    res.status(500).json({ response: "Internal Server Error" });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
