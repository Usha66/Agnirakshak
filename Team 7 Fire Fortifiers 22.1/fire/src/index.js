import React, { useState, useEffect } from "react";
import ReactDOM from "react-dom";

import BotMessage from "./components/BotMessage";
import UserMessage from "./components/UserMessage";
import Messages from "./components/Messages";
import Input from "./components/Input";
import Footer from "./components/Footer";
import Navbar from "./components/Navbar";

import API from "./ChatbotAPI";

import "./styles.css";

function Chatbot() {
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    async function loadWelcomeMessage() {
      const welcomeMessage = "Hello, I'm AgniRakshak, Your fire assistant. How can I assist you?";
      setMessages([<BotMessage key="0" fetchMessage={() => welcomeMessage} />]);
    }
    loadWelcomeMessage();
  }, []);



  const send = async text => {
    const newMessages = messages.concat(
      <UserMessage key={messages.length + 1} text={text} />,
      <BotMessage
        key={messages.length + 2}
        fetchMessage={async () => await API.GetChatbotResponse(text)}
      />
    );
    setMessages(newMessages);
  };




  return (
    <div className="maindiv">
      <Navbar />
      <div className="chatsection">
        <div className="chatbot">
          <div className="header">&nbsp;AgniRakshak</div>
          <div className="bot-message" class>
            <Messages messages={messages} />
          </div>
          <Input onSend={send} />
        </div>
      </div>
      <Footer />
    </div>
  );
}



const rootElement = document.getElementById("root");
ReactDOM.render(<Chatbot />, rootElement);
