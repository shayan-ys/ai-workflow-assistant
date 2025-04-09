from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
# from langchain.schema.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI


class ChatClient:
    def __init__(self, memory: ConversationBufferMemory = None):
        self.chat_model = self.get_chat_model()
        self.memory = memory or self.get_memory()
        self.conversation_chain = self.create_conversation_chain(self.memory)
        self.history = []
        self.model_temperature = 0.7
        self.model = "llama3"

    def get_chat_model(self):
        return ChatOpenAI(
            temperature=self.model_temperature,
            openai_api_base="http://localhost:1234/v1",
            openai_api_key="lm-studio",  # Dummy key for LM Studio
            model=self.model,
        )

    def get_memory(self):
        return ConversationBufferMemory(return_messages=True)

    def create_conversation_chain(self):
        return ConversationChain(
            llm=self.chat_model,
            memory=self.memory,
            verbose=True
        )
