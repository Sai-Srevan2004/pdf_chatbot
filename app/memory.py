
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

class ConversationMemory:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: list[BaseMessage] = []

    def add_turn(self, question: str, answer: str):
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=answer))

        max_messages = self.max_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def get_history(self) -> list[BaseMessage]:
        return self.history

    def clear(self):
        self.history = []

    def summary(self) -> list[dict]:
        result = []
        for msg in self.history:
            result.append({
                "role":    "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content
            })
        return result