import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:1b"

def ask_llm(prompt: str, context: list = None) -> tuple:
    """
    Send prompt to LLM with optional context.
    Returns (response_text, new_context) tuple.
    """

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "context": context if isinstance(context, list) else []
    }
    
    response = requests.post(OLLAMA_URL, json=payload, stream=True)
    reply = ""
    new_context = None
    
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    token = chunk.get("response", "")
                    print(token, end="", flush=True)
                    reply += token
                    
                    if chunk.get("done", False):
                        new_context = chunk.get("context", [])
                except Exception as e:
                    print(f"[Stream error: {e}]")
        print()
        print("Context:", context)
        return reply.strip(), new_context
    else:
        error = f"Error: {response.status_code} - {response.text}"
        print(error)
        print("Context:", context)
        return error, None

if __name__ == "__main__":
    print("🔁 Conversational LLM Query Engine")
    print("Type your question (or 'exit' to quit):")
    
    # Maintain token context for model state
    current_context = None
    # Maintain conversation history for display
    conversation_history = []
    
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        
        # Add user message to history
        conversation_history.append({"role": "user", "content": user_input})
        
        # Format current prompt with conversation history
        formatted_prompt = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in conversation_history
        ) + "\nAssistant: "
        
        # Get LLM response with current context
        reply, new_context = ask_llm(formatted_prompt, current_context)
        
        # Update context for next iteration
        if new_context is not None:
            current_context = new_context
        
        # Add assistant response to history
        conversation_history.append({"role": "assistant", "content": reply})
        print(f"\nLLM: {reply}\n")