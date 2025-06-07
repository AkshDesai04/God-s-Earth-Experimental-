import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:1b"

def ask_llm(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "").strip()
    else:
        return f"Error: {response.status_code} - {response.text}"

if __name__ == "__main__":
    print("🔁 Stateless LLM Query Engine")
    print("Type your question (or 'exit' to quit):")

    while True:
        query = input("User: ").strip()
        if query.lower() in ("exit", "quit"):
            break

        reply = ask_llm(query)
        print(f"LLM: {reply}\n")
