import requests
import json
import ast

class LlamaWithContext:
    def __init__(self, model_name="llama3.2:1b", api_url="http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.api_url = api_url
        self.current_context = None

    def run_with_context(self, context):
        """Run the model with the provided context"""
        self.current_context = context
        print(f"\nModel initialized with context (length: {len(self.current_context)})")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ('exit', 'quit'):
                    break
                
                response = self._generate_response(user_input)
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                print("\nExiting conversation...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")

    def _generate_response(self, prompt):
        """Generate response using the current context"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "context": self.current_context,
            "stream": True
        }
        response = requests.post(self.api_url, json=payload, stream=True)
        reply = ""
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    try:
                        data = line.decode("utf-8")
                        chunk = json.loads(data)
                        print(chunk.get("response", ""), end="", flush=True)
                        reply += chunk.get("response", "")
                        # Update context if present in chunk
                        if "context" in chunk:
                            self.current_context = chunk["context"]
                    except Exception as e:
                        print(f"[Stream error: {e}]")
            print()  # Newline after streaming
            return reply.strip()
        else:
            response.raise_for_status()

def get_context_input():
    """Get and validate context input from user"""
    print("Enter the context array (e.g., [128006, 9125, 128007,...]):")
    while True:
        try:
            input_str = input("> ").strip()
            context = ast.literal_eval(input_str)
            if isinstance(context, list) and all(isinstance(x, int) for x in context):
                return context
            print("Please enter a valid list of integers")
        except (ValueError, SyntaxError) as e:
            print(f"Invalid input: {str(e)}. Please try again.")

def main():
    print("🦙 Llama Context Loader")
    print("----------------------")
    
    # Initialize model
    llama = LlamaWithContext()
    
    # Get context from user
    context = get_context_input()
    
    # Start conversation
    llama.run_with_context(context)

if __name__ == "__main__":
    main()