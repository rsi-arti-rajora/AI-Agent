# managers/mom_manager.py
from llama_cpp import Llama
import os

class MoMManager:
    def __init__(self, model_path: str = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        try:
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=32,
                n_ctx=32768,
                verbose=False
            )
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load Llama model: {e}")

    def generate_from_transcript(self, transcript_path: str) -> str:
        if not os.path.exists(transcript_path):
            return "❌ Transcript file not found."

        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = f.read().strip()

        if len(transcript.split()) < 10:
            return "❌ Transcript too short or not meaningful. MoM generation skipped."

        prompt = f"""
You are an assistant that generates detailed Minutes of Meeting (MoM) from transcripts.
Split the output into two sections: 'Key Discussion Points' and 'Action Items'.
Each Action Item should be specific, assigned (if possible), and time-bound.

Transcript:
{transcript}

Minutes of Meeting:
"""
        try:
            response = self.llm.create_completion(prompt=prompt, max_tokens=512)
            return response["choices"][0]["text"].strip()
        except Exception as e:
            return f"❌ Error during MoM generation: {e}"
        
        

# Example usage:
if __name__ == "__main__":
    mom_manager = MoMManager()
    transcript_path = "sessions/testing/transcript.txt"  # Replace with your transcript file path
    mom_output = mom_manager.generate_from_transcript(transcript_path)
    print("Generated MoM:")
    print(mom_output)
    
