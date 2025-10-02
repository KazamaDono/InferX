from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 

# Load a small model for demo purposes
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Simple "naive scheduler"
class NaiveScheduler:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size 
        self.queue = []
    
    def add_request(self, prompt: str):
        self.queue.append(prompt)
        if len(self.queue) >= self.batch_size:
            return self.run_batch()
        return None  # Return None if batch not ready
    
    def run_batch(self):
        inputs = tokenizer(self.queue, return_tensors="pt", padding=True).to(device)
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        
        # Only decode the NEW tokens (not the input prompt)
        input_length = inputs["input_ids"].shape[1] # [batch_size, sequence_length] -> getting sequence_length
        new_tokens = outputs[:, input_length:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        
        results = list(zip(self.queue, decoded))
        self.queue = []
        return results

if __name__ == "__main__":
    scheduler = NaiveScheduler(batch_size=3)
    
    prompts = [
        "The future of AI is",
        "Once upon a time",
        "In a galaxy far away",
        "Python is great because",
        "Data science is"
    ]
    
    # Simulate sequential requests 
    for prompt in prompts:
        result = scheduler.add_request(prompt)
        if result:
            print("Batch completed:")
            for inp, out in result:
                print(f"Prompt: {inp}\nGenerated: {out}\n")
    
    # Process remaining prompts in queue
    if scheduler.queue:
        print("\nProcessing remaining prompts...")
        result = scheduler.run_batch()
        for inp, out in result:
            print(f"Prompt: {inp}\nGenerated: {out}\n")