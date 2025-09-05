import torch
import torch.nn.functional as F
import os
from transformers import AutoTokenizer

class memorymanager():
    def __init__(self,tokenizer=None):        
        if tokenizer is None:
            raise ValueError("Tokenizer from HF must be provided.")
        
        self.tokenizer=AutoTokenizer.from_pretrained(tokenizer)

        self.short_mem_dir="./files/short_term"
        self.medium_term_dir="./files/medium_term"
        self.long_mem_dir="./files/long_term"

        self.short_mem= "mem.pt"
        self.medium_mem= "mem.txt"
        self.long_mem = "mem.txt"

        self.short_context=50
        self.medium_context=200

    
    """ Short Term Memory (using nested lists inside json)
    Structure of .pt file:
    [
    "<ctx> context 1 <inp> input 1 <out> output 1",
    "<ctx> context 2 <inp> input 2 <out> output 2",
    ...
    ]
    
    """
    def save_short_term(self, context, inp, out):
        

        file_path = f"{self.short_mem_dir}/{self.short_mem}"
        # Load existing data or initialize
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            data = torch.load(file_path)
        else:
            data = []

        # Format as a single string, then tokenize
        entry_str = f"<ctx> {context} <inp> {inp} <out> {out}"
        entry_tensor = self.tokenizer(entry_str, return_tensors="pt")["input_ids"].squeeze(0)
        data.append(entry_tensor)

        if len(data) > self.short_context:
            data.pop(0)

        torch.save(data, file_path)
        
    def search_short_term(self,query,topk=5):
        file_path = f"{self.short_mem_dir}/{self.short_mem}"
        data = torch.load(f"{self.short_mem_dir}/{self.short_mem}")
        query_tensor = self.tokenizer(query, return_tensors="pt")["input_ids"].squeeze(0)


        if os.path.getsize(file_path) == 0:
            return ["Short term memory was empty.",0]
        
        best_matches=[]
        for tensor in data:
            decoded = self.tokenizer.decode(tensor)
            if len(best_matches)<topk:
                score = F.cosine_similarity(query_tensor.float(), tensor.float(), dim=0).item()
                best_matches.append((decoded,score))
                best_matches = sorted(best_matches, key=lambda x: x[1], reverse=True)
            else:
                score = F.cosine_similarity(query_tensor.float(), tensor.float(), dim=0).item()
                if score > best_matches[-1][1]:
                    best_matches[-1] = (decoded,score)
                    best_matches = sorted(best_matches, key=lambda x: x[1], reverse=True)
        
        print(best_matches)
        return best_matches
                
        



#testing

manager = memorymanager("gpt2")
manager.save_short_term("You are Neura an ai Vtuber","How are you?","I am fine, thank you!")
manager.save_short_term("You are Neura an ai Vtuber","What is your name?","My name is Neura!")
manager.save_short_term("You are Neura an ai Vtuber","What do you do?","I chat with people!")

manager.save_short_term("AI is the future","What is AI?","AI stands for Artificial Intelligence.")
manager.save_short_term("AI is the future","How does AI work?","AI works by learning from data.")
results = manager.search_short_term("example",topk=3)
results = manager.search_short_term("name",topk=3)
results = manager.search_short_term("Artificial",topk=3)
results = manager.search_short_term("AI",topk=3)
