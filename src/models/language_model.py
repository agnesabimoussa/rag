from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.models.model_download import ModelDownload


class LLM:
    def __init__(self, model_path: str = "Qwen/Qwen3-0.6B", system_prompt: str = None):
        local_weights_dir = ModelDownload._ensure_local_weights(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(local_weights_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            local_weights_dir,
            torch_dtype="auto",
            device_map="auto",
        )
        if not system_prompt:
            system_prompt = """You are a grounded question-answering assistant. Answer the user's question using
            only the information provided in the retrieved sources.
            - Do not use outside knowledge or make assumptions.
            - If the sources do not contain enough information to answer, say: "The provided sources do
            not contain enough information to answer this question."
            - Give a clear, coherent, and direct answer.
            - Keep your answer to a maximum of 2 sentences.
            - Do not mention the retrieval process or refer to the sources as "chunks."
            """
        self.system_prompt = system_prompt

    def add_user_message(self,
                         messages: List[Dict[str, str]],
                         message: str) -> None:
        user_message = {"role": "user", "content": message}
        messages.append(user_message)

    def add_assistant_message(self,
                              messages: List[Dict[str, str]],
                              message: str) -> None:
        assistant_message = {"role": "system", "content": message}
        messages.append(assistant_message)

    def chat(self,
             messages: List[Dict[str, str]],
             enable_thinking: bool = False,
             max_new_tokens: float = 1000,
             do_sample: bool = True,
             temperature: float = 0.7,
             top_p: float = 0.8,
             top_k: float = 20,
             repetition_penalty: float = 1.05) -> str:
        self.add_assistant_message(messages, self.system_prompt)
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        answer: str = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        return str(answer)
