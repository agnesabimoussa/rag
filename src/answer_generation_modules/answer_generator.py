from src.data_models.search_result import (StudentSearchResults,
                                           MinimalSearchResults)
from src.data_models.search_answer import (StudentSearchResultsAndAnswer,
                                           MinimalAnswer)
from src.error_handling_modules.inavlid_json import InvalidJSON
from pathlib import Path
from typing import Optional
from pydantic import TypeAdapter, ValidationError
import json
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


class AnswerGenerator:
    def __init__(self,
                 student_search_results_path: str,
                 save_dir: str,
                 system_prompt: Optional[str] = None,
                 model_path: str = "Qwen/Qwen3-0.6B") -> None:
        self.student_search_results_path = Path(student_search_results_path)
        self.save_dir = Path(save_dir)
        self.search_results = self._read_search_results()
        self.k = self.search_results.k
        if not system_prompt:
            system_prompt = """You are a careful assistant answering questions from the retrieved source context only.
Answer directly and concisely. Be coherent and understandable, grounded in the provided sources, and avoid major 
hallucinations. Answer the question actually asked, not a broader topic.
Use the retrieved source excerpts as your evidence base and stay faithful to them.
"""
        self.system_prompt = system_prompt
        local_weights_dir = self._ensure_local_weights(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(local_weights_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            local_weights_dir,
            torch_dtype="auto",
            device_map="auto",
        )

    @staticmethod
    def _ensure_local_weights(model_path: str) -> str:
        slug = model_path.split("/")[-1].lower()
        local_dir = Path("models") / slug
        if not (local_dir / "config.json").is_file():
            snapshot_download(repo_id=model_path, local_dir=str(local_dir))
        return str(local_dir)

    def _read_search_results(self) -> StudentSearchResults:
        try:
            adapter = TypeAdapter(StudentSearchResults)
            with open(self.student_search_results_path, "r", encoding="utf-8") as file:
                content = json.load(file)
            return adapter.validate_python(content)
        except (ValidationError, json.JSONDecodeError):
            raise InvalidJSON("InvalidJSON exception occured."
                              f"{self.student_search_results_path} contains invalid JSON.")

    def answer_prompt(self, question: MinimalSearchResults) -> str:
        retrieved_sources = ", ".join(
            f"{source.file_path} [{source.first_character_index}:{source.last_character_index}]"
            for source in question.retrieved_sources
        )
        system_prompt = self.system_prompt + ". Retrieved sources: " + retrieved_sources
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question.question}
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
            )
        answer: str = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        return str(answer)

    def answer_dataset(self) -> StudentSearchResultsAndAnswer:
        search_results_and_answer = StudentSearchResultsAndAnswer(k=self.k, search_results=[])
        for search_result in tqdm(self.search_results.search_results, desc="Generating answers"):
            answer = MinimalAnswer(**search_result.model_dump(),
                                   answer=self.answer_prompt(search_result))
            search_results_and_answer.search_results.append(answer)
        return search_results_and_answer

    def write_answers(self) -> None:
        answers = self.answer_dataset()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        full_path = self.save_dir / self.student_search_results_path.name
        with open(full_path, "w", encoding="utf-8") as file:
            json.dump(answers.model_dump(),
                      file,
                      indent=4)
