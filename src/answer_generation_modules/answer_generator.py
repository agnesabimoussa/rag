from src.data_models.search_result import (StudentSearchResults,
                                           MinimalSearchResults)
from src.data_models.search_answer import (StudentSearchResultsAndAnswer,
                                           MinimalAnswer)
from src.error_handling_modules.inavlid_json import InvalidJSON
from pathlib import Path
from pydantic import TypeAdapter, ValidationError
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class AnswerGenerator:
    model_path = "./models/qwen3-0.6b"

    def __init__(self,
                 student_results_path: str,
                 save_dir: str,
                 system_prompt: str | None = None) -> None:
        self.student_results_path = Path(student_results_path)
        self.save_dir = Path(save_dir)
        self.search_results = self._read_search_results()
        self.k = self.search_results.k
        if not system_prompt:
            system_prompt = """A default prompt"""
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
        )

    def _read_search_results(self) -> StudentSearchResults:
        try:
            adapter = TypeAdapter(StudentSearchResults)
            with open(self.student_results_path, "r", encoding="utf-8") as file:
                content = json.load(file)
            search_results = adapter.validate_python(content)
            return search_results
        except ValidationError:
            raise InvalidJSON("InvalidJSON exception occured."
                              f"{self.student_results_path} contains invalid JSON.")

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
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
            )
        answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        return answer

    def answer_dataset(self) -> StudentSearchResultsAndAnswer:
        search_results_and_answer = StudentSearchResultsAndAnswer(k=self.k, search_results=[])
        for search_result in self.search_results.search_results:
            answer = MinimalAnswer(**search_result.model_dump(),
                                   answer=self.answer_prompt(search_result))
            search_results_and_answer.search_results.append(answer)
        return search_results_and_answer
    
    def write_answers(self) -> None:
        answers = self.answer_dataset()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        full_path = self.save_dir / self.student_results_path.name
        with open(full_path, "w", encoding="utf-8") as file:
            json.dump(answers.model_dump(),
                      file,
                      indent=4)
