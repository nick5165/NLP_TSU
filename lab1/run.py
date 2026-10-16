import re
import nltk
import pymorphy3
from pathlib import Path
from typing import List, Tuple, Optional


class PairExtractor:
    TARGET_POS = {"NOUN", "ADJF", "ADJS"}

    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
        self._init_nltk()

    @staticmethod
    def _init_nltk() -> None:
        resource = "tokenizers/punkt_tab"
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1], quiet=True)

    @staticmethod
    def _read_file(file_path: str | Path) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        return path.read_text(encoding="utf-8")

    def _get_valid_parses(self, word: str) -> List[pymorphy3.analyzer.Parse]:
        if not re.match(r"^[а-яА-ЯёЁ-]+$", word):
            return []

        parses = self.morph.parse(word)
        valid_parses = []

        for p in parses:
            tag_str = str(p.tag)

            if (
                p.tag.POS in self.TARGET_POS
                and "Apro" not in tag_str
                and "Npro" not in tag_str
                and p.score >= 0.01
            ):
                valid_parses.append(p)

        return valid_parses

    def _are_grammatically_agreed(
        self, p1: pymorphy3.analyzer.Parse, p2: pymorphy3.analyzer.Parse
    ) -> bool:
        if not p1.tag.number or not p2.tag.number or p1.tag.number != p2.tag.number:
            return False

        if p1.tag.case and p2.tag.case and p1.tag.case != p2.tag.case:
            return False

        if p1.tag.gender and p2.tag.gender and p1.tag.gender != p2.tag.gender:
            return False

        return True

    def _find_matching_parse_pair(
        self, word1: str, word2: str
    ) -> Optional[Tuple[str, str]]:
        parses1 = self._get_valid_parses(word1)
        parses2 = self._get_valid_parses(word2)

        if not parses1 or not parses2:
            return None

        for p1 in parses1:
            for p2 in parses2:
                if self._are_grammatically_agreed(p1, p2):
                    return p1.normal_form, p2.normal_form

        return None

    def extract_pairs_from_text(self, text: str) -> List[Tuple[str, str]]:
        result_pairs: List[Tuple[str, str]] = []

        sentences = nltk.sent_tokenize(text, language="russian")

        for sent in sentences:
            raw_tokens = nltk.word_tokenize(sent, language="russian")

            for i in range(len(raw_tokens) - 1):
                w1, w2 = raw_tokens[i], raw_tokens[i + 1]

                pair_lemmas = self._find_matching_parse_pair(w1, w2)
                if pair_lemmas:
                    result_pairs.append(pair_lemmas)

        return result_pairs

    def process_file(self, file_path: str | Path) -> List[Tuple[str, str]]:
        text = self._read_file(file_path)
        return self.extract_pairs_from_text(text)


if __name__ == "__main__":
    input_filename = Path(__file__).parent / "test_nlp_input.txt"

    extractor = PairExtractor()
    extracted_pairs = extractor.process_file(input_filename)

    print(f"Найдено согласованных пар: {len(extracted_pairs)}\n")
    print("Список пар (в виде лемм):")
    for lemma1, lemma2 in extracted_pairs:
        print(f"• {lemma1} {lemma2}")