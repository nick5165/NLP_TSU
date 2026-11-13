import gensim


class WordVectorAnalyzer:
    NOUN_SUFFIX = "_NOUN"
    RESULT_COUNT = 10

    def __init__(self, model_path="cbow.txt"):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            model_path, binary=False
        )

    @classmethod
    def _noun_token(cls, word: str) -> str:
        token = word.strip().lower()
        if token.endswith(cls.NOUN_SUFFIX):
            return token
        if "_" in token:
            raise ValueError(f"Ожидалось имя существительное: {word}")
        return f"{token}{cls.NOUN_SUFFIX}"

    def nearest_nouns(self, positive_words, negative_words):
        positive = [self._noun_token(w) for w in positive_words]
        negative = [self._noun_token(w) for w in negative_words]

        for word in positive + negative:
            if word not in self.model:
                raise KeyError(f"Слово отсутствует в модели: {word}")

        topn = self.RESULT_COUNT
        while True:
            nearest = self.model.most_similar(
                positive=positive,
                negative=negative,
                topn=topn
            )
            noun_nearest = [
                (word, similarity)
                for word, similarity in nearest
                if word.endswith(self.NOUN_SUFFIX)
            ]
            if len(noun_nearest) >= self.RESULT_COUNT:
                return noun_nearest[: self.RESULT_COUNT]
            if topn >= len(self.model):
                raise RuntimeError("В модели недостаточно слов-существительных.")
            topn = min(topn * 2, len(self.model))


def main():
    target_pair = ("ветка", "акула")
    
    positive_words = ["сук", "пиранья", "осьминог"]
    negative_words = ["рыба"]

    analyzer = WordVectorAnalyzer("cbow.txt")

    expression = " + ".join(positive_words) + " - " + " - ".join(negative_words)
    print(f"Целевая пара преподавателя: {target_pair[0]}, {target_pair[1]}")
    print(f"Линейная комбинация: {expression}\n")
    print("10 самых близких существительных к результату:")

    results = analyzer.nearest_nouns(positive_words, negative_words)
    for index, (word, similarity) in enumerate(results, start=1):
        display_word = word.removesuffix(WordVectorAnalyzer.NOUN_SUFFIX)
        marker = " <-- целевое слово" if display_word in target_pair else ""
        print(f"{index:2d}. {display_word:<15} {similarity:.4f}{marker}")


if __name__ == "__main__":
    main()