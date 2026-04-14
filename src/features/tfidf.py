from sklearn.feature_extraction.text import TfidfVectorizer


# builds the vectorizer with the given params
def build_tfidf(
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int | float = 2,
    max_df: float | int = 0.95,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,  # dampens effect of high frequency terms
        stop_words="english",
    )


# fits on training text and returns the tfidf matrix
def fit_transform_tfidf(texts: list[str], vectorizer: TfidfVectorizer):
    return vectorizer.fit_transform(texts)


# transforms new text using the already fitted vectorizer
def transform_tfidf(texts: list[str], vectorizer: TfidfVectorizer):
    return vectorizer.transform(texts)
