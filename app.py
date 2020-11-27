import pandas as pd
import streamlit as st

from similar_text_search_ja.config import Config
from similar_text_search_ja.es import ES
from similar_text_search_ja.vectorizers import SentenceVectorizer

if __name__ == "__main__":
    st.title("日本語類似検索デモ")
    query = st.text_input("これと似ている文章を探す:")
    if query:
        conf = Config("mlit-sample")
        es = ES([conf.es_url])
        vectorizer = SentenceVectorizer.create(
            {
                SentenceVectorizer.CONF_KEY_TRAINED_MODEL_PATH: str(conf.model_dir),
                SentenceVectorizer.CONF_KEY_TRANSFORMER_MODEL_NAME: conf.transformer_model,
            }
        )
        vector = vectorizer.vectorize([query])[0]
        result = es.cosine_by_vector(
            conf.es_index_name, conf.es_embedding_field, vector
        )

        scores = []
        anss = []
        texts = []
        for d in es.get_hit(result):
            scores.append(d["_score"])
            anss.append(d["_source"][conf.ans_field])
            texts.append(d["_source"][conf.target_fields[0]])

        df = pd.DataFrame({"scores": scores, "answers": anss, "text": texts})
        st.table(df)
