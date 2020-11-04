from vectorizer import JaVectorizer


def test_encode():
    vectorizer = JaVectorizer()
    input_ids = vectorizer.encode(["こんにちは世界", "こんにちはトランプ大統領"])
    outputs = vectorizer.vectorize(input_ids)
    print(len(outputs.last_hidden_state))
    print(len(outputs.last_hidden_state[0][0][:].tolist()))