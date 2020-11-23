# similar-text-search

Similar sentence search samples

## Usage

### Initial settings

```
# Clone this repository

$ docker build -t similar-text-search-ja .
$ docker run -it --gpus all -v $PWD:/opt/app -v ~/.gitconfig:/etc/gitconfig --rm similar-text-search-ja

# Attach to the container

# python -m venv .venv
# sourse .venv/bin/activate 

(.venv) # pip install -r requirements.txt
(.venv) # pip install transformers["ja"]  # Could someone tell how I could write this in requirements.txt?
```

### Index sample data

```
$ docker-compose up
$ docker exec -it python01 bash

# python -m venv .venv
# sourse .venv/bin/activate 
(.venv) # python -m similar_text_search_ja.index mlit-sample
```

### Try Elasticsearch More Like This search (TF-IDF based similar text search)

Open `http://kib.localhost/app/dev_tools#/console` from your browser and try this query:
```
GET mlit/_search
{
  "query": {
    "more_like_this": {
      "fields": [
        "申告内容の要約"
      ],
      "like": "ステアリングが動かない",
      "min_term_freq" : 1
    }
  }
}
```