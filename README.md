# similar-text-search-ja
Find similar sentences using huggingface/transformers

## Usage

```
# Clone this repository

$ docker build similar-text-search-ja .
$ docker run -it --gpus all -v $PWD:/opt/app -v ~/.gitconfig:/etc/gitconfig --rm similar-text-search-ja

# Attach to the container

# python -m venv .venv
# sourse .venv/bin/activate 

(.venv) # pip install -r requirements.txt
(.venv) # pip install transformers["ja"]  # Could someone tell how I could write this in requirements.txt?
```