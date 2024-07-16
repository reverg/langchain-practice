# langchain-practice
## RAG Chatbot
A simple RAG chatbot that responds based on the contents of a PDF file.

### How to run
#### Requirements
* [OpenAI API key](https://platform.openai.com/api-keys)
* PDF file containing your data
  * Rename it as `sample.pdf` or change `server.py` line 13
  * Place in the same directory as `server.py`

#### Local environment
```bash
pip install -r requirements.txt
OPENAI_API_KEY="your_key" streamlit run server.py
```

#### Using Docker
```bash
docker build -t langchain-practice .
docker run -e OPENAI_API_KEY="your_key" -p 8501:8501 langchain-practice
```

## Tutorials
Practices based on [official tutorials](https://python.langchain.com/v0.2/docs/tutorials/) of LangChain