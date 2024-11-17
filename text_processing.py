import re
import spacy
from transformers import AutoTokenizer, pipeline
from langchain_community.llms import OpenAI

nlp = spacy.load("en_core_web_sm")
sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentiment_pipeline = pipeline("sentiment-analysis", 
                             model="distilbert-base-uncased-finetuned-sst-2-english",
                             tokenizer=sentiment_tokenizer,
                             max_length=512,
                             truncation=True)
llm = OpenAI(temperature=0.7)

def preprocess_text(text):
    """Clean and preprocess text"""
    cleaned_text = text.lower()
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    return cleaned_text

def extract_entities(text):
    """Extract named entities from text"""
    doc = nlp(text)
    return {ent.label_: ent.text for ent in doc.ents}

def get_sentiment(text):
    """Perform sentiment analysis on text"""
    tokens = sentiment_tokenizer.encode(text, max_length=510, truncation=True)
    tokens = [sentiment_tokenizer.cls_token_id] + tokens + [sentiment_tokenizer.sep_token_id]
    
    if len(tokens) < 512:
        tokens = tokens + [sentiment_tokenizer.pad_token_id] * (512 - len(tokens))
    elif len(tokens) > 512:
        tokens = tokens[:512]
    
    truncated_text = sentiment_tokenizer.decode(tokens)
    result = sentiment_pipeline(truncated_text)[0]
    return result['label'], result['score']

def summarize_news(text):
    """Generate a summary of the news article"""
    summary_prompt = f"Summarize the following news article in a concise paragraph:\n\n{text}"
    result = llm.generate([summary_prompt])
    return result.generations[0][0].text