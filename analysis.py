import spacy
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from collections import Counter
from news_scraper import fetch_related_news, scrape_article_content
from database import store_embedding, generate_embedding
from text_processing import preprocess_text, summarize_news, extract_entities, get_sentiment

nlp = spacy.load("en_core_web_sm")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_context_embedding(text, related_news):
    """Create a combined embedding from the main text and related news"""
    related_texts = [article['title'] + ' ' + article.get('description', '') for article in related_news]
    combined_text = text + ' ' + ' '.join(related_texts)
    return generate_embedding(combined_text)

def analyze_trends(main_article, related_articles):
    """Analyze trending words"""
    main_words = Counter(main_article.lower().split())
    related_words = Counter(" ".join(related_articles).lower().split())
    
    trending_up = [word for word in main_words if main_words[word] > related_words[word]][:5]
    trending_down = [word for word in main_words if main_words[word] < related_words[word]][:5]
    
    return trending_up, trending_down

def advanced_news_analysis(image, user_query):
    """Perform comprehensive analysis on news image"""
    text = extract_text(image)
    preprocessed_text = preprocess_text(text)
    embedding = generate_embedding(preprocessed_text)
    store_embedding(embedding, {"text": preprocessed_text})
    entities = extract_entities(preprocessed_text)
    sentiment_label, sentiment_score = get_sentiment(preprocessed_text)
    summary = summarize_news(preprocessed_text)
    
    main_entity = list(entities.values())[0] if entities else ""
    related_news = fetch_related_news(main_entity)
    related_texts = [article['title'] for article in related_news]
    
    trend_analysis = analyze_trends(preprocessed_text, related_texts)
    context_embedding = create_context_embedding(preprocessed_text, related_news)
    
    return {
        "text": preprocessed_text,
        "summary": summary,
        "entities": entities,
        "sentiment": (sentiment_label, sentiment_score),
        "related_news": related_news,
        "trend_analysis": trend_analysis,
        "context_embedding": context_embedding
    }