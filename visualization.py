import matplotlib.pyplot as plt
from wordcloud import WordCloud

def generate_word_cloud(text):
    """Generate a word cloud visualization"""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

def format_google_news_url(url: str) -> str:
    """Clean and format Google News URLs for better display"""
    try:
        article_id = url.split('read/')[-1].split('?')[0]
        shortened_id = article_id[:8]
        return f"Article {shortened_id}..."
    except Exception:
        return "News article"