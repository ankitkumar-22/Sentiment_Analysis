import requests
from bs4 import BeautifulSoup
import random
import time

def fetch_related_news(query: str) -> List[Dict[str, str]]:
    """Fetch related news articles from Google News"""
    url = f"https://news.google.com/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    max_retries = 3
    articles = []
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            selectors = ['div[jscontroller="d0DtYd"]', 'article', 'div.NiLAwe', 
                         'h3.ipQwMb', 'div.xrnccd']
            
            for selector in selectors:
                items = soup.select(selector)
                if items:
                    for item in items[:5]:
                        title_elem = item.select_one('h3 a, h4 a, a')
                        if title_elem:
                            title = title_elem.text.strip()
                            link = title_elem.get('href', '')
                            
                            if link.startswith('./'):
                                link = 'https://news.google.com' + link[1:]
                            elif not link.startswith('http'):
                                link = 'https://news.google.com' + link
                            
                            snippet_elem = item.select_one('div[jsname="sngebd"], div.GI74Re')
                            snippet = snippet_elem.text.strip() if snippet_elem else ''
                            
                            articles.append({
                                'title': title,
                                'url': link,
                                'description': snippet
                            })
                    
                    if articles:
                        break
            
            if articles:
                break
            else:
                time.sleep(random.uniform(1, 3))
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to fetch news: {str(e)}")
            time.sleep(random.uniform(1, 3))
    
    return articles

def scrape_article_content(url: str) -> str:
    """Scrape the main content of an article"""
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.text for p in paragraphs])
        return content
    except Exception as e:
        print(f"Failed to scrape article content: {str(e)}")
        return ""

def gather_comprehensive_info(query: str) -> str:
    """Gather comprehensive information about a topic"""
    articles = fetch_related_news(query)
    selected_articles = articles[:3]

    comprehensive_info = f"Information about {query}:\n\n"
    
    for article in selected_articles:
        comprehensive_info += f"Title: {article['title']}\n"
        comprehensive_info += f"Description: {article['description']}\n"
        content = scrape_article_content(article['url'])
        comprehensive_info += f"Content: {content[:500]}...\n\n"
    
    return comprehensive_info