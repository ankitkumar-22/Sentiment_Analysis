import streamlit as st
from image_processing import extract_text
from speech_processing import load_speech_model, record_audio, speech_to_text, wav2vec2_speech_to_text
from analysis import advanced_news_analysis
from visualization import generate_word_cloud, format_google_news_url
from rag import query_context, generate_rag_response

def main():
    st.set_page_config(page_title="Advanced News Analysis", page_icon="📰", layout="wide")
    st.title("📰 Advanced Context Aware News Analysis")

    speech_model = load_speech_model()

    # Sidebar
    st.sidebar.header("Upload Newspaper Cutout")
    uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    analysis_result = None

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        st.sidebar.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Performing advanced analysis..."):
            analysis_result = advanced_news_analysis(image, "Analyze this news article")

        display_analysis_results(analysis_result)
        handle_user_questions(analysis_result, speech_model)
    else:
        st.info("Please upload a newspaper cutout to get started.")

    display_sidebar_info()

def display_analysis_results(analysis_result):
    st.header("News Summary")
    st.write(analysis_result["summary"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Named Entities")
        st.write(analysis_result["entities"])
        st.subheader("Sentiment Analysis")
        st.write(f"Sentiment: {analysis_result['sentiment'][0]}")
        st.write(f"Confidence: {analysis_result['sentiment'][1]:.2f}")
    
    with col2:
        st.subheader("Word Cloud")
        word_cloud_plot = generate_word_cloud(analysis_result["text"])
        st.pyplot(word_cloud_plot)
    
    display_trend_analysis(analysis_result)
    display_related_news(analysis_result)

def display_trend_analysis(analysis_result):
    st.header("Trend Analysis")
    trending_up, trending_down = analysis_result["trend_analysis"]
    st.write("Trending Up:", ", ".join(trending_up))
    st.write("Trending Down:", ", ".join(trending_down))

def display_related_news(analysis_result):
    st.header("Related News Articles")
    if analysis_result["related_news"]:
        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]
        
        for i, article in enumerate(analysis_result["related_news"]):
            if article['url']:
                with columns[i % 3]:
                    try:
                        article_title = format_google_news_url(article['url'])
                        
                        with st.container():
                            st.markdown(
                                f"""
                                <div style="border:1px solid #ccc; border-radius:5px; padding:10px; margin-bottom:10px;">
                                    <h4>{article_title}</h4>
                                    <a href="{article['url']}" target="_blank">Open article →</a>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                    except Exception as e:
                        st.error(f"Error displaying article: {str(e)}")

def handle_user_questions(analysis_result, speech_model):
    st.header("Ask Questions About the Article")
    input_method = st.radio("Choose input method:", ("Text", "Custom Speech Model", "Wav2Vec2 Model"))
    user_question = ""

    if input_method == "Text":
        user_question = st.text_input("What would you like to know about this article?")
    else:
        if st.button("Start Recording"):
            with st.spinner("Listening..."):
                audio = record_audio()
                if input_method == "Custom Speech Model":
                    user_question = speech_to_text(audio, 16000, speech_model)
                else:  # Wav2Vec2 Model
                    user_question = wav2vec2_speech_to_text(audio, 16000)
            st.write(f"Recognized Text: {user_question}")

    if user_question:
        process_user_question(user_question, analysis_result)

def process_user_question(user_question, analysis_result):
    with st.spinner("Generating context-aware response..."):
        context_results = query_context(user_question, analysis_result["context_embedding"])
        rag_response = generate_rag_response(
            user_question, 
            analysis_result["text"], 
            context_results
        )
    
    st.subheader("Answer")
    st.write(rag_response)
    
    if st.checkbox("Show sources used for the answer"):
        st.subheader("Sources Referenced")
        for idx, result in enumerate(context_results, 1):
            st.markdown(f"**Source {idx}:**")
            st.write(result.metadata.get('text', '')[:200] + '...')

def display_sidebar_info():
    st.sidebar.markdown("""
    ## How to use:
    1. Upload a newspaper cutout image
    2. View the comprehensive analysis including summary, entities, sentiment, and more
    3. Explore related news and comparisons
    4. Ask questions about the article using text or speech input (custom model or Wav2Vec2)
    5. Check the sources used for answers if needed
    """)
    st.sidebar.markdown("Created by Keshav")

if __name__ == "__main__":
    main()