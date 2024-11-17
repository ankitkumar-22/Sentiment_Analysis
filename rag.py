from transformers import GPT2Tokenizer
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from database import index
from news_scraper import gather_comprehensive_info

tokenizer_2 = GPT2Tokenizer.from_pretrained("gpt2")
llm = OpenAI(temperature=0.7)

def summarize_text(text: str, max_tokens: int) -> str:
    """Summarize the given text to fit within the specified token limit"""
    tokens = tokenizer_2.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer_2.decode(tokens[:max_tokens], skip_special_tokens=True) + "..."

def query_context(user_query: str, context_embedding: np.ndarray) -> List[Dict]:
    """Query Pinecone for relevant context using the user's question"""
    query_embedding = generate_embedding(user_query)
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=3,
        include_metadata=True
    )
    return results.matches

def generate_rag_response(user_query: str, original_text: str, context_results: List[Dict]) -> str:
    """Generate a response using the original text and retrieved context"""
    MAX_MODEL_TOKENS = 4097
    MIN_COMPLETION_TOKENS = 100

    original_summary = summarize_text(original_text, max_tokens=500)
    context_texts = [summarize_text(result.metadata.get('text', ''), max_tokens=200) 
                     for result in context_results]
    
    comprehensive_info = gather_comprehensive_info(user_query)
    combined_context = f"{original_text}\n\nAdditional Context:\n{comprehensive_info}"

    query_tokens = len(tokenizer_2.encode(user_query))
    template_tokens = 100 
    available_prompt_tokens = MAX_MODEL_TOKENS - query_tokens - template_tokens - MIN_COMPLETION_TOKENS

    context_tokens = tokenizer_2.encode(combined_context)
    if len(context_tokens) > available_prompt_tokens:
        truncated_context = tokenizer_2.decode(context_tokens[:available_prompt_tokens], 
                                              skip_special_tokens=True)
    else:
        truncated_context = combined_context

    prompt_template = PromptTemplate(
        input_variables=["query", "context"],
        template="""
        Based on the following context, please answer the question concisely. If the context doesn't 
        contain enough information, use your general knowledge but indicate this in your response.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.run(query=user_query, context=truncated_context)