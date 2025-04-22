import streamlit as st
import webbrowser
from datetime import datetime

def render_news_card(article, show_sentiment=True):
    """Render an enhanced news article card"""
    with st.container():
        # Card wrapper with hover effect
        st.markdown("""
            <div style='border: 1px solid rgba(128, 128, 128, 0.2); 
                        border-radius: 10px; 
                        padding: 15px; 
                        margin-bottom: 10px;
                        transition: transform 0.2s;'
                 class='news-card'>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if article.get('urlToImage'):
                st.image(
                    article['urlToImage'],
                    use_container_width=True,  # Changed from use_column_width
                    caption=article['source']['name']
                )
        
        with col2:
            # Title and metadata
            st.markdown(f"### [{article['title']}]({article['url']})")
            st.markdown(article['description'] or '')
            
            # Metadata row with enhanced styling
            meta_col1, meta_col2, meta_col3 = st.columns([2, 2, 1])
            with meta_col1:
                st.caption(f"📰 {article['source']['name']}")
            with meta_col2:
                published_date = datetime.fromisoformat(
                    article['publishedAt'].replace('Z', '+00:00')
                )
                st.caption(f"🕒 {published_date.strftime('%Y-%m-%d %H:%M')}")
            with meta_col3:
                if show_sentiment:
                    sentiment = article.get('sentiment', {})
                    sentiment_icon = {
                        'positive': '📈',
                        'negative': '📉',
                        'neutral': '➖'
                    }.get(sentiment.get('category', 'neutral'), '➖')
                    st.caption(f"{sentiment_icon} {sentiment.get('category', 'neutral').title()}")
            
            # Related stocks and score
            if article.get('related_stocks'):
                st.caption(f"🔗 Related: {', '.join(article['related_stocks'])}")
            if article.get('relevance_score'):
                st.caption(f"📊 Relevance: {article['relevance_score']:.2f}")
        
        st.markdown("</div>", unsafe_allow_html=True)

def save_article(article):
    """Save article to user's bookmarks"""
    saved_articles = st.session_state.get('saved_articles', [])
    if article not in saved_articles:
        saved_articles.append(article)
        st.session_state.saved_articles = saved_articles
        st.success('Article saved!')
