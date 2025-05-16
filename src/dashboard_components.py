"""
Dashboard components for Mistral Docs Assistant.
Provides visualizations and insights about document content.
"""
import streamlit as st
import pandas as pd
import re
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import altair as alt
from collections import Counter

def render_document_stats(document_analysis: Dict[str, Dict[str, Any]]):
    """Render document statistics dashboard."""
    if not document_analysis:
        st.info("No document analysis available. Please process documents first.")
        return
    
    # Extract basic stats for all documents
    doc_stats = []
    total_words = 0
    total_chars = 0
    
    for filename, analysis in document_analysis.items():
        text_analysis = analysis.get("text_analysis", {})
        word_count = text_analysis.get("word_count", 0)
        char_count = text_analysis.get("char_count", 0)
        
        doc_stats.append({
            "Document": filename,
            "Words": word_count,
            "Characters": char_count,
            "Sentences": text_analysis.get("sentence_count", 0),
            "Reading Time (min)": text_analysis.get("reading_time_minutes", 0)
        })
        
        total_words += word_count
        total_chars += char_count
    
    # Overall stats
    st.subheader("📊 Overall Document Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", len(document_analysis))
    with col2:
        st.metric("Total Words", total_words)
    with col3:
        st.metric("Total Characters", total_chars)
    
    # Document comparison chart
    if len(doc_stats) > 1:  # Only show comparison if multiple documents
        st.subheader("Document Comparison")
        df = pd.DataFrame(doc_stats)
        
        # Word count chart
        word_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Document', sort='-y', title=None),
            y=alt.Y('Words', title='Word Count'),
            color=alt.Color('Document', legend=None),
            tooltip=['Document', 'Words', 'Sentences', 'Reading Time (min)']
        ).properties(
            height=300
        )
        
        st.altair_chart(word_chart, use_container_width=True)
    
    # Document stats table
    st.subheader("Document Details")
    stats_df = pd.DataFrame(doc_stats)
    st.dataframe(stats_df, use_container_width=True)

def render_content_insights(document_analysis: Dict[str, Dict[str, Any]]):
    """Render content insights and keyword analysis."""
    if not document_analysis:
        return
    
    st.subheader("🔍 Content Insights")
    
    # Combine all keywords for overall topic analysis
    all_keywords = []
    for filename, analysis in document_analysis.items():
        keywords = analysis.get("keywords", [])
        all_keywords.extend(keywords)
    
    keyword_counts = Counter(all_keywords)
    top_keywords = keyword_counts.most_common(10)
    
    # Create keyword chart
    if top_keywords:
        keywords_df = pd.DataFrame(top_keywords, columns=["Keyword", "Count"])
        
        keyword_chart = alt.Chart(keywords_df).mark_bar().encode(
            x=alt.X('Count:Q'),
            y=alt.Y('Keyword:N', sort='-x'),
            color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues'), legend=None),
            tooltip=['Keyword', 'Count']
        ).properties(
            title="Common Topics Across Documents",
            height=min(300, len(top_keywords) * 25)  # Dynamic height based on number of keywords
        )
        
        st.altair_chart(keyword_chart, use_container_width=True)
    
    # Document-specific keyword analysis
    if len(document_analysis) > 1:  # Only if multiple documents
        st.subheader("Topic Distribution by Document")
        
        # Create a matrix of keywords by document
        unique_keywords = list(set(all_keywords))
        keyword_matrix = []
        
        for filename, analysis in document_analysis.items():
            doc_keywords = analysis.get("keywords", [])
            doc_keyword_counts = Counter(doc_keywords)
            
            row = {"Document": filename}
            for keyword in unique_keywords:
                row[keyword] = doc_keyword_counts.get(keyword, 0)
            
            keyword_matrix.append(row)
        
        if keyword_matrix:
            matrix_df = pd.DataFrame(keyword_matrix)
            
            # Melt the dataframe for visualization
            melted_df = pd.melt(
                matrix_df, 
                id_vars=["Document"], 
                value_vars=[k for k in matrix_df.columns if k != "Document"],
                var_name="Keyword", 
                value_name="Presence"
            )
            
            # Filter to only show keywords that appear in documents
            melted_df = melted_df[melted_df["Presence"] > 0]
            
            if not melted_df.empty:
                heatmap = alt.Chart(melted_df).mark_rect().encode(
                    x=alt.X('Document:N', title=None),
                    y=alt.Y('Keyword:N', title=None),
                    color=alt.Color('Presence:Q', scale=alt.Scale(scheme='blues')),
                    tooltip=['Document', 'Keyword', 'Presence']
                ).properties(
                    width=min(600, len(document_analysis) * 100)
                )
                
                st.altair_chart(heatmap, use_container_width=True)

def render_readability_metrics(document_analysis: Dict[str, Dict[str, Any]]):
    """Render readability metrics for documents."""
    if not document_analysis:
        return
    
    st.subheader("📖 Readability Analysis")
    
    # Extract readability metrics
    metrics = []
    for filename, analysis in document_analysis.items():
        text_analysis = analysis.get("text_analysis", {})
        
        metrics.append({
            "Document": filename,
            "Avg. Sentence Length": text_analysis.get("avg_sentence_length", 0),
            "Reading Time (min)": text_analysis.get("reading_time_minutes", 0)
        })
    
    if metrics:
        metrics_df = pd.DataFrame(metrics)
        
        # Create readability chart
        chart = alt.Chart(metrics_df).mark_circle(size=100).encode(
            x=alt.X('Avg. Sentence Length:Q', scale=alt.Scale(zero=False)),
            y=alt.Y('Reading Time (min):Q', scale=alt.Scale(zero=False)),
            color='Document:N',
            tooltip=['Document', 'Avg. Sentence Length', 'Reading Time (min)']
        ).properties(
            title="Document Complexity",
            height=300
        )
        
        # Add text labels
        text = chart.mark_text(
            align='left',
            baseline='middle',
            dx=15
        ).encode(
            text='Document:N'
        )
        
        st.altair_chart(chart + text, use_container_width=True)
        
        # Add interpretation
        st.info("""
        **Reading this chart:**
        - **Higher on Y-axis**: Longer reading time (more content)
        - **Higher on X-axis**: Longer average sentences (potentially more complex)
        """)

def render_document_dashboard(document_analysis: Dict[str, Dict[str, Any]]):
    """Main function to render the document dashboard."""
    if not document_analysis:
        st.info("No document analysis available. Please process documents first.")
        return
    
    dashboard_tabs = st.tabs(["Statistics", "Content Insights", "Readability"])
    
    with dashboard_tabs[0]:
        render_document_stats(document_analysis)
        
    with dashboard_tabs[1]:
        render_content_insights(document_analysis)
        
    with dashboard_tabs[2]:
        render_readability_metrics(document_analysis)
