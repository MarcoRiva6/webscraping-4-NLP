import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import pandas as pd
from langdetect import detect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator

import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

def perform_analysis(main_path, sources_path):
    initial_datasets = []
    processed_datasets = []

    source_template = {'name': None, 'data': None, 'filename': None}

    # Load the data
    sources = os.listdir(sources_path)
    for source in sources:
        source_data = pd.read_csv(f"{sources_path}/{source}")
        source_template['name'] = source.split('_')[0]
        source_template['data'] = source_data
        source_template['filename'] = f"{source}.csv"
        initial_datasets.append(source_template.copy())

    # Initialize tools
    vader_analyzer = SentimentIntensityAnalyzer()
    translator = Translator()

    def detect_language(text):
        try:
            return detect(text)
        except:
            return "unknown"

    def translate_to_english(text, lang):
        try:
            if lang == 'en':
                return text  # No translation needed
            translated = translator.translate(text, src=lang, dest='en')
            return translated.text
        except Exception as e:
            print(f"Error translating text: {text}. Error: {e}")
            return None

    # Analyze sentiment using VADER
    def analyze_sentiment(text):
        sentiment = vader_analyzer.polarity_scores(text)
        return sentiment['compound']

    # Process the dataframe
    def process_reviews(dataframe, source_name):
        if 'language' not in dataframe.columns or len(dataframe['language'].isnull()) > 0:
            dataframe['language'] = dataframe['text'].apply(detect_language)
        dataframe['translated_text'] = dataframe.apply(
            lambda row: translate_to_english(row['text'], row['language']), axis=1
        )
        dataframe['sentiment_score'] = dataframe['translated_text'].apply(analyze_sentiment)
        dataframe.drop(columns=['translated_text'], inplace=True)  # Remove translated_text from final output
        return dataframe

    processed_datasets = []
    for dataset in initial_datasets:
        if dataset['name'] in [source['name'] for source in processed_datasets]: # Skip already loaded datasets
            continue
        final_filename = f"{dataset['name']}_reviews_with_sentiment.csv"
        processed_data = process_reviews(dataset['data'], dataset['name']) if not os.path.exists(main_path + final_filename) else pd.read_csv(main_path + final_filename)

        source_template['name'] = dataset['name']
        source_template['data'] = processed_data
        source_template['filename'] = final_filename
        processed_datasets.append(source_template.copy())

    return processed_datasets

    print("Sentiment analysis complete")

def plot_graphs(processed_datasets):
    for source in processed_datasets:
        source['data']['source'] = source['name']

    # Combine datasets
    combined_reviews = pd.concat([source['data'] for source in processed_datasets], ignore_index=True)

    # Categorize sentiment
    def categorize_sentiment(score):
        if score > 0.5:
            return 'Positive'
        elif score < -0.5:
            return 'Negative'
        else:
            return 'Neutral'

    combined_reviews['sentiment_category'] = combined_reviews['sentiment_score'].apply(categorize_sentiment)

    # Overall Distribution of Sentiment Scores
    st.title("Overall Distribution of Sentiment Scores")
    st.markdown("This graph shows the combined distribution of sentiment scores across all sources.")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(
        data=combined_reviews,
        x='sentiment_score',
        hue='source',
        kde=True,
        bins=50,
        palette="viridis",
        alpha=0.7,
        ax=ax
    )
    ax.set_title("Distribution of Sentiment Scores (Overall)")
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Sentiment Scores for Individual Sources
    for i, source in enumerate(processed_datasets):
        st.title(f"Sentiment Scores for {source['name']}")
        st.markdown(f"This graph shows the distribution of sentiment scores specifically for the source {source['name']}.")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(
            data=source['data'],
            x='sentiment_score',
            kde=True,
            bins=50,
            alpha=0.7,
            ax=ax
        )
        ax.set_title(f"Sentiment Scores ({source['name']})")
        ax.set_xlabel("Sentiment Score")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    # Violin Plot of Sentiment Scores by Source
    st.title("Violin Plot of Sentiment Scores by Source")
    st.markdown("This graph shows a violin plot of sentiment scores for each source, displaying the distribution and quartiles.")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(
        data=combined_reviews,
        x='source',
        y='sentiment_score',
        hue='source',
        inner="quartile",
        palette="pastel",
        ax=ax
    )
    ax.set_title("Violin Plot of Sentiment Scores by Source")
    ax.set_xlabel("Source")
    ax.set_ylabel("Sentiment Score")
    ax.legend([], [], frameon=False)
    st.pyplot(fig)

    # Density Plot of Sentiment Scores by Source
    st.title("Density Plot of Sentiment Scores by Source")
    st.markdown("This graph shows overlapping density plots for sentiment scores from all sources.")
    fig, ax = plt.subplots(figsize=(12, 6))
    for source in processed_datasets:
        sns.kdeplot(
            data=source['data'],
            x='sentiment_score',
            fill=True,
            alpha=0.5,
            label=source['name'],
            ax=ax
        )
    ax.set_title("Density Plot of Sentiment Scores by Source")
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Density")
    ax.legend(title="Source", loc="upper left")
    st.pyplot(fig)

    # Sentiment Categories Proportions
    st.title("Sentiment Categories Proportions")
    st.markdown("This section shows the proportions of sentiment categories for each source.")
    source_sentiment_counts = combined_reviews.groupby('source')['sentiment_category'].value_counts(normalize=True).unstack()
    colors = sns.color_palette("pastel")

    for i, source in enumerate(processed_datasets):
        st.subheader(f"Sentiment Categories for {source['name']}")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            source_sentiment_counts.loc[source['name']],
            labels=source_sentiment_counts.loc[source['name']].index,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors,
        )
        ax.set_title(f"Sentiment Categories ({source['name']})")
        st.pyplot(fig)

    st.success("Sentiment analysis completed and visualized!")

# Streamlit app
st.title("Exploratory Sentiment Analysis Project")

st.markdown(
    """
    ## Project Overview
    This app demonstrates sentiment analysis on user reviews from restaurants. 
    It processes data uploaded as CSV files, performs sentiment analysis, and visualizes the results.

    Upload your data to get started!
    """
)

# Create directories if they don't exist
if not os.path.exists('./temp_sources'):
    os.makedirs('./temp_sources')

# File upload
st.markdown("## Upload Your Data")
st.info("Upload CSV files containing reviews from different sources.")
uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    # Save uploaded files to temp_sources
    for uploaded_file in uploaded_files:
        with open(os.path.join('./temp_sources', uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())

    st.success(f"Uploaded {len(uploaded_files)} files successfully!")

    # Perform sentiment analysis
    st.write("Running sentiment analysis pipeline...")
    try:
        processed_datasets = perform_analysis(main_path='temp_output', sources_path='temp_sources')
        st.success("Sentiment analysis pipeline executed successfully!")
    except Exception as e:
        st.error("Error executing the sentiment analysis pipeline.")
        print(e)
        st.stop()

    # Load processed datasets
    st.markdown("## Sentiment Analysis Results")

    plot_graphs(processed_datasets)

    st.success("Sentiment analysis completed and visualized!")

# Clean up directories
if os.path.exists('./temp_sources'):
    shutil.rmtree('./temp_sources')

