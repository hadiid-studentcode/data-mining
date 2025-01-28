import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Read the results
df = pd.read_csv('hasil_sentimen_naive_bayes_optimized.csv')

# Create figure with subplots
plt.figure(figsize=(15, 6))

# 1. Pie Chart for Sentiment Distribution
plt.subplot(1, 2, 1)
sentiment_counts = df['naive_bayes_prediction'].value_counts()
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
plt.title('Distribution of Sentiments')

# 2. Word Cloud for each sentiment
plt.subplot(1, 2, 2)
text_data = " ".join(df['text'])
wordcloud = WordCloud(
    width=800, 
    height=400,
    background_color='white',
    max_words=100
).generate(text_data)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of All Text')

# Adjust layout and save
plt.tight_layout()
plt.savefig('sentiment_analysis_visualization.png')
plt.show()

# Create separate wordclouds for each sentiment
plt.figure(figsize=(10, 5))  # Adjusted figure size for 2 plots instead of 3
for i, sentiment in enumerate(['Positif', 'Negatif'], 1):
    plt.subplot(1, 2, i)  # Changed to 1, 2 for two subplots
    text_data = " ".join(df[df['naive_bayes_prediction'] == sentiment]['text'])
    if text_data.strip():  # Check if there's any text
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=50
        ).generate(text_data)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud - {sentiment} Sentiments')

plt.tight_layout()
plt.savefig('sentiment_specific_wordclouds.png')
plt.show()

# Print summary statistics
print("\nSentiment Distribution:")
print(sentiment_counts)
print("\nPercentage Distribution:")
print((sentiment_counts/len(df)*100).round(2), "%")
