# sentiment_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud

# Load your review data
# Replace 'ethiopian_banks_reviews.csv' with your actual file
df = pd.read_csv('ethiopian_banks_reviews.csv')

# Ensure necessary columns exist
if not {'app_name', 'review'}.issubset(df.columns):
    raise ValueError("CSV file must contain 'app_name' and 'review' columns")

# Drop rows with missing reviews
df = df.dropna(subset=['review'])

# Sentiment scoring
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['sentiment_score'] = df['review'].apply(get_sentiment)

# Classify sentiment
def classify_sentiment(score):
    if score > 0.1:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['sentiment_score'].apply(classify_sentiment)

# Show sentiment distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='app_name', hue='sentiment', palette='Set2')
plt.title('Sentiment Distribution by App')
plt.xlabel('App')
plt.ylabel('Number of Reviews')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.savefig('sentiment_distribution.png')
plt.show()

# Generate word cloud
all_reviews = ' '.join(df['review'].dropna())

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Reviews')
plt.tight_layout()
plt.savefig('wordcloud_reviews.png')
plt.show()

# Save processed data
df.to_csv('processed_reviews_with_sentiment.csv', index=False)
print("Analysis complete. Sentiment data saved to 'processed_reviews_with_sentiment.csv'")
