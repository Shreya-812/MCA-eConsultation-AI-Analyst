# sentiment_analysis.py
# Step 1: Import the necessary tools
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Step 2: Load our sample data
print("📂 Loading data...")
# NOTICE THE CHANGE: We use 'Comment' instead of 'comment' to match your CSV header.
df = pd.read_csv('data.csv')
print(f"Found {len(df)} comments to analyze!")
print(df.head()) # Let's peek at the first few rows

# Step 3: Download the VADER lexicon (it's a one-time download)
print("\n⬇️ Downloading VADER lexicon (if needed)...")
import nltk
nltk.download('vader_lexicon')

# Step 4: Initialize the sentiment analyzer
print("🔧 Initializing sentiment analyzer...")
sia = SentimentIntensityAnalyzer()

# Step 5: Let's test it on a single, hardcoded comment first
test_comment = "This new rule is fantastic for transparency!"
scores = sia.polarity_scores(test_comment)
print(f"\n🧪 TEST SENTIMENT for '{test_comment}':")
print(scores)

# Step 6: Create a function to get the sentiment category
def get_sentiment_label(compound_score):
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Step 7: Apply the sentiment analysis to every row in our DataFrame
print("\n⚙️ Analyzing sentiments for all comments...")
# NOTICE THE CHANGE: We use df['Comment'] now.
df['sentiment_scores'] = df['Comment'].apply(lambda text: sia.polarity_scores(str(text)))
df['compound_score'] = df['sentiment_scores'].apply(lambda scores: scores['compound'])
df['sentiment'] = df['compound_score'].apply(get_sentiment_label)

# Step 8: Show the results!
print("✅ Analysis complete! Here are the results:")
print(df[['Comment', 'sentiment', 'compound_score']].head(10))

# Optional: Save this results to a new CSV for later use
df.to_csv('comments_with_sentiment.csv', index=False)
print("\n💾 Results saved to 'comments_with_sentiment.csv'")

# BONUS: Let's see a quick count of sentiments
print("\n📊 Sentiment Distribution:")
print(df['sentiment'].value_counts())
