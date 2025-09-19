# generate_wordcloud.py
# Step 1: Import necessary libraries
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
import re

# Step 2: Load the data with sentiments
print("📂 Loading analyzed comments...")
df = pd.read_csv('comments_with_sentiment.csv')
print(f"Loaded {len(df)} comments.")

# Step 3: Combine all comments into one big text
all_text = ' '.join(df['Comment'].astype(str).tolist())
print("Combined all text for processing.")

# Step 4: Clean and preprocess the text
# Convert to lowercase and split into words
words = re.findall(r'\w+', all_text.lower())

# Define stopwords - common words to exclude
stopwords = set(STOPWORDS)
# Add custom stopwords specific to this legal/consultation context
custom_stopwords = {
    "draft", "amendment", "law", "rule", "rules", "process", "implementation",
    "new", "will", "may", "much", "good", "need", "needs", "required", "us", "it", "its"
}
all_stopwords = stopwords.union(custom_stopwords)

# Filter out stopwords and short words
filtered_words = [word for word in words if word not in all_stopwords and len(word) > 2]
print(f"Filtered words. Now working with {len(filtered_words)} significant terms.")

# Step 5: Generate the Word Cloud
print("🎨 Generating word cloud...")
wordcloud = WordCloud(
    width=1200, 
    height=800, 
    background_color='white',
    stopwords=all_stopwords,
    colormap='viridis', # You can change this: try 'plasma', 'inferno', 'magma'
    max_words=100
).generate(' '.join(filtered_words))

# Step 6: Display and save the word cloud
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Turn off axis lines and labels
plt.title('Most Frequent Words in eConsultation Comments', fontsize=16, pad=20)
plt.tight_layout(pad=0)

# Save the image for your dashboard
wordcloud_image_path = 'wordcloud.png'
plt.savefig(wordcloud_image_path, dpi=300, bbox_inches='tight')
print(f"💾 Word cloud saved as '{wordcloud_image_path}'")

# (Optional) Show the most common words as text too
print("\n📊 Top 10 Most Frequent Words:")
word_freq = Counter(filtered_words)
for word, count in word_freq.most_common(10):
    print(f"{word}: {count}")

print("\n✅ Word cloud generation complete!")
