# summarize_comments.py
# Step 1: Import libraries
import pandas as pd
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Step 2: Load the data with sentiments
print("📂 Loading comments with sentiment data...")
df = pd.read_csv('comments_with_sentiment.csv')
print(f"Loaded {len(df)} comments.")

# Step 3: Initialize the Sumy summarizer
# We'll use the LSA (Latent Semantic Analysis) method for extractive summarization
print("🔧 Initializing summarizer...")
stemmer = Stemmer("english")
summarizer = LsaSummarizer(stemmer)
summarizer.stop_words = get_stop_words("english")

# Step 4: Create a function to summarize a single comment
def summarize_comment(text, sentences_count=1):
    """
    Uses Sumy's LSA summarizer to extract the most important sentence(s)
    from the provided text.
    """
    # Check for very short comments to avoid errors
    if len(text.strip().split()) < 10:
        return text # Return the short text as its own summary
    
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summary_sentences = summarizer(parser.document, sentences_count)
        summary = " ".join(str(sentence) for sentence in summary_sentences)
        return summary
    except Exception as e:
        # If anything goes wrong, return a placeholder
        print(f"Could not summarize comment: {e}")
        return "Summary not available."

# Step 5: Apply the summarization to each comment
print("⚙️ Generating summaries for all comments (this may take a moment)...")
# We use .progress_apply for a nice progress bar if you have tqdm installed.
# If you don't have tqdm, just use: df['summary'] = df['Comment'].apply(summarize_comment)
try:
    from tqdm import tqdm
    tqdm.pandas() # Enable progress_apply
    df['summary'] = df['Comment'].progress_apply(summarize_comment)
except ImportError:
    print("(Tip: Install 'tqdm' with 'pip install tqdm' for a progress bar)")
    df['summary'] = df['Comment'].apply(summarize_comment)

# Step 6: Save the final, enriched dataset
final_filename = 'final_analyzed_comments.csv'
df.to_csv(final_filename, index=False)
print(f"💾 Final analyzed data saved to '{final_filename}'")

# Step 7: Let's see a preview of the original comment vs. the summary
print("\n🔍 Preview of original comments and their AI-generated summaries:")
for i, row in df.head(5).iterrows():
    print(f"\nComment {i+1} ({row['sentiment']}):")
    print(f"ORIGINAL: {row['Comment'][:100]}...") # Show first 100 chars
    print(f"SUMMARY:  {row['summary']}")
    print("-" * 50)
