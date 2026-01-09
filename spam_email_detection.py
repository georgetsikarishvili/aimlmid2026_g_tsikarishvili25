import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# --- 1. Load Data ---
filename = 'g_tsikarishvili25_93455.csv'
try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"Error: File '{filename}' not found. Please check the file path.")
    exit()

# --- 2. Train Logistic Regression Model ---
# Define Features and Target
X = df[['words', 'links', 'capital_words', 'spam_word_count']]
y = df['is_spam']

# Split Data (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Output Coefficients (Required for Report)
coeffs = dict(zip(X.columns, model.coef_[0]))
intercept = model.intercept_[0]
print("--- Model Parameters ---")
print(f"Intercept: {intercept:.4f}")
print(f"Coefficients: {coeffs}")

# --- 3. Validation (Confusion Matrix & Accuracy) ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n--- Validation Results ---")
print(f"Accuracy: {acc * 100:.2f}%")
print("Confusion Matrix:\n", cm)

# --- 4. Application: Email Checker ---
# Helper list for simulation (keywords to count for 'spam_word_count')
spam_keywords = ["free", "win", "prize", "money", "offer", "click", "buy", "urgent", "cash", "winner"]


def check_email(text):
    words_list = text.split()

    # Extract features
    f_words = len(words_list)
    f_links = text.count('http') + text.count('www')
    f_caps = sum(1 for w in words_list if w.isupper() and len(w) > 1)
    f_spam = sum(1 for w in words_list if w.lower() in spam_keywords)

    # Create a DataFrame with column names to silence the Sklearn warning
    features = pd.DataFrame(
        [[f_words, f_links, f_caps, f_spam]],
        columns=['words', 'links', 'capital_words', 'spam_word_count']
    )

    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    return "SPAM" if prediction == 1 else "LEGITIMATE", prob


# --- 5 & 6. Manual Testing ---
print("\n--- Manual Email Testing ---")

# Spam Case
spam_text = "URGENT! WINNER!! You have WON $1000000 FREE CASH! CLICK HERE: http://win.com http://claim.net http://prize.org. BUY NOW!"
res_spam, prob_spam = check_email(spam_text)
print(f"Email 1 (Spam): Classified as {res_spam} (Probability: {prob_spam:.4f})")

# Legitimate Case
legit_text = "Hi Sarah, I hope you are doing well. Please find the attached meeting notes. Best, John."
res_legit, prob_legit = check_email(legit_text)
print(f"Email 2 (Legit): Classified as {res_legit} (Probability: {prob_legit:.4f})")

# --- 7. Visualizations ---
# Plot A: Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Spam'], yticklabels=['Legitimate', 'Spam'])
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\nSaved visualization: confusion_matrix.png")

# Plot B: Feature Coefficients (Feature Importance)
plt.figure(figsize=(8, 5))

# FIX: Added 'hue' and 'legend=False' to silence the Seaborn warning
sns.barplot(
    x=list(coeffs.keys()),
    y=list(coeffs.values()),
    hue=list(coeffs.keys()),
    legend=False,
    palette='viridis'
)

plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.ylabel('Coefficient Value')
plt.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Saved visualization: feature_importance.png")