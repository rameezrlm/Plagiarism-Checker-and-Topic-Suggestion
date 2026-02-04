from flask import Flask,request,render_template
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
df = pd.read_csv('essays.csv')

def suggest_topic(text):
    model = joblib.load('model.pkl')
    res = model.predict([text])
    return res[0]


@app.route('/')
def home():
    return render_template('index.html')
def check_plagiarism(user_input, dataset_texts, threshold=0.2):
    all_texts = dataset_texts.tolist() + [user_input]

    vectorizer = TfidfVectorizer().fit(all_texts)
    tfidf_matrix = vectorizer.transform(all_texts)

    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    max_sim = max(cosine_similarities)
   
    percentage = round(max_sim * 100, 1)
    if max_sim >= threshold:
        return percentage
    else:
        return percentage

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/plagiarism', methods=['POST'])
def plagiarism_check():
    user_text = request.form.get("text", "")
    if not user_text.strip():
        return "Please enter some text."

    result = check_plagiarism(user_text, df['text'])
    return render_template('plagiarism.html',similarity = result ,topic=suggest_topic(user_text),text = user_text)

if __name__ == "__main__":
    app.run(debug=True)