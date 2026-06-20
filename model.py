import pandas as pd
import numpy as np
import re
import nltk
import shap
import warnings
warnings.filterwarnings('ignore')

from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemma = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# ============================================================
# TEXT CLEANING
# ============================================================
def clean(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|@\w+|#\w+|[^a-z\s]', '', text)
    tokens = [lemma.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(tokens)


# ============================================================
# EMOTION DETECTION (Lexicon Based - NRC inspired, with fallback)
# ============================================================
EMOTION_LEXICON = {
    'Anger':    ['hate','kill','stupid','idiot','dumb','angry','furious','rage','ugly','shut',
                  'loser','trash','pathetic','worthless','disappear','bad','horrible','awful',
                  'annoying','terrible','mad'],
    'Sadness':  ['sad','cry','depressed','lonely','hurt','pain','upset','hopeless','tears','miserable'],
    'Fear':     ['scared','afraid','threat','die','danger','worried','terrorist','attack'],
    'Joy':      ['happy','great','good','love','nice','wonderful','awesome','amazing','best',
                  'fun','smile','enjoy','positive','kind'],
    'Disgust':  ['disgusting','gross','nasty','sick','vomit','filthy','ew'],
    'Surprise': ['wow','omg','shock','unbelievable','surprised']
}

def detect_emotion(text, polarity=0):
    text = text.lower()
    scores = {emo: 0 for emo in EMOTION_LEXICON}
    for emo, words in EMOTION_LEXICON.items():
        for w in words:
            if re.search(r'\b' + w + r'\b', text):
                scores[emo] += 1

    if max(scores.values()) == 0:
        # Fallback: use sentiment polarity to guess emotion
        if polarity < -0.2:
            return 'Anger', scores
        elif polarity > 0.2:
            return 'Joy', scores
        return 'Neutral', scores

    top = max(scores, key=scores.get)
    return top, scores


# ============================================================
# SENTIMENT ANALYSIS (TextBlob)
# ============================================================
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        label = 'Positive'
    elif polarity < -0.1:
        label = 'Negative'
    else:
        label = 'Neutral'
    return label, round(polarity, 3)


# ============================================================
# SEVERITY PREDICTION
# ============================================================
def get_severity(pred_label, confidence, emotion, polarity):
    if pred_label == 'not_cyberbullying':
        return 'None'

    score = confidence
    if emotion in ['Anger', 'Disgust', 'Fear']:
        score += 10
    if polarity < -0.3:
        score += 10

    if score >= 85:
        return 'High'
    elif score >= 70:
        return 'Medium'
    else:
        return 'Low'


# ============================================================
# MODEL TRAINING
# ============================================================
def train_models():
    df = pd.read_csv("cyberbullying_tweets.csv")
    df['clean'] = df['tweet_text'].apply(clean)

    le = LabelEncoder()
    y = le.fit_transform(df['cyberbullying_type'])

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean'], y, test_size=0.2, random_state=42
    )

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tr = tfidf.fit_transform(X_train)
    X_te = tfidf.transform(X_test)

    candidates = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=5),
        'Random Forest': RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42)
    }

    accuracies = {}
    reports = {}
    for name, m in candidates.items():
        m.fit(X_tr, y_train)
        y_pred = m.predict(X_te)
        accuracies[name] = round(accuracy_score(y_test, y_pred) * 100, 2)
        reports[name] = classification_report(
            y_test, y_pred, target_names=le.classes_, output_dict=True
        )

    # Deployed model = Logistic Regression (interpretable + SHAP compatible)
    deployed = candidates['Logistic Regression']
    y_pred_best = deployed.predict(X_te)
    cm = confusion_matrix(y_test, y_pred_best)

    bg_sample = X_tr[:100].toarray()

    return {
        'model': deployed,
        'tfidf': tfidf,
        'le': le,
        'accuracies': accuracies,
        'reports': reports,
        'df': df,
        'cm': cm,
        'bg_sample': bg_sample
    }


# ============================================================
# FULL ANALYSIS PIPELINE
# ============================================================
def analyze(text, art):
    model = art['model']
    tfidf = art['tfidf']
    le = art['le']

    cleaned = clean(text)
    vec = tfidf.transform([cleaned])

    pred_idx = model.predict(vec)[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    proba = model.predict_proba(vec)[0]
    confidence = round(max(proba) * 100, 2)

    sentiment, polarity = get_sentiment(text)
    emotion, emotion_scores = detect_emotion(text, polarity)
    severity = get_severity(pred_label, confidence, emotion, polarity)
    calibrated = False
    if pred_label != 'not_cyberbullying' and polarity > 0.3 and confidence < 75:
        pred_label = 'not_cyberbullying'
        severity = 'None'
        calibrated = True

    return {
        'cleaned': cleaned,
        'prediction': pred_label,
        'confidence': confidence,
        'proba': proba,
        'sentiment': sentiment,
        'polarity': polarity,
        'emotion': emotion,
        'emotion_scores': emotion_scores,
        'severity': severity,
        'vec': vec,
        'pred_idx': pred_idx ,
        'calibrated': calibrated 
    }


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================
def explain_prediction(result, art, top_n=8):
    model = art['model']
    tfidf = art['tfidf']
    vec = result['vec']
    pred_idx = result['pred_idx']
    feature_names = np.array(tfidf.get_feature_names_out())

    try:
        explainer = shap.LinearExplainer(model, art['bg_sample'])
        vec_dense = vec.toarray()
        shap_values = explainer.shap_values(vec_dense)

        if isinstance(shap_values, list):
            values = shap_values[pred_idx][0]
        elif shap_values.ndim == 3:
            values = shap_values[0, :, pred_idx]
        else:
            values = shap_values[0]

        nonzero_idx = vec.nonzero()[1]
        contributions = [(feature_names[i], float(values[i])) for i in nonzero_idx]
        contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:top_n]
        return contributions, "SHAP"

    except Exception:
        coefs = model.coef_[pred_idx]
        vec_arr = vec.toarray()[0]
        nonzero_idx = vec.nonzero()[1]
        contributions = [(feature_names[i], float(vec_arr[i] * coefs[i])) for i in nonzero_idx]
        contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:top_n]
        return contributions, "Feature Importance"