import streamlit as st
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- Load and preprocess dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("phishing_site_urls.csv")
    # Use 'Label' column for phishing indicator
    st.write("### Label Distribution:")
    st.write(df['Label'].value_counts())
    df['target'] = (df['Label'] == 'bad').astype(int)  # 1=phishing, 0=legitimate
    return df

# --- Feature extraction for URLs ---
def extract_features_from_url(url):
    features = {}
    parsed = urlparse(url)
    features["length_url"] = len(url)
    features["length_hostname"] = len(parsed.netloc)
    features["ip"] = 1 if re.search(r"\d+\.\d+\.\d+\.\d+", url) else 0
    features["nb_dots"] = url.count('.')
    features["nb_hyphens"] = url.count('-')
    features["nb_at"] = url.count('@')
    features["nb_qm"] = url.count('?')
    features["nb_and"] = url.count('&')
    features["nb_or"] = url.count('|')
    features["nb_eq"] = url.count('=')
    features["nb_underscore"] = url.count('_')
    features["nb_tilde"] = url.count('~')
    features["nb_percent"] = url.count('%')
    features["nb_slash"] = url.count('/')
    features["nb_star"] = url.count('*')
    features["nb_colon"] = url.count(':')
    features["nb_comma"] = url.count(',')
    features["nb_semicolumn"] = url.count(';')
    features["nb_dollar"] = url.count('$')
    features["nb_space"] = url.count(' ')
    features["nb_www"] = url.lower().count('www')
    features["nb_com"] = url.lower().count('com')
    features["nb_dslash"] = url.count('//')
    features["http_in_path"] = 1 if 'http' in parsed.path else 0
    features["https_token"] = 1 if 'https' in url.lower() else 0
    features["ratio_digits_url"] = sum(c.isdigit() for c in url) / len(url)
    features["ratio_digits_host"] = sum(c.isdigit() for c in parsed.netloc) / (len(parsed.netloc) or 1)
    features["punycode"] = 1 if 'xn--' in url else 0
    features["port"] = 1 if parsed.port else 0
    features["tld_in_path"] = 1 if re.search(r"\.[a-z]{2,6}(/|$)", parsed.path) else 0
    features["tld_in_subdomain"] = 1 if re.search(r"\.[a-z]{2,6}\.", parsed.netloc) else 0
    features["abnormal_subdomain"] = 1 if len(parsed.netloc.split('.')) > 3 else 0
    features["nb_subdomains"] = len(parsed.netloc.split('.')) - 2 if len(parsed.netloc.split('.')) > 2 else 0
    features["prefix_suffix"] = 1 if '-' in parsed.netloc else 0
    return features

# --- Prepare features for all URLs ---
def build_feature_dataframe(df):
    feature_list = []
    for url in df['URL']:
        feature_list.append(extract_features_from_url(url))
    feature_df = pd.DataFrame(feature_list)
    return feature_df

# --- Main App ---
st.set_page_config(page_title="Phishing URL Detector", layout="wide", page_icon=":shield:")
st.title(":shield: Phishing URL Detector")
st.markdown("""
This app uses a machine learning model trained on a phishing URL dataset to detect if a link is phishing or legitimate. Enter a URL below to analyze it.
""")

with st.spinner("Loading data and training model..."):
    df = load_data()
    X = build_feature_dataframe(df)
    y = df['target']
    feature_cols = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

st.success(f"Model trained with accuracy: {acc:.2%}")

with st.sidebar:
    st.header("Dataset Preview")
    st.dataframe(df.head(10))
    st.header("Feature Importance")
    importances = model.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)
    st.bar_chart(imp_df.set_index("feature"))

st.subheader("Check a URL")
url_input = st.text_input("Enter a URL to check:", "https://example.com")

if st.button("Analyze URL"):
    with st.spinner("Analyzing..."):
        feats = extract_features_from_url(url_input)
        input_feats = [feats.get(col, 0) for col in feature_cols]
        input_df = pd.DataFrame([input_feats], columns=feature_cols)
        proba = model.predict_proba(input_df)[0]
        pred = model.predict(input_df)[0]
        st.markdown(f"### Result: {'🛑 Phishing' if pred == 1 else '✅ Legitimate'}")
        st.progress(proba[1] if pred == 1 else proba[0])
        st.write(f"**Phishing Probability:** {proba[1]:.2%}")
        st.write(f"**Legitimate Probability:** {proba[0]:.2%}")
        st.write("---")
        st.write("#### Feature Analysis:")
        feat_imp = pd.DataFrame({"Feature": feature_cols, "Value": input_feats, "Importance": importances})
        st.dataframe(feat_imp.sort_values("Importance", ascending=False))
        st.write("#### Model Explanation:")
        st.write("Top features influencing this prediction:")
        for i, row in feat_imp.sort_values("Importance", ascending=False).head(5).iterrows():
            st.write(f"- {row['Feature']}: Value={row['Value']}, Importance={row['Importance']:.3f}")
        st.write("#### Security Recommendations:")
        if pred == 1:
            st.error("Do not enter any personal information or credentials on this site. Report to your security team.")
        else:
            st.success("This link appears legitimate, but always verify before entering sensitive information.")

with st.expander("Show Model Evaluation on Test Set"):
    st.text(classification_report(y_test, y_pred, target_names=['legitimate','phishing']))