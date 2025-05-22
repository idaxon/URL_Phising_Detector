import pandas as pd
import numpy as np
import re
import tkinter as tk
from tkinter import ttk, messagebox, font
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ttkthemes

# --- Feature Extraction Functions ---

def extract_features(url):
    features = {}
    
    # URL length
    features['url_length'] = len(url)
    
    # Domain length
    domain = urlparse(url).netloc
    features['domain_length'] = len(domain)
    
    # Number of dots in URL
    features['dots_count'] = url.count('.')
    
    # Number of special characters
    features['special_chars'] = len(re.findall(r'[^a-zA-Z0-9.]', url))
    
    # Contains IP address
    features['has_ip'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0
    
    # Contains suspicious words
    suspicious_words = ['secure', 'account', 'update', 'banking', 'login', 'verify']
    features['suspicious_words'] = sum(1 for word in suspicious_words if word in url.lower())
    
    # Uses HTTPS
    features['is_https'] = 1 if url.startswith('https') else 0
    
    # URL has @ symbol
    features['has_at_symbol'] = 1 if '@' in url else 0
    
    # URL has double slash redirect
    features['has_double_slash'] = 1 if '//' in url[8:] else 0
    
    # URL has prefix/suffix
    features['has_prefix_suffix'] = 1 if '-' in domain else 0
    
    return features

# --- Dataset Creation ---

def create_dataset():
    # Create a synthetic dataset for demonstration
    # In a real scenario, you would use an actual phishing dataset
    
    # Legitimate URLs
    legitimate_urls = [
        'https://www.google.com',
        'https://www.youtube.com',
        'https://www.facebook.com',
        'https://www.amazon.com',
        'https://www.wikipedia.org',
        'https://www.twitter.com',
        'https://www.instagram.com',
        'https://www.linkedin.com',
        'https://www.github.com',
        'https://www.apple.com'
    ]
    
    # Phishing URLs (synthetic examples)
    phishing_urls = [
        'http://googlee-verify.com/account',
        'http://secure-paypal.com.verify.info/login',
        'http://banking.secure-login.com@192.168.1.1',
        'http://facebook-verify.com/login.php',
        'http://account-update-required.com',
        'http://verification-account.com/secure',
        'http://apple.com-verify-account.com',
        'http://secure-banking-login.com//update',
        'http://amazon.com.secure-payment.info',
        'http://verify-your-account-now.com'
    ]
    
    # Create dataframe
    urls = legitimate_urls + phishing_urls
    labels = [0] * len(legitimate_urls) + [1] * len(phishing_urls)
    
    data = {'url': urls, 'is_phishing': labels}
    df = pd.DataFrame(data)
    
    # Extract features
    features_list = []
    for url in urls:
        features = extract_features(url)
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    # Combine features with labels
    final_df = pd.concat([features_df, pd.Series(labels, name='is_phishing')], axis=1)
    
    return final_df, urls, labels

# --- Model Training ---

def train_model(df):
    # Separate features and target
    X = df.drop('is_phishing', axis=1)
    y = df['is_phishing']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    
    return model

# --- Prediction Function ---

def predict_phishing(url, model):
    # Extract features from URL
    features = extract_features(url)
    features_df = pd.DataFrame([features])
    
    # Make prediction
    prediction_proba = model.predict_proba(features_df)[0]
    is_phishing = prediction_proba[1] > 0.5
    confidence = prediction_proba[1] if is_phishing else prediction_proba[0]
    
    # Get feature importance
    feature_importance = dict(zip(features.keys(), model.feature_importances_))
    
    # Analyze specific risk factors
    risk_factors = []
    if features['has_ip'] == 1:
        risk_factors.append("Contains IP address in URL")
    if features['has_at_symbol'] == 1:
        risk_factors.append("Contains @ symbol in URL")
    if features['has_double_slash'] == 1:
        risk_factors.append("Contains double slash redirect")
    if features['is_https'] == 0:
        risk_factors.append("Not using HTTPS protocol")
    if features['suspicious_words'] > 0:
        risk_factors.append(f"Contains {features['suspicious_words']} suspicious keywords")
    if features['has_prefix_suffix'] == 1:
        risk_factors.append("Contains hyphens in domain name")
    if features['url_length'] > 75:
        risk_factors.append("Unusually long URL")
    
    # Generate security recommendations
    recommendations = []
    if is_phishing:
        recommendations = [
            "Do not enter any personal information on this website",
            "Do not download any files from this website",
            "Do not click on any links within this website",
            "Report this URL to your organization's security team"
        ]
    else:
        recommendations = [
            "Always verify the website's identity before entering sensitive information",
            "Check for HTTPS and valid certificates",
            "Be cautious of unexpected requests for personal information"
        ]
    
    return is_phishing, confidence, features, feature_importance, risk_factors, recommendations

# --- UI Class ---

class PhishingDetectorApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        
        # Apply theme
        self.style = ttkthemes.ThemedStyle(root)
        self.style.set_theme("arc")
        
        self.root.title("Advanced Phishing Link Detector")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Custom fonts
        self.title_font = font.Font(family="Helvetica", size=16, weight="bold")
        self.header_font = font.Font(family="Helvetica", size=12, weight="bold")
        self.normal_font = font.Font(family="Helvetica", size=10)
        
        # Main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Advanced Phishing URL Analyzer", font=self.title_font)
        title_label.pack(pady=(0, 20))
        
        # URL Entry Frame
        url_frame = ttk.Frame(main_frame)
        url_frame.pack(fill="x", pady=10)
        
        ttk.Label(url_frame, text="Enter URL:", font=self.normal_font).pack(side="left")
        self.url_entry = ttk.Entry(url_frame, width=60, font=self.normal_font)
        self.url_entry.pack(side="left", padx=5, fill="x", expand=True)
        self.url_entry.focus()
        
        # Check Button with improved styling
        check_button = ttk.Button(url_frame, text="Analyze URL", command=self.check_url, style="Accent.TButton")
        check_button.pack(side="right", padx=5)
        
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="both", expand=True, pady=10)
        
        # Tab 1: Summary Results
        self.summary_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.summary_frame, text="Summary")
        
        # Result Labels in Summary tab
        result_frame = ttk.LabelFrame(self.summary_frame, text="Analysis Result", padding=10)
        result_frame.pack(fill="both", expand=True, pady=5)
        
        # Status with icon
        self.status_frame = ttk.Frame(result_frame)
        self.status_frame.pack(fill="x", pady=5)
        
        self.status_label = ttk.Label(self.status_frame, text="Status: Not checked", font=self.header_font)
        self.status_label.pack(side="left", pady=5)
        
        # Confidence score with circular progress
        self.confidence_frame = ttk.Frame(result_frame)
        self.confidence_frame.pack(fill="x", pady=5)
        
        self.confidence_label = ttk.Label(self.confidence_frame, text="Confidence: N/A", font=self.normal_font)
        self.confidence_label.pack(side="left", pady=5)
        
        # Progress Bar with better styling
        self.confidence_bar = ttk.Progressbar(self.confidence_frame, orient="horizontal", length=400, mode="determinate", style="Horizontal.TProgressbar")
        self.confidence_bar.pack(side="left", padx=10, pady=5)
        
        # Risk assessment
        self.risk_frame = ttk.LabelFrame(result_frame, text="Risk Assessment", padding=10)
        self.risk_frame.pack(fill="both", expand=True, pady=10)
        
        self.risk_text = tk.Text(self.risk_frame, height=5, width=70, font=self.normal_font, wrap="word")
        self.risk_text.pack(fill="both", expand=True)
        self.risk_text.config(state="disabled")
        
        # Recommendations
        self.rec_frame = ttk.LabelFrame(result_frame, text="Security Recommendations", padding=10)
        self.rec_frame.pack(fill="both", expand=True, pady=10)
        
        self.rec_text = tk.Text(self.rec_frame, height=5, width=70, font=self.normal_font, wrap="word")
        self.rec_text.pack(fill="both", expand=True)
        self.rec_text.config(state="disabled")
        
        # Tab 2: Detailed Analysis
        self.detail_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.detail_frame, text="Detailed Analysis")
        
        # Features table
        self.features_frame = ttk.LabelFrame(self.detail_frame, text="URL Features", padding=10)
        self.features_frame.pack(fill="both", expand=True, pady=5)
        
        # Create treeview for features
        self.features_tree = ttk.Treeview(self.features_frame, columns=("Feature", "Value", "Impact"), show="headings", height=10)
        self.features_tree.heading("Feature", text="Feature")
        self.features_tree.heading("Value", text="Value")
        self.features_tree.heading("Impact", text="Impact on Decision")
        self.features_tree.column("Feature", width=200)
        self.features_tree.column("Value", width=100)
        self.features_tree.column("Impact", width=200)
        self.features_tree.pack(fill="both", expand=True, side="left")
        
        # Add scrollbar to treeview
        features_scroll = ttk.Scrollbar(self.features_frame, orient="vertical", command=self.features_tree.yview)
        features_scroll.pack(side="right", fill="y")
        self.features_tree.configure(yscrollcommand=features_scroll.set)
        
        # Tab 3: Visualization
        self.viz_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.viz_frame, text="Visualization")
        
        # Frame for charts
        self.chart_frame = ttk.Frame(self.viz_frame)
        self.chart_frame.pack(fill="both", expand=True)
        
        # Status message at bottom
        self.status_message = ttk.Label(main_frame, text="Ready to analyze URLs", font=self.normal_font)
        self.status_message.pack(pady=10, anchor="w")
    
    def check_url(self):
        url = self.url_entry.get().strip()
        
        if not url:
            messagebox.showerror("Error", "Please enter a URL")
            return
        
        # Add http:// prefix if missing
        if not url.startswith('http'):
            url = 'http://' + url
        
        try:
            # Update status message
            self.status_message.config(text="Analyzing URL...")
            self.root.update()
            
            # Get prediction and analysis
            is_phishing, confidence, features, feature_importance, risk_factors, recommendations = predict_phishing(url, self.model)
            
            # Update Summary tab
            self.update_summary(url, is_phishing, confidence, risk_factors, recommendations)
            
            # Update Details tab
            self.update_details(features, feature_importance)
            
            # Update Visualization tab
            self.update_visualization(features, feature_importance, is_phishing)
            
            # Switch to Summary tab
            self.notebook.select(0)
            
            # Update status message
            self.status_message.config(text=f"Analysis completed for {url}")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_message.config(text="Error during analysis")
    
    def update_summary(self, url, is_phishing, confidence, risk_factors, recommendations):
        # Update status
        status_text = "HIGH RISK - PHISHING DETECTED!" if is_phishing else "LOW RISK - LEGITIMATE URL"
        status_color = "#FF5252" if is_phishing else "#4CAF50"
        
        self.status_label.config(text=f"Status: {status_text}", foreground=status_color)
        self.confidence_label.config(text=f"Confidence: {confidence:.2f}")
        
        # Update progress bar
        self.confidence_bar["value"] = confidence * 100
        
        # Update risk factors
        self.risk_text.config(state="normal")
        self.risk_text.delete(1.0, tk.END)
        
        if risk_factors:
            for factor in risk_factors:
                self.risk_text.insert(tk.END, f"• {factor}\n")
        else:
            self.risk_text.insert(tk.END, "No specific risk factors identified.")
        
        self.risk_text.config(state="disabled")
        
        # Update recommendations
        self.rec_text.config(state="normal")
        self.rec_text.delete(1.0, tk.END)
        
        for rec in recommendations:
            self.rec_text.insert(tk.END, f"• {rec}\n")
        
        self.rec_text.config(state="disabled")
    
    def update_details(self, features, feature_importance):
        # Clear existing items
        for item in self.features_tree.get_children():
            self.features_tree.delete(item)
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Add features to treeview
        for feature_name, importance in sorted_features:
            value = features[feature_name]
            
            # Determine impact text and color based on importance
            if importance > 0.15:
                impact = "High"
                tag = "high"
            elif importance > 0.05:
                impact = "Medium"
                tag = "medium"
            else:
                impact = "Low"
                tag = "low"
                
            self.features_tree.insert("", "end", values=(feature_name, value, f"{impact} ({importance:.3f})"), tags=(tag,))
        
        # Configure tags for colors
        self.features_tree.tag_configure("high", background="#FFCDD2")
        self.features_tree.tag_configure("medium", background="#FFF9C4")
        self.features_tree.tag_configure("low", background="#C8E6C9")
    
    def update_visualization(self, features, feature_importance, is_phishing):
        # Clear previous charts
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        # Create figure with two subplots
        fig = plt.Figure(figsize=(10, 6), dpi=100)
        
        # Feature importance chart
        ax1 = fig.add_subplot(121)
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1])
        feature_names = [name for name, _ in sorted_features]
        importance_values = [value for _, value in sorted_features]
        
        # Horizontal bar chart
        bars = ax1.barh(feature_names, importance_values, color='#2196F3')
        ax1.set_title('Feature Importance')
        ax1.set_xlabel('Importance')
        
        # Add values to bars
        for bar in bars:
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                    ha='left', va='center', fontsize=8)
        
        # Risk score gauge chart
        ax2 = fig.add_subplot(122, polar=True)
        
        # Define the gauge
        gauge_min, gauge_max = 0, 10
        gauge_range = gauge_max - gauge_min
        
        # Convert confidence to gauge value (0-10)
        if is_phishing:
            gauge_value = features['suspicious_words'] + features['has_ip'] * 2 + features['has_at_symbol'] * 2 + \
                         (1 - features['is_https']) * 2 + features['has_double_slash'] * 2 + features['has_prefix_suffix']
            gauge_value = min(max(gauge_value, gauge_min), gauge_max)  # Clamp between min and max
        else:
            gauge_value = gauge_min
        
        # Normalize to 0-1 for the gauge
        norm_value = (gauge_value - gauge_min) / gauge_range
        
        # Create the gauge
        theta = np.linspace(0.25 * np.pi, 1.75 * np.pi, 100)
        r = np.ones_like(theta)
        
        # Background
        ax2.bar(theta, r, width=np.pi * 1.5 / len(theta), color='#E0E0E0', edgecolor='white', alpha=0.1)
        
        # Value
        value_theta = theta[:int(norm_value * len(theta))]
        value_r = np.ones_like(value_theta)
        
        # Color based on value (green to red)
        if gauge_value < 3.5:
            color = '#4CAF50'  # Green
        elif gauge_value < 7:
            color = '#FFC107'  # Yellow
        else:
            color = '#FF5252'  # Red
            
        ax2.bar(value_theta, value_r, width=np.pi * 1.5 / len(theta), color=color, edgecolor='white')
        
        # Remove ticks and spines
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.spines['polar'].set_visible(False)
        
        # Add gauge labels
        ax2.text(0, 0, f'{gauge_value:.1f}/10', ha='center', va='center', fontsize=24, fontweight='bold')
        ax2.text(0, -0.2, 'Risk Score', ha='center', va='center', fontsize=12)
        
        # Add low/high labels
        ax2.text(0.25 * np.pi, 1.1, 'Low', ha='left', va='center', fontsize=10)
        ax2.text(1.75 * np.pi, 1.1, 'High', ha='right', va='center', fontsize=10)
        
        # Adjust layout
        fig.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

# --- Main Function ---

def main():
    # Create dataset and train model
    print("Creating dataset...")
    df, urls, labels = create_dataset()
    
    print("Training model...")
    model = train_model(df)
    
    # Create UI
    root = tk.Tk()
    app = PhishingDetectorApp(root, model)
    root.mainloop()

if __name__ == "__main__":
    main()