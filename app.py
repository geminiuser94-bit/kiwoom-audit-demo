import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from faker import Faker
import requests
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime, timedelta
import json
import re
from newspaper import Article

# ... (Previous imports) ...

# ... (Inside NewsCrawler class) ...

st.set_page_config(
    page_title="키움증권 내부감사 AI 시스템",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Kiwoom Securities Theme (Pink/Navy)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
    
    /* Global Font & Colors */
    :root {
        --primary-color: #EB008B; /* Kiwoom Pink */
        --secondary-color: #002060; /* Kiwoom Navy */
        --background-color: #F0F2F6;
        --text-color: #333333;
    }
    
    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', sans-serif;
        color: var(--text-color);
    }
    
    .stApp {
        background-color: var(--background-color);
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, var(--secondary-color) 0%, #003399 100%);
        padding: 25px 30px;
        border-radius: 15px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.0rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255,255,255,0.8);
        margin: 5px 0 0 0;
        font-size: 0.95rem;
    }
    
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: var(--primary-color);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 8px;
        color: #666;
        font-weight: 600;
        border: 1px solid #e0e0e0;
        padding: 0 20px;
        transition: all 0.3s;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--secondary-color) !important;
        color: white !important;
        border: none;
        box-shadow: 0 4px 10px rgba(0,32,96,0.3);
    }
    
    /* Button Styling */
    div.stButton > button {
        background-color: var(--secondary-color);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 700;
        transition: all 0.3s;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div.stButton > button:hover {
        background-color: var(--primary-color);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(235,0,139,0.3);
    }
    
    /* News Card Styling */
    .news-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        border-left: 5px solid #ddd;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: all 0.2s;
    }
    .news-card:hover {
        transform: translateX(5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .news-card.source-naver { border-left-color: #03C75A; }
    .news-card.source-google { border-left-color: #4285F4; }
    .news-card.source-fss { border-left-color: #002060; }
    .news-card.source-fsc { border-left-color: #EB008B; }
    
    .news-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        color: white;
        margin-bottom: 8px;
        text-transform: uppercase;
    }
    .badge-naver { background-color: #03C75A; }
    .badge-google { background-color: #4285F4; }
    .badge-fss { background-color: #002060; }
    .badge-fsc { background-color: #EB008B; }
    
    h4 {
        margin: 5px 0 10px 0 !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        line-height: 1.4 !important;
    }
    
    /* Perplexity Report Styling */
    .perplexity-report-container {
        background-color: #ffffff;
        padding: 35px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        margin-top: 25px;
        font-family: 'Noto Sans KR', sans-serif;
        color: #333;
        line-height: 1.7;
    }
    .perplexity-report-container h3 {
        color: #002060;
        border-bottom: 3px solid #EB008B;
        padding-bottom: 12px;
        margin-top: 30px;
        margin-bottom: 20px;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .perplexity-report-container h4 {
        background-color: #f8f9fa;
        padding: 15px 20px;
        border-left: 6px solid #002060;
        color: #333;
        margin-top: 25px;
        margin-bottom: 15px;
        border-radius: 0 8px 8px 0;
        font-size: 1.15rem;
        font-weight: 700;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }
    .perplexity-report-container strong {
        color: #EB008B;
        font-weight: 700;
        background-color: rgba(235, 0, 139, 0.05);
        padding: 0 4px;
        border-radius: 4px;
    }
    .perplexity-report-container ul {
        margin-bottom: 20px;
        padding-left: 25px;
    }
    .perplexity-report-container li {
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Helper Classes (Logic)
# -----------------------------------------------------------------------------
class AuditDataGenerator:
    """Generates synthetic corporate card data with injected anomalies."""
    def __init__(self):
        self.fake = Faker('ko_KR')
        self.employees = [self.fake.name() for _ in range(50)]
        self.departments = ['IB사업부', '리테일금융팀', 'IT개발팀', '리스크관리팀', '인사팀', '법인영업팀']
        # Assign home regions to employees (excluding Yeouido which is the office location)
        self.regions = ['강남구', '서초구', '송파구', '마포구', '용산구', '성동구', '분당구', '일산']
        self.office_region = '영등포구(여의도)'
        self.employee_homes = {emp: random.choice(self.regions) for emp in self.employees}
        
    def generate_base_data(self, n_rows=10000):
        data = []
        start_date = datetime.now() - timedelta(days=90)
        
        for _ in range(n_rows):
            # Normal transaction logic
            dt = start_date + timedelta(days=random.randint(0, 90), 
                                      hours=random.randint(9, 22), 
                                      minutes=random.randint(0, 59))
            is_holiday = dt.weekday() >= 5 # 5=Sat, 6=Sun
            
            emp_name = random.choice(self.employees)
            
            # Normal transactions mostly happen near office or business districts
            if random.random() < 0.8:
                merchant_region = self.office_region
            else:
                merchant_region = random.choice(self.regions + ['종로구', '중구'])
                
            row = {
                'transaction_time': dt,
                'amount': round(random.lognormvariate(10, 1) * 1000, -2), # Log-normal distribution
                'merchant_name': self.fake.company() + " 식당",
                'merchant_region': merchant_region,
                'mcc_code': '일반음식점',
                'employee_name': '김OO',
                'home_region': self.employee_homes[emp_name],
                'department': 'OO팀',
                'is_holiday': is_holiday,
                'anomaly_type': 'Normal'
            }
            data.append(row)
        return pd.DataFrame(data)

    def inject_anomalies(self, df):
        anomalies = []
        
        # Scenario A: Split Payments (쪼개기 결제)
        for _ in range(35):
            base_time = df['transaction_time'].sample().values[0]
            base_time = pd.to_datetime(base_time)
            emp = random.choice(self.employees)
            dept = random.choice(self.departments)
            merchant = "한우 오마카세 " + self.fake.word()
            region = self.office_region # Usually near office
            
            total_amount = random.randint(500000, 1500000)
            split_count = random.randint(2, 4)
            amount_per_txn = total_amount // split_count
            
            for i in range(split_count):
                anomalies.append({
                    'transaction_time': base_time + timedelta(minutes=i*2),
                    'amount': amount_per_txn,
                    'merchant_name': merchant,
                    'merchant_region': region,
                    'mcc_code': '일반음식점',
                    'employee_name': '김OO',
                    'home_region': self.employee_homes[emp],
                    'department': 'OO팀',
                    'is_holiday': base_time.weekday() >= 5,
                    'anomaly_type': 'Split Payment'
                })

        # Scenario B: Late Night / Holiday / Entertainment (심야/휴일 유흥)
        for _ in range(20):
            base_time = df['transaction_time'].sample().values[0]
            base_time = pd.to_datetime(base_time).replace(hour=random.choice([23, 0, 1, 2, 3]))
            emp = random.choice(self.employees)
            
            anomalies.append({
                'transaction_time': base_time,
                'amount': random.randint(200000, 800000),
                'merchant_name': random.choice(['강남 룸싸롱', 'VIP 노래방', '황제 유흥주점']),
                'merchant_region': '강남구', # Entertainment district
                'mcc_code': '유흥주점',
                'employee_name': '김OO',
                'home_region': self.employee_homes[emp],
                'department': 'OO팀',
                'is_holiday': base_time.weekday() >= 5,
                'anomaly_type': 'Restricted Time/Sector'
            })

        # Scenario C: Clean Card Violation (Misleading Merchant Name)
        for _ in range(15):
            base_time = df['transaction_time'].sample().values[0]
            base_time = pd.to_datetime(base_time)
            emp = random.choice(self.employees)
            
            anomalies.append({
                'transaction_time': base_time,
                'amount': random.randint(150000, 400000),
                'merchant_name': '시크릿 Bar ' + self.fake.word(),
                'merchant_region': random.choice(self.regions), # Random location
                'mcc_code': '일반음식점', # Disguised as restaurant
                'employee_name': '김OO',
                'home_region': self.employee_homes[emp],
                'department': 'OO팀',
                'is_holiday': base_time.weekday() >= 5,
                'anomaly_type': 'Clean Card Violation'
            })

        # Scenario D: Personal Expense Near Home (자택 근처 결제)
        for _ in range(25):
            base_time = df['transaction_time'].sample().values[0]
            base_time = pd.to_datetime(base_time)
            # Weekend or Late Night usually
            if random.random() > 0.5:
                base_time = base_time.replace(hour=random.choice([20, 21, 22, 10, 11])) # Late night or weekend brunch
            
            emp = random.choice(self.employees)
            home = self.employee_homes[emp]
            
            anomalies.append({
                'transaction_time': base_time,
                'amount': random.randint(50000, 200000), # Not necessarily huge amount
                'merchant_name': f"{home} {self.fake.word()} 마트",
                'merchant_region': home, # Match Home Region
                'mcc_code': '마트/편의점',
                'employee_name': '김OO',
                'home_region': home,
                'department': 'OO팀',
                'is_holiday': True, # Often weekends
                'anomaly_type': 'Personal Expense (Near Home)'
            })
            
        return pd.concat([df, pd.DataFrame(anomalies)], ignore_index=True)

class AnomalyDetector:
    """Detects anomalies using XGBoost (Supervised Learning)."""
    def train_and_predict(self, df):
        # 1. Preprocessing
        # Create Target Variable (Labeling)
        df['target'] = df['anomaly_type'].apply(lambda x: 0 if x == 'Normal' else 1)
        
        # Feature Engineering
        df['hour'] = df['transaction_time'].dt.hour
        df['day_of_week'] = df['transaction_time'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # New Feature: Is Near Home?
        df['is_near_home'] = (df['merchant_region'] == df['home_region']).astype(int)
        
        # Encode Categorical Variables
        le_dept = LabelEncoder()
        df['department_encoded'] = le_dept.fit_transform(df['department'])
        
        le_mcc = LabelEncoder()
        df['mcc_code_encoded'] = le_mcc.fit_transform(df['mcc_code'])
        
        # Select Features for Training
        features = ['amount', 'hour', 'is_weekend', 'department_encoded', 'mcc_code_encoded', 'is_near_home']
        X = df[features]
        y = df['target']
        
        # 2. Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 3. Train XGBoost Model
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # 4. Predict & Score
        # We use the probability of class 1 (Anomaly) as the score
        df['anomaly_score'] = model.predict_proba(X)[:, 1]
        
        # Thresholding (e.g., probability > 0.5 is an anomaly)
        # Since we have ground truth labels in this simulation, we can also just use the prediction
        df['is_anomaly_detected'] = model.predict(X) == 1
        
        # Optional: Print accuracy to console for debugging
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"XGBoost Model Accuracy: {acc:.4f}")
        
        return df

class NewsCrawler:
    """Crawls Naver News, Google News, FSS, and FSC."""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

    def get_all_news(self, pplx_api_key=None):
        """Aggregates news. Uses Perplexity if key provided, otherwise legacy crawlers."""
        all_news = []
        
        # 1. Official Sources (Always Crawl Direct URLs for accuracy)
        all_news.extend(self.crawl_fsc())
        all_news.extend(self.crawl_fss())
        
        # 2. Internet News (Naver & Google) - Broad Crawling
        # We crawl broadly and then let AI filter the results
        naver_queries = ["증권사 금융사고", "금융감독원 제재", "주가조작", "횡령 배임", "자본시장법 위반"]
        for q in naver_queries:
            all_news.extend(self.crawl_naver(q))
            time.sleep(0.2)
        
        google_queries = ["금융감독원 제재", "증권사 내부통제", "금융사고"]
        for q in google_queries:
            all_news.extend(self.crawl_google_rss(q))

        # 3. AI-Curated News (Perplexity)
        if pplx_api_key:
            # Use Perplexity for high-quality, summarized news search
            pplx_news = self.fetch_news_with_perplexity(pplx_api_key)
            if isinstance(pplx_news, list):
                all_news.extend(pplx_news)
            else:
                print("Perplexity Fallback triggered due to API error.")
        
        # Deduplicate by title
        unique_news = list({news['title']: news for news in all_news}.values())
        
        # Filter & Rank
        if pplx_api_key:
            # AI-Based Filtering (Verification)
            filtered_news = self.filter_news_with_ai(unique_news, pplx_api_key)
        else:
            # Rule-Based Filtering
            filtered_news = [n for n in unique_news if self.is_relevant(n)]
        
        # Rank
        ranked_news = self.rank_news(filtered_news)
        
        return ranked_news

    def rank_news(self, news_list):
        """Rank news by importance using weighted keywords."""
        # High Risk Keywords (Weight: 3)
        critical_keywords = ['횡령', '배임', '구속', '압수수색', '제재', '과징금', '영업정지', '등록취소', '검찰', '고발']
        # Medium Risk Keywords (Weight: 2)
        warning_keywords = ['주의', '경고', '적발', '위반', '불공정', '조작', '미공개', '손실', '부실', '사고', '검사']
        # Low Risk Keywords (Weight: 1)
        general_keywords = ['금융위', '금감원', '감독', '규제', '개정', '발표']
        
        scored_news = []
        for news in news_list:
            score = 0
            title = news['title']
            summary = news['summary']
            combined = title + " " + summary
            
            for k in critical_keywords:
                if k in combined: score += 3
            for k in warning_keywords:
                if k in combined: score += 2
            for k in general_keywords:
                if k in combined: score += 1
                
            score += random.random() # Tie-breaker
            news['score'] = score
            scored_news.append(news)
            
        return sorted(scored_news, key=lambda x: x['score'], reverse=True)

    def is_relevant(self, news):
        """Checks if the news is relevant using detailed audit keywords."""
        # 1. Critical Risk (Must Catch)
        risk_keywords = [
            '횡령', '배임', '차명', '선행매매', '스캘핑', '일임매매', '과당매매', '부당권유', '이면계약', '꺾기', '자금세탁', '리베이트', # Fraud
            '시세조종', '주가조작', '통정매매', '가장매매', '미공개정보', '무차입공매도', '허수주문', '블록딜', '자전거래', '채권파킹', '윈도우드레싱', # Market Manipulation
            'PF부실', '기한이익상실', 'EOD', '브릿지론', '책임준공', '우발채무', '대손충당금', '순자본비율', 'NCR', '유동성비율', 'LCR', # IB/Risk
            '불완전판매', 'ELS', 'DLS', '랩어카운트', '사모펀드', '환매중단', '원금손실', '해피콜', # Consumer Protection
            '전산장애', '망분리', '개인정보유출', '접근통제', 'DDoS', '랜섬웨어', '이상금융거래', 'FDS', '오픈API', '클라우드', # IT
            '책무구조도', '내부통제', '기관경고', '기관주의', '임원문책', '직무정지', '과징금', '과태료', '공시위반', '대주주적격성', # Regulation
            '분식회계', '법인카드', '접대비', '가지급금', '성희롱', '채용비리', '내부고발', 'STO', '가상자산' # General/New Biz
        ]
        
        # 2. Irrelevant Contexts (Filter Out)
        ignore_keywords = ['채용공고', '이벤트', '우승', '스포츠', '날씨', '부고', '인사동정', '광고', '홍보', '캠페인', '봉사활동', 'MOU체결']
        
        title = news['title']
        summary = news['summary']
        combined = (title + " " + summary)
        
        # Filter out irrelevant
        if any(bad in combined for bad in ignore_keywords):
            return False
            
        # Check for relevant keywords
        if any(good in combined for good in risk_keywords):
            return True
            
        return False

    def filter_news_with_ai(self, news_list, api_key):
        """Uses Perplexity to verify relevance of news items."""
        if not news_list: return []
        
        # Prepare list for prompt
        news_text = "\n".join([f"{i}. {n['title']} (Summary: {n['summary']})" for i, n in enumerate(news_list)])
        
        system_prompt = """
        You are an expert Audit Assistant for Kiwoom Securities.
        Review the provided list of news items. 
        Select ONLY the items that are critical for the Internal Audit Team (Fraud, Regulation, Risk, IT Security, Consumer Protection).
        Discard marketing, general market news, or irrelevant items.
        Return ONLY a JSON array of the INDICES (0-based integers) of the relevant items.
        Example: [0, 2, 5]
        """
        
        payload = {
            "model": "sonar",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"News List:\n{news_text}"}
            ]
        }
        
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post("https://api.perplexity.ai/chat/completions", json=payload, headers=headers)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                # Extract JSON array
                match = re.search(r'\[.*\]', content, re.DOTALL)
                if match:
                    indices = json.loads(match.group(0))
                    return [news_list[i] for i in indices if i < len(news_list)]
        except Exception as e:
            print(f"AI Filter Error: {e}")
            
        # Fallback to rule-based if AI fails
        return [n for n in news_list if self.is_relevant(n)]

    @staticmethod
    def fetch_news_with_perplexity(api_key):
        """Fetches latest financial news using Perplexity API."""
        url = "https://api.perplexity.ai/chat/completions"
        
        # Specific Prompt for Korean Securities Risks
        # Specific Prompt for Korean Securities Risks (Updated with detailed categories)
        system_prompt = """
        You are a specialized news aggregator for 'Kiwoom Securities' Audit Team. 
        Focus ONLY on South Korean financial news related to the following critical risk categories:
        
        1. Fraud & Embezzlement: Embezzlement, Breach of Trust, Borrowed Name Accounts, Front Running, Scalping, Churning, Rebates.
        2. Market Manipulation: Stock Manipulation, Insider Trading, Naked Short Selling, High Frequency Trading (HFT) Risks, Bond Parking, Window Dressing.
        3. IB & Credit Risk: PF Default, EOD, Bridge Loan Risks, Contingent Liabilities, NCR/LCR Issues.
        4. Consumer Protection: Misselling, ELS/DLS Knock-in, Fund Redemption Suspension.
        5. IT Security: System Failure, Network Separation Violation, Data Leakage, DDoS, Cloud Risks.
        6. Regulation: FSS/FSC Sanctions, CEO Risks, Disclosure Violations, Governance Issues.
        
        Exclude general market analysis, stock price updates, ESG campaigns, or global economy news unless directly relevant to these compliance risks.
        """
        user_prompt = "Search for the most recent (last 7 days) critical news items fitting the criteria. Provide 15 distinct items. Return ONLY a JSON array with keys: 'title', 'summary', 'source', 'link'. The content must be in Korean."
        
        payload = {
            "model": "sonar",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "User-Agent": "KiwoomAuditSystem/2.2"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                content = content.replace("```json", "").replace("```", "").strip()
                try:
                    news_data = json.loads(content)
                    return news_data
                except json.JSONDecodeError:
                    return f"⚠️ Failed to parse Perplexity response: {content[:100]}..."
            else:
                return f"⚠️ Perplexity API Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"⚠️ API Call Failed: {str(e)}"

    def crawl_naver(self, query):
        url = f"https://search.naver.com/search.naver?where=news&query={query}&sm=tab_opt&sort=1&photo=0&field=0&pd=0&ds=&de=&docid=&related=0&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so%3Add%2Cp%3Aall&is_sug_officeid=0"
        
        news_list = []
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                items = soup.select('.news_area')
                
                for item in items[:3]:
                    title_tag = item.select_one('.news_tit')
                    if title_tag:
                        title = title_tag.get_text()
                        link = title_tag['href']
                        dsc = item.select_one('.news_dsc')
                        summary = dsc.get_text() if dsc else "요약 없음"
                        news_list.append({'title': title, 'link': link, 'summary': summary, 'source': 'Naver'})
        except Exception as e:
            print(f"Naver Crawl Error: {e}")
        return news_list

    def crawl_google_rss(self, query):
        """Crawls Google News via RSS."""
        url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
        news_list = []
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, features='xml')
                items = soup.find_all('item')
                
                for item in items[:3]:
                    title = item.title.text
                    link = item.link.text
                    desc_html = item.description.text
                    summary = BeautifulSoup(desc_html, 'html.parser').get_text()[:100] + "..."
                    news_list.append({'title': title, 'link': link, 'summary': summary, 'source': 'Google'})
        except Exception as e:
            print(f"Google RSS Error: {e}")
        return news_list

    def crawl_fss(self):
        """Financial Supervisory Service - Improved Crawling with Fallback Labeling"""
        target_url = "https://www.fss.or.kr/fss/bbs/B0000188/list.do?menuNo=200218"
        news_list = []
        try:
            response = requests.get(target_url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            rows = soup.select('.bd-list tbody tr')
            
            for row in rows[:5]:
                title_tag = row.select_one('.subject a')
                if title_tag:
                    title = title_tag.get_text().strip()
                    href = title_tag['href']
                    date_tag = row.select_one('.date')
                    date = date_tag.get_text().strip() if date_tag else ""
                    full_link = "https://www.fss.or.kr" + href if href.startswith('/') else href
                    news_list.append({
                        'title': f"[금감원] {title}",
                        'link': full_link,
                        'summary': f"등록일: {date} | 금융감독원 보도자료입니다.",
                        'source': 'FSS'
                    })
            
            if not news_list:
                # Fallback to Google News but label as FSS
                fallback = self.crawl_google_rss("금융감독원 보도자료")
                for item in fallback:
                    item['source'] = 'FSS (via Google)'
                return fallback
                
        except Exception as e:
            print(f"FSS Crawl Error: {e}")
            fallback = self.crawl_google_rss("금융감독원")
            for item in fallback:
                item['source'] = 'FSS (via Google)'
            return fallback
            
        return news_list

    def crawl_fsc(self):
        """Financial Services Commission - Improved Crawling with Fallback Labeling"""
        target_url = "https://www.fsc.go.kr/no010101"
        news_list = []
        try:
            response = requests.get(target_url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # More robust finding: Look for links containing the board path
            links = soup.find_all('a', href=True)
            
            for a in links:
                href = a['href']
                # Filter for specific article links (usually contain /no010101/ + a number)
                if '/no010101/' in href and any(c.isdigit() for c in href):
                    title = a.get_text().strip()
                    if len(title) < 10: continue # Skip short nav links
                    
                    # Deduplicate
                    if any(n['title'] == f"[금융위] {title}" for n in news_list):
                        continue
                        
                    full_link = "https://www.fsc.go.kr" + href if href.startswith('/') else href
                    
                    news_list.append({
                        'title': f"[금융위] {title}",
                        'link': full_link,
                        'summary': "금융위원회 보도자료 및 주요 정책 발표입니다.",
                        'source': 'FSC'
                    })
                    if len(news_list) >= 5: break
                
            if not news_list:
                 # Fallback to Google News but label as FSC
                fallback = self.crawl_google_rss("금융위원회 보도자료")
                for item in fallback:
                    item['source'] = 'FSC (via Google)'
                return fallback

        except Exception as e:
            print(f"FSC Crawl Error: {e}")
            fallback = self.crawl_google_rss("금융위원회")
            for item in fallback:
                item['source'] = 'FSC (via Google)'
            return fallback
            
        return news_list



class GenAISimulator:
    """Simulates LLM text generation."""
    @staticmethod
    def generate_audit_report(anomaly_row):
        merchant = anomaly_row['merchant_name']
        amount = f"{anomaly_row['amount']:,}"
        time_str = anomaly_row['transaction_time'].strftime('%Y-%m-%d %H:%M')
        emp = anomaly_row['employee_name']
        
        prompt_context = ""
        if "Split" in str(anomaly_row.get('anomaly_type', '')):
            prompt_context = "동일 가맹점 단시간 반복 결제(쪼개기 결제) 의심"
        elif "Restricted" in str(anomaly_row.get('anomaly_type', '')):
            prompt_context = "심야/휴일 제한 업종(유흥) 결제 의심"
        elif "Clean" in str(anomaly_row.get('anomaly_type', '')):
            prompt_context = "클린카드 금지 업종 위장 결제 의심"
        elif "Personal" in str(anomaly_row.get('anomaly_type', '')):
            prompt_context = "자택 인근 사적 유용 의심 (Personal Expense Near Home)"
        else:
            prompt_context = "통상적이지 않은 고액 결제 패턴"

        report_template = f"""
### 📑 AI 감사 조서 초안 (Draft Audit Report)
**생성 일시:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**대상 직원:** {emp} ({anomaly_row['department']})
---
#### 1. 위반 혐의 분석 (Anomaly Analysis)
*   **탐지 유형:** {prompt_context}
*   **상세 내용:** {time_str}에 '{merchant}'에서 {amount}원이 결제되었습니다. 해당 건은 내부 통제 기준(Rule-Set) 및 AI 이상 탐지 모델에 의해 **Risk Score 98점**으로 분류되었습니다.
*   **특이 사항:** 동시간대 유사 업종 평균 결제액 대비 300% 이상 높으며, 결제 패턴이 비정상적입니다.

#### 2. 관련 내부 규정 (Regulation Check)
*   **제 3조 2항 (법인카드 사용 제한):** 유흥업종, 골프장, 심야 시간대(23:00~06:00) 사용을 원칙적으로 금지함.
*   **제 5조 1항 (분할 결제 금지):** 전결 규정 회피를 목적으로 한 분할 결제(일명 쪼개기)는 징계 대상임.

#### 3. 소명 요청 및 조치 계획 (Action Plan)
1.  **소명 자료 제출 요구:** {emp} 직원에게 해당 결제 건에 대한 영수증 및 사유서 제출 요청 (기한: 3일 내).
2.  **부서장 통보:** {anomaly_row['department']}장에게 위반 의심 사례 통보 및 관리 감독 강화 요청.
3.  **환수 조치 검토:** 소명 불충분 시 전액 환수 및 인사위원회 회부 검토.
---
*본 보고서는 생성형 AI가 작성한 초안이며, 감사인의 최종 검토가 필요합니다.*
"""
        return report_template

    @staticmethod
    def generate_threat_analysis(news_title):
        # Determine context based on keywords in the title
        title_lower = news_title.lower()
        
        if any(k in title_lower for k in ['횡령', '배임', '유용']):
            category = "자금 횡령 및 유용"
            risk_analysis = "해당 기사는 임직원에 의한 자금 횡령 및 사적 유용 가능성을 시사합니다. 특히 자금 집행 권한이 집중된 부서에서의 내부통제 실패가 주요 원인으로 분석됩니다."
            checklist = """
*   **[긴급]** 자금 집행 부서(PF, 법인영업)의 **직무 분리(Segregation of Duties)** 및 순환 근무 현황 점검.
*   **[상시]** 법인카드 및 계좌 이체 내역에 대한 이상 징후 모니터링 강화.
*   **[시스템]** 고액 자금 이체 시 다단계 승인 절차(Multi-Approval) 우회 여부 전수 조사."""
            
        elif any(k in title_lower for k in ['제재', '과징금', '기관경고', '조치']):
            category = "금융당국 제재 및 규제 위반"
            risk_analysis = "금융감독원 및 금융위원회의 제재 조치는 회사의 평판 리스크와 직결됩니다. 해당 건은 불완전 판매 또는 공시 의무 위반과 관련된 것으로 보입니다."
            checklist = """
*   **[긴급]** 최근 3년간 유사 사례에 대한 내부 감사 기록 재검토.
*   **[교육]** 전 임직원 대상 컴플라이언스(Compliance) 준수 교육 강화.
*   **[보고]** 제재 원인 분석 보고서 작성 및 재발 방지 대책 이사회 보고."""
            
        elif any(k in title_lower for k in ['주가', '시세', '불공정', '미공개']):
            category = "불공정 거래 및 시세 조종"
            risk_analysis = "미공개 정보 이용 또는 시세 조종 혐의는 자본시장법 위반의 중대 사안입니다. 임직원의 자기매매 및 차명 계좌 운용 가능성을 배제할 수 없습니다."
            checklist = """
*   **[긴급]** 임직원 자기매매 신고 내역과 실제 거래 내역 대사(Cross-Check).
*   **[모니터링]** 사내 메신저 및 이메일 키워드 검색을 통한 미공개 정보 유통 정황 포착.
*   **[시스템]** 매매 주문 기록(Log) 보존 상태 점검."""
            
        else:
            category = "기타 금융 사고 및 리스크"
            risk_analysis = "해당 기사는 일반적인 금융권 리스크 또는 정책 변화를 다루고 있습니다. 선제적인 내부 규정 정비가 필요할 수 있습니다."
            checklist = """
*   **[상시]** 관련 내부 규정(사규)의 현행 법규 부합 여부 검토.
*   **[모니터링]** 타사 사례를 벤치마킹하여 내부통제 사각지대 발굴.
*   **[점검]** 리스크 관리 위원회 안건 상정 검토."""

        return f"""
### 🛡️ 생성형 AI 위협 분석 (Threat Intelligence)
**분석 대상 뉴스:** {news_title}
**분류:** {category}
---
#### 1. 사건 개요 및 핵심 위험 (Key Risks)
{risk_analysis}

#### 2. 키움증권 내부 점검 필요 항목 (Internal Checklist)
{checklist}

#### 3. 대응 감사 계획 (Audit Plan)
*   **감사 명:** {category} 대응 특별/상시 감사
*   **예상 소요 기간:** 2주 (AI 사전 분석 3일 + 현장 감사 7일)
---
*AI Analysis Completed.*
"""

    @staticmethod
    def generate_analysis_with_perplexity(api_key, news_item):
        """Calls Perplexity API for deep analysis using RAG (Retrieval-Augmented Generation)."""
        # Basic Validation
        if not api_key.startswith("pplx-"):
            return "⚠️ 유효하지 않은 API Key 형식입니다. 'pplx-'로 시작하는 키를 입력해주세요."

        news_title = news_item.get('title', '제목 없음')
        news_link = news_item.get('link', '')
        
        # 1. RAG: Extract Full Content using newspaper3k
        full_text = ""
        try:
            if news_link and news_link != "#":
                article = Article(news_link, language='ko')
                article.download()
                article.parse()
                full_text = article.text[:3000] # Limit context window if necessary
        except Exception as e:
            full_text = f"(본문 추출 실패: {str(e)})"

        # 2. Construct Prompt with Full Context
        system_prompt = """
        당신은 키움증권 내부감사팀의 수석 감사역(Chief Auditor) AI입니다.
        제공된 뉴스 기사의 **본문(Full Text)**을 정밀 분석하여, 우리 회사(증권사)에 미칠 수 있는 잠재적 위협을 식별하고 구체적인 감사 대응 시나리오를 수립하세요.
        
        반드시 다음 4가지 섹션으로 구성된 전문적인 감사 보고서를 작성해 주세요:
        1. 🔍 **사건 심층 요약 (Executive Summary)**: 기사의 핵심 팩트와 연루된 금융 사고 유형을 명확히 요약.
        2. ⚠️ **핵심 리스크 식별 (Key Risk Indicators)**: 이 사건이 우리 회사에서 발생할 경우 예상되는 법적, 재무적, 평판 리스크.
        3. 🛡️ **감사 대응 시나리오 (Audit Response Scenario)**: 
           - 만약 이 사건이 우리 회사에서 발생했다면, 어떤 데이터와 로그를 확인해야 하는가?
           - 구체적인 감사 절차(Audit Procedure)와 적발 기법.
        4. ✅ **즉시 점검 체크리스트 (Actionable Checklist)**: 내일 당장 현업 부서에 배포할 구체적인 점검 항목 (부서명 명시).
        
        답변은 키움증권의 내부 보고서 톤앤매너(전문적, 직관적, 핵심 위주)를 유지하세요.
        """
        
        user_prompt = f"""
        [분석 대상 뉴스]
        - 제목: {news_title}
        - 링크: {news_link}
        - 본문 내용:
        {full_text}
        
        위 내용을 바탕으로 심층 감사 보고서를 작성해 주세요.
        """

        url = "https://api.perplexity.ai/chat/completions"
        
        payload = {
            "model": "sonar", 
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "User-Agent": "KiwoomAuditSystem/2.3"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            
            try:
                result = response.json()
            except ValueError:
                return f"⚠️ API Error (Non-JSON Response): {response.status_code} - {response.text[:200]}..."

            if response.status_code == 200:
                content = result['choices'][0]['message']['content']
                
                # Post-process Markdown for Kiwoom Styling
                content = re.sub(r'### (.*)', r'<h3>\1</h3>', content)
                content = re.sub(r'#### (.*)', r'<h4>\1</h4>', content)
                content = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color: #EB008B;">\1</strong>', content)
                
                return content
            else:
                error_msg = result.get('error', {}).get('message', 'Unknown Error')
                return f"⚠️ Perplexity API Error: {response.status_code} - {error_msg}"
        except Exception as e:
            return f"⚠️ API Call Failed: {str(e)}"

# -----------------------------------------------------------------------------
# 3. Main Application Logic
# -----------------------------------------------------------------------------
def main():
    # Sidebar
    with st.sidebar:
        st.image("logo.png", width=200)
        st.markdown("---")
        st.header("⚙️ 시스템 제어")
        
        audit_date = st.date_input("감사 기준일", datetime.now())
        
        st.markdown("---")
        st.markdown("### 🔑 API Key 설정")
        pplx_api_key = st.text_input("Perplexity API Key", type="password", help="뉴스 수집 및 분석용 (Perplexity Pro)")
        
        st.info(f"기준일: {audit_date.strftime('%Y-%m-%d')}")
        
        st.markdown("---")
        st.caption("Developed for Kiwoom Securities Audit Team")
        st.caption("v2.3.0 (All-in-One Perplexity)")

    # Tabs
    tab1, tab2 = st.tabs(["📊 내부 데이터 감사 (Internal Audit)", "🌍 외부 위협 대응 (Threat Intelligence)"])
    
    # -------------------------------------------------------------------------
    # TAB 1: Internal Audit Simulation
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("💳 법인카드 이상 징후 탐지 (FDS)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 1. 데이터 생성 및 분석")
            st.write("최근 3개월치 법인카드 사용 내역 10,000건을 생성하고, XGBoost (Supervised Learning) 알고리즘으로 이상 징후를 탐지합니다.")
            
            if st.button("🚀 데이터 생성 및 AI 분석 시작", key="btn_run_audit"):
                with st.spinner("데이터 생성 및 이상 탐지 모델 구동 중..."):
                    # 1. Generate Data
                    generator = AuditDataGenerator()
                    df = generator.generate_base_data()
                    df = generator.inject_anomalies(df)
                    
                    # 2. Detect Anomalies
                    detector = AnomalyDetector()
                    df = detector.train_and_predict(df)
                    
                    st.session_state['audit_data'] = df
                    st.success("분석 완료! 우측 대시보드를 확인하세요.")
        
        with col2:
            if 'audit_data' in st.session_state:
                df = st.session_state['audit_data']
                anomalies = df[df['is_anomaly_detected']]
                
                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("총 결제 건수", f"{len(df):,}건")
                m2.metric("총 결제 금액", f"{df['amount'].sum()//100000000}억원")
                m3.metric("이상 징후 탐지", f"{len(anomalies):,}건", delta="Risk", delta_color="inverse")
                m4.metric("이상 비율", f"{len(anomalies)/len(df)*100:.2f}%")
                
                # Chart
                st.markdown("### 2. 이상 탐지 시각화")
                fig = px.scatter(
                    df, 
                    x="transaction_time", 
                    y="amount", 
                    color="is_anomaly_detected",
                    color_discrete_map={True: '#EB008B', False: '#002060'}, # Kiwoom Colors
                    hover_data=['merchant_name', 'employee_name', 'anomaly_type'],
                    title="시간대별 법인카드 결제 금액 분포 (Red: 이상 징후)",
                    opacity=0.6,
                    height=500
                )
                fig.update_layout(plot_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)
                
                # Detail Analysis & Report Generation
                st.markdown("### 3. AI 감사 리포트 생성")
                st.write("탐지된 이상 징후 중 하나를 선택하여 AI 감사 조서를 생성합니다.")
                
                # Filter only anomalies for selection
                anomaly_options = anomalies.sort_values('amount', ascending=False).head(20)
                selected_idx = st.selectbox(
                    "분석할 이상 거래 선택 (Top 20 Risk Items):",
                    options=anomaly_options.index,
                    format_func=lambda x: f"[{anomaly_options.loc[x, 'anomaly_type']}] {anomaly_options.loc[x, 'merchant_name']} - {anomaly_options.loc[x, 'amount']:,}원 ({anomaly_options.loc[x, 'employee_name']})"
                )
                
                if st.button("📝 AI 감사 조서 작성 (Generate Report)", key="btn_report"):
                    row = df.loc[selected_idx]
                    report = GenAISimulator.generate_audit_report(row)
                    
                    st.markdown('<div class="perplexity-report-container">', unsafe_allow_html=True)
                    st.markdown(report)
                    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 2: External Threat Intelligence
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("🌍 외부 금융 위협 정보 수집 및 분석")
        st.markdown("""
        **Perplexity AI**를 활용하여 최신 금융권 위협 정보를 수집하고 심층 분석을 수행합니다.
        """)
        
        if st.button("🔍 최신 금융권 위협 정보 수집 (Hybrid)", key="btn_news_auto"):
            with st.spinner("금융위원회/금감원 공식 자료 및 Perplexity AI 기반 뉴스 수집 중..."):
                crawler = NewsCrawler()
                # Pass API key to get_all_news
                news_results = crawler.get_all_news(pplx_api_key if pplx_api_key else None)
                st.session_state['news_results'] = news_results
            st.success(f"총 {len(news_results)}건의 중요 뉴스 수집 완료! (FSC/FSS + AI Search)")
            
            if not pplx_api_key:
                st.warning("⚠️ Perplexity API Key가 입력되지 않아 일반 크롤링 모드로 동작했습니다. 더 정확한 결과를 위해 Key를 입력해주세요.")
            
        if 'news_results' in st.session_state:
            news_list = st.session_state['news_results']
            
            for i, news in enumerate(news_list):
                # Badge Color
                source_lower = news.get('source', '').lower()
                badge_class = "badge-naver"
                card_class = "source-naver"
                
                if "google" in source_lower:
                    badge_class = "badge-google"
                    card_class = "source-google"
                elif "fss" in source_lower or "금감원" in source_lower:
                    badge_class = "badge-fss"
                    card_class = "source-fss"
                elif "fsc" in source_lower or "금융위" in source_lower:
                    badge_class = "badge-fsc"
                    card_class = "source-fsc"
                
                with st.container():
                    st.markdown(f"""
                    <div class="news-card {card_class}">
                        <span class="news-badge {badge_class}">{news.get('source', 'Unknown')}</span>
                        <h4 style="margin: 5px 0;">{news.get('title', 'No Title')}</h4>
                        <p style="color: #666; font-size: 0.9rem; margin-bottom: 10px;">{news.get('summary', '')}</p>
                        <a href="{news.get('link', '#')}" target="_blank" style="text-decoration: none; color: #002060; font-weight: bold; font-size: 0.85rem;">🔗 원문 보기</a>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # AI Analysis Button for each news item
                    if st.button(f"🤖 AI 심층 분석 (Deep Analysis)", key=f"btn_analyze_{i}"):
                        if not pplx_api_key:
                            st.warning("⚠️ Perplexity API Key를 입력하면 더 정확한 심층 분석이 가능합니다. (현재는 템플릿 사용)")
                            # Fallback to Template
                            analysis_result = GenAISimulator.generate_threat_analysis(news['title'])
                            st.markdown(analysis_result)
                        else:
                            with st.spinner("Perplexity AI가 해당 사건을 정밀 분석 중입니다..."):
                                # Use Perplexity API for Analysis
                                analysis_result = GenAISimulator.generate_analysis_with_perplexity(pplx_api_key, news)
                                
                                # Display in a styled container
                                st.markdown('<div class="perplexity-report-container">', unsafe_allow_html=True)
                                st.markdown(analysis_result, unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
