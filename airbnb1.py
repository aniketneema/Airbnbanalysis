import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np
import swifter  # Speeds up pandas apply functions
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import json


# --- Google Drive API Setup ---
# Load service account credentials
SERVICE_ACCOUNT_FILE = json.loads(st.secrets["google_drive"]) # Path to your JSON key file

# Authenticate and create a service object
credentials = service_account.Credentials.from_service_account_info(
   SERVICE_ACCOUNT_FILE, scopes=["https://www.googleapis.com/auth/drive.readonly"])
service = build("drive", "v3", credentials=credentials)

# --- Function to Stream File from Google Drive ---
def stream_file_from_google_drive(file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return fh

# --- Load Dataset in Chunks ---

def load_data_in_chunks(file_id, chunksize=10000,nrows=50000):
    fh = stream_file_from_google_drive(file_id)
    chunks = pd.read_csv(fh, chunksize=chunksize,nrows=nrows)
    return chunks

# --- Load the Data ---

def load_data():
    # Google Drive file IDs (replace with your file IDs)
    listings_file_id = "1kGbGzSujS6s3lofsBh8lkZZ85LNPNDKn"
    reviews_file_id = "1tvrsGjFHcZ2SX5EQB-5a099XpFOGW0HS"

    # Load listings data
    listings_chunks = load_data_in_chunks(listings_file_id)
    listings = pd.concat(listings_chunks)

    # Load reviews data
    reviews_chunks = load_data_in_chunks(reviews_file_id)
    reviews = pd.concat(reviews_chunks)

    # Convert price column to numeric
    listings["price"] = listings["price"].replace("[\$,]", "", regex=True).astype(float)

    # Convert date columns
    listings["last_scraped"] = pd.to_datetime(listings["last_scraped"], errors='coerce')
    reviews["date"] = pd.to_datetime(reviews["date"], errors='coerce')

    return listings, reviews

listings, reviews = load_data()

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filter Listings")
price_range = st.sidebar.slider("Select Price Range (£)", int(listings["price"].min()), int(listings["price"].max()), (50, 300))
room_type = st.sidebar.selectbox("Room Type", listings["room_type"].unique(), key="room_type_filter")
neighborhood = st.sidebar.selectbox("Select Neighborhood", listings["neighbourhood_cleansed"].unique(), key="neighborhood_filter")

# --- Filter Data Based on Selection ---
filtered_data = listings[(listings["price"].between(price_range[0], price_range[1])) &
                         (listings["room_type"] == room_type) &
                         (listings["neighbourhood_cleansed"] == neighborhood)]

# --- Display Listings ---
st.title("🏡 London Airbnb Analysis Dashboard")
st.subheader("📌 Showing 10 Filtered Listings")
st.write(filtered_data[["name", "host_name", "price", "room_type", "neighbourhood_cleansed"]].head(10))

# --- Price Distribution (Filtered) ---
st.subheader("💰 Price Distribution")
fig = px.histogram(filtered_data, x="price", nbins=40, title="Filtered Price Distribution")
st.plotly_chart(fig)

# --- Top Hosts by Listings (Filtered) ---
st.subheader("👑 Top Hosts")
top_hosts = filtered_data.groupby("host_name")["host_listings_count"].sum().sort_values(ascending=False).head(10)
fig = px.bar(x=top_hosts.index, y=top_hosts.values, labels={"x": "Host", "y": "Number of Listings"}, title="Top 10 Hosts in Filtered Data")
st.plotly_chart(fig)

# --- Sentiment Analysis ---
st.subheader("📝 Review Sentiment Analysis")

@st.cache_data
def get_sentiment_scores(reviews):
    reviews["sentiment_score"] = reviews["comments"].dropna().swifter.apply(lambda text: TextBlob(str(text)).sentiment.polarity)
    return reviews

reviews = get_sentiment_scores(reviews)
fig = px.histogram(reviews, x="sentiment_score", nbins=20, title="Sentiment Score Distribution")
st.plotly_chart(fig)


st.subheader("📖 What Each Sentiment Score Means")
sentiment_table = pd.DataFrame({
    "Sentiment Score Range": ["+0.7 to +1.0", "+0.3 to +0.7", "+0.3 to -0.3", "-0.3 to -0.7", "-0.7 to -1.0"],
    "Sentiment Category": ["⭐️ Very Positive", "✅ Positive", "😐 Neutral", "⚠️ Negative", "❌ Very Negative"],
    "What It Conveys": [
        "Guests are extremely satisfied, praising the host, location, or amenities.",
        "Guests had a good experience, with minor issues (if any).",
        "Mixed opinions or factual statements without emotion.",
        "Guests had some issues, like cleanliness or noisy neighbors.",
        "Guests had a bad experience and would not recommend."
    ],
    "Example Review": [
        "Amazing stay! The host was super friendly, and the place was spotless!",
        "Nice place, comfortable and clean. The check-in process was easy.",
        "The apartment was as described. No issues, but nothing special.",
        "Decent location, but the apartment was smaller than expected and not very clean.",
        "Terrible stay! No hot water, and the host was unresponsive!"
    ]
})

st.table(sentiment_table)

# --- Word Cloud ---
st.subheader("💬 Most Common Words in Reviews")
all_reviews = " ".join(reviews["comments"].dropna())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_reviews)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot(plt)

# --- Map Visualization (Filtered) ---
st.subheader("📍 Airbnb Listings on Map")
fig = px.scatter_mapbox(filtered_data, lat="latitude", lon="longitude", color="price", size="price",
                         hover_data=["name", "host_name", "room_type"],
                         title="London Airbnb Map",
                         mapbox_style="open-street-map")
st.plotly_chart(fig)

# --- Airbnb Price Prediction Model ---
st.subheader("🔮 Predict Airbnb Price")

@st.cache_data
def preprocess_data(listings):
    df_model = listings[["price", "room_type", "neighbourhood_cleansed", "accommodates", "bedrooms", "beds", "bathrooms_text"]].dropna()
    
    # Convert bathrooms_text to numeric
    df_model["bathrooms"] = df_model["bathrooms_text"].str.extract("(\d+)").astype(float)

    # Label Encoding
    le_room_type = LabelEncoder()
    le_neighborhood = LabelEncoder()

    df_model["room_type_enc"] = le_room_type.fit_transform(df_model["room_type"])
    df_model["neighborhood_enc"] = le_neighborhood.fit_transform(df_model["neighbourhood_cleansed"])

    return df_model, le_room_type, le_neighborhood

df_model, le_room_type, le_neighborhood = preprocess_data(listings)

# Select Features & Target
X = df_model[["room_type_enc", "neighborhood_enc", "accommodates", "bedrooms", "beds", "bathrooms"]]
y = df_model["price"]

# Train Model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(X, y)

# --- Prediction Form ---
st.write("🔢 Enter details below to predict the price of an Airbnb listing:")

room_type_input = st.selectbox("Room Type", listings["room_type"].unique(), key="room_type_predict")
neighborhood_input = st.selectbox("Neighborhood", listings["neighbourhood_cleansed"].unique(), key="neighborhood_predict")
accommodates_input = st.slider("Accommodates", 1, 10, 2, key="accommodates_predict")
bedrooms_input = st.slider("Bedrooms", 0, 5, 1, key="bedrooms_predict")
beds_input = st.slider("Beds", 0, 5, 1, key="beds_predict")
bathrooms_input = st.slider("Bathrooms", 0.0, 5.0, 1.0, key="bathrooms_predict")

# Encode inputs using stored LabelEncoders
room_type_enc = le_room_type.transform([room_type_input])[0]
neighborhood_enc = le_neighborhood.transform([neighborhood_input])[0]

# Predict price
if st.button("💰 Predict Price", key="predict_price_btn"):
    input_data = np.array([[room_type_enc, neighborhood_enc, accommodates_input, bedrooms_input, beds_input, bathrooms_input]])
    predicted_price = model.predict(input_data)[0]
    st.success(f"💵 Predicted Price: £{predicted_price:.2f}")

st.write("🔍 *Data sourced from InsideAirbnb*")
