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

# Load the data
@st.cache_data
def load_data():
    listings = pd.read_csv("/Users/apple/Downloads/listings.csv")
    reviews = pd.read_csv("/Users/apple/Downloads/reviews.csv")

    # Convert price column to numeric
    listings["price"] = listings["price"].replace("[\$,]", "", regex=True).astype(float)

    # Convert date columns
    listings["last_scraped"] = pd.to_datetime(listings["last_scraped"], errors='coerce')
    reviews["date"] = pd.to_datetime(reviews["date"], errors='coerce')

    return listings, reviews

listings, reviews = load_data()

# Streamlit UI
st.title("🏡 London Airbnb Analysis Dashboard")
st.sidebar.header("🔍 Filter Listings")

# Sidebar filters
price_range = st.sidebar.slider("Select Price Range (£)", int(listings["price"].min()), int(listings["price"].max()), (50, 300))
room_type = st.sidebar.selectbox("Room Type", listings["room_type"].unique(), index=0)
neighborhood = st.sidebar.selectbox("Select Neighborhood", listings["neighbourhood_cleansed"].unique(), index=0)

# Filter data based on user input
filtered_data = listings[(listings["price"].between(price_range[0], price_range[1])) &
                         (listings["room_type"] == room_type) &
                         (listings["neighbourhood_cleansed"] == neighborhood)]

st.subheader(f"📌 Showing10 Listings")
st.write(filtered_data[["name", "host_name", "price", "room_type", "neighbourhood_cleansed"]].head(10))

# --- Price Distribution ---
st.subheader("💰 Price Distribution")
fig = px.histogram(listings, x="price", nbins=40, title="Price Distribution of Airbnb Listings")
st.plotly_chart(fig)

# --- Room Type vs Price ---
st.subheader("🏠 Room Type vs Price")
fig = px.box(listings, x="room_type", y="price", title="Price Comparison by Room Type", color="room_type")
st.plotly_chart(fig)

# --- Top Hosts by Listings ---
st.subheader("👑 Top Hosts")
top_hosts = listings.groupby("host_name")["host_listings_count"].sum().sort_values(ascending=False).head(10)
fig = px.bar(x=top_hosts.index, y=top_hosts.values, labels={"x": "Host", "y": "Number of Listings"}, title="Top 10 Hosts")
st.plotly_chart(fig)

# --- Sentiment Analysis ---
st.subheader("📝 Review Sentiment Analysis")
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

reviews["sentiment_score"] = reviews["comments"].apply(get_sentiment)

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


# --- Word Cloud for Reviews ---
st.subheader("💬 Most Common Words in Reviews")
all_reviews = " ".join(reviews["comments"].dropna())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_reviews)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot(plt)


listings["price"] = pd.to_numeric(listings["price"], errors='coerce')  # Convert to numeric
listings["price"].fillna(listings["price"].median(), inplace=True)  # Replace NaN with median price

# --- Map Visualization ---
st.subheader("📍 Airbnb Listings on Map")
fig = px.scatter_mapbox(listings, lat="latitude", lon="longitude", color="price", size="price",
                         hover_data=["name", "host_name", "room_type"],
                         title="London Airbnb Map",
                         mapbox_style="open-street-map")
st.plotly_chart(fig)

st.write("🔍 *Data sourced from InsideAirbnb*")


st.subheader("🔮 Predict Airbnb Price")

# --- Data Preprocessing ---
df_model = listings[["price", "room_type", "neighbourhood_cleansed", "accommodates", "bedrooms", "beds", "bathrooms_text"]].dropna()

# Convert bathrooms_text to numeric
df_model["bathrooms"] = df_model["bathrooms_text"].str.extract("(\d+)").astype(float)

# Label Encoding for categorical variables
le_room_type = LabelEncoder()
le_neighborhood = LabelEncoder()

df_model["room_type_enc"] = le_room_type.fit_transform(df_model["room_type"])
df_model["neighborhood_enc"] = le_neighborhood.fit_transform(df_model["neighbourhood_cleansed"])

# Select features & target
X = df_model[["room_type_enc", "neighborhood_enc", "accommodates", "bedrooms", "beds", "bathrooms"]]
y = df_model["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Prediction Form ---

st.write("🔢 Enter details below to predict the price of an Airbnb listing:")

room_type_input = st.selectbox("Room Type", listings["room_type"].unique(), key="room_type_select")
neighborhood_input = st.selectbox("Neighborhood", listings["neighbourhood_cleansed"].unique(), key="neighborhood_select")
accommodates_input = st.slider("Accommodates", 1, 10, 2, key="accommodates_slider")
bedrooms_input = st.slider("Bedrooms", 0, 5, 1, key="bedrooms_slider")
beds_input = st.slider("Beds", 0, 5, 1, key="beds_slider")
bathrooms_input = st.slider("Bathrooms", 0.0, 5.0, 1.0, key="bathrooms_slider")

# Encode inputs
room_type_enc = le_room_type.transform([room_type_input])[0]
neighborhood_enc = le_neighborhood.transform([neighborhood_input])[0]

# Predict price
if st.button("💰 Predict Price", key="predict_price_btn"):
    input_data = np.array([[room_type_enc, neighborhood_enc, accommodates_input, bedrooms_input, beds_input, bathrooms_input]])
    predicted_price = model.predict(input_data)[0]
    st.success(f"💵 Predicted Price: £{predicted_price:.2f}")




