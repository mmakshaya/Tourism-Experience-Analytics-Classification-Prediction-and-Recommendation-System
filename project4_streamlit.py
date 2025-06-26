import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
import math
from sklearn.tree import DecisionTreeRegressor
import pickle
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Load your dataset
@st.cache_data
def load_data():
    df = pd.read_csv("tourism_sql.csv")  
    return df

df = load_data()

# --- Page Configuration ---
st.set_page_config(
    page_title="Tourism Experience Analytics: Classification, Prediction, and Recommendation SystemTourism Experience Analytics: Classification, Prediction, and Recommendation System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("🧳 Customize Your Trip")

#---Sidebar Navigation----
page = st.sidebar.radio(
    "Select Dataset",
    ["🏠 Home", "🌍 Geographic Insights", "🗓️ Time-Based Analysis","💬 Attraction Ratings Analysis","🧑‍🤝‍🧑 User Behavior Analysis","🧭 Exploring Attraction Categories","🗓️ Predictor","🧾Detailed Report"]
)


if page == "🏠 Home":
# Header
    st.title("🌍 Tourism Experience Analytics")
    st.subheader("📊 Key Metrics Overview")
    
    st.markdown("""
    <style>
    .card-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: space-between;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 16px;
        width: 23%;
        box-shadow: 1px 1px 8px rgba(0,0,0,0.05);
        font-size: 15px;
    }
    .card-title {
        font-weight: bold;
        color: #333;
        margin-bottom: 5px;
        font-size: 14px;
    }
    .card-value {
       font-size: 16px;
       color: #0077b6;
    }
    </style>

    <div class="card-container">
      <div class="card"><div class="card-title">📈 Avg. Attraction Rating</div><div class="card-value">4.3</div></div>
      <div class="card"><div class="card-title">⭐ Top-Rated Attraction</div><div class="card-value">Sacred Monkey Forest Sanctuary</div></div>
      <div class="card"><div class="card-title">🎯 Most Visited Attraction</div><div class="card-value">Waterbom Bali</div></div>
      <div class="card"><div class="card-title">💎 Hidden Gem</div><div class="card-value">Balekambang Beach</div></div>

      <div class="card"><div class="card-title">🌐 Top Country by Attractions</div><div class="card-value">Indonesia</div></div>
      <div class="card"><div class="card-title">🗺️ Top City by Attractions</div><div class="card-value">Jakarta</div></div>
      <div class="card"><div class="card-title">📍 Country with Highest Avg. Rating</div><div class="card-value">Libya / Bolivia (~5.0)</div></div>
      <div class="card"><div class="card-title">📉 Lowest Rated Country</div><div class="card-value">Moldova</div></div>

      <div class="card"><div class="card-title">👥 Avg. Ratings per User</div><div class="card-value">3.5</div></div>
      <div class="card"><div class="card-title">🔁 Most Repeat-Visited</div><div class="card-value">Sacred Monkey Forest</div></div>
      <div class="card"><div class="card-title">📆 Peak Month</div><div class="card-value">July / August</div></div>
      <div class="card"><div class="card-title">📉 Off-Peak Month</div><div class="card-value">February / November</div></div>

      <div class="card"><div class="card-title">🧭 Most Common Type</div><div class="card-value">Nature & Wildlife Areas (25%)</div></div>
      <div class="card"><div class="card-title">🛁 Highest-Rated Type</div><div class="card-value">Water Parks, Spas (~4.6+)</div></div>
      <div class="card"><div class="card-title">🕵️ Least Visited Type</div><div class="card-value">Spas, Specialty Museums</div></div>
      <div class="card"><div class="card-title">🏛️ Lowest Rated Type</div><div class="card-value">Beaches, Historic Sites</div></div>
    </div>
    """, unsafe_allow_html=True)


# Optional: Info box
    st.info("Navigate through the sidebar to explore detailed insights, predictions, and recommendations.")

# --- Home Page ---
elif page == "🌍 Geographic Insights":
    
    st.subheader("🗺️ Geographic Insights into Tourism Distribution")
    st.markdown("Visual insights into tourist distributions, top attractions, and location-based patterns.")

#  🧭 Top Cities with Most Attractions
    city_view = st.radio(
        "Select View",
        ["Top 20 Cities with Most Attractions", "Bottom 20 Cities with Fewest Attractions"],
        horizontal=True
    )

# Group by city
    city_attractions = df.groupby('CityName')['AttractionId'].nunique().reset_index()

    if city_view == "Top 20 Cities with Most Attractions":
    # Sort descending for Top 20
        city_attractions_sorted = city_attractions.sort_values(by='AttractionId', ascending=False).head(20)
        title = "🧭 Top 20 Cities with Most Attractions"
        color_scale = 'sunset'
    else:
    # Sort ascending for Bottom 20
        city_attractions_sorted = city_attractions.sort_values(by='AttractionId', ascending=True).head(20)
        title = "🔍 Bottom 20 Cities with Fewest Attractions"
        color_scale = 'sunset'

# Plot
    fig = px.bar(
       city_attractions_sorted,
       x='CityName',
       y='AttractionId',
       labels={'AttractionId': 'Number of Attractions', 'CityName': 'City'},
       title=title,
       color='AttractionId',
       color_continuous_scale=color_scale
    )

# Optional: Rotate x-axis labels for readability
    fig.update_layout(xaxis_tickangle=45)

# Display chart
    st.plotly_chart(fig, use_container_width=True)

# 🌍 Country/Region-wise Attraction Count

    col1, col2 = st.columns(2)
    # 📊 Column 1: Bar Chart
    with col1:
        
    # --- Prepare Data ---
        region_attractions = df.groupby(['Region', 'Country'])['AttractionId'].nunique().reset_index()
        region_attractions = region_attractions.rename(columns={'AttractionId': 'AttractionCount'})
        country_view = st.session_state.get("country_view_radio", "Top 25 Countries")

    # --- Choose Top or Bottom ---
        if country_view == "Top 25 Countries":
           selected_data = region_attractions.sort_values(by='AttractionCount', ascending=False).head(25)
           title = "🌎 Top 25 Countries by Attraction Count"
           color_scale = 'sunset'
        else:
           selected_data = region_attractions.sort_values(by='AttractionCount', ascending=True).head(25)
           title = "🌍 Bottom 25 Countries by Attraction Count"
           color_scale = 'sunset'

    # --- Plot ---
        fig = px.bar(
           selected_data,
           x='AttractionCount',
           y='Country',
           color='Region',
           orientation='h',
           title=title,
           labels={'AttractionCount': 'Attraction Count', 'Country': 'Country'},
           height=450,
           color_discrete_sequence=px.colors.qualitative.Safe
        )

        st.plotly_chart(fig, use_container_width=True)

        country_view = st.radio(
        "Choose View",
        ["Top 25 Countries", "Bottom 25 Countries"],
        horizontal=True,
        key="country_view_radio"
        )


# 🗺️ Column 2: Choropleth Map
    with col2:
    
# Group by country and count unique attractions
        country_attractions = df.groupby('Country')['AttractionId'].nunique().reset_index()
        country_attractions.columns = ['Country', 'AttractionCount']

# Create Choropleth map
        fig_choropleth = px.choropleth(
               country_attractions,
               locations='Country',
               locationmode='country names',
               color='AttractionCount',
               color_continuous_scale='YlOrRd',
               title='Attraction Density by Country (Choropleth map)',
               labels={'AttractionCount': 'Number of Attractions'},
               projection='natural earth'
        )

        fig_choropleth.update_geos(showframe=False, showcoastlines=True)

        st.plotly_chart(fig_choropleth, use_container_width=True)

# 3. 📍 Most Visited Attraction Locations (by Rating or Count)
# Count of visits or ratings
    most_visited = df.groupby(['Attraction', 'CityName', 'AttractionAddress'])['Rating'].count().reset_index()
    most_visited.columns = ['Attraction', 'City', 'Address', 'VisitCount']
    most_visited = most_visited.sort_values(by='VisitCount', ascending=False).head(100)

    fig3 = px.treemap(most_visited, 
                  path=['City', 'Attraction'], 
                  values='VisitCount',
                  title="📍Top 100 Most Visited Attractions",
                  color='VisitCount',
                  color_continuous_scale='Oranges')
    st.plotly_chart(fig3)

    

if page == "🗓️ Time-Based Analysis":
   
    st.subheader("🧭 Tourism Timeline")
    st.markdown("Understand seasonality and yearly trends in tourist activity")

# Group and plot yearly visit counts
    yearly_visits = df['VisitYear'].value_counts().reset_index()
    yearly_visits.columns = ['Year', 'VisitCount']
    yearly_visits = yearly_visits.sort_values('Year')

    fig_year = px.line(
        yearly_visits,
        x='Year',
        y='VisitCount',
        markers=True,
        title="📅 Yearly Visit Trends",
        labels={'VisitCount': 'Number of Visits'},
        width=500,      # Width in pixels
        height=450      # Height in pixels
    )
 

#monthly pattern
# Aggregate visit counts per month
    monthly_visits = df['VisitMonth'].value_counts().reset_index()
    monthly_visits.columns = ['Month', 'VisitCount']
    monthly_visits = monthly_visits.sort_values('Month')  # Ensure Jan–Dec order

# Optional: month names
    month_labels = {
       1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
       7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    monthly_visits['MonthName'] = monthly_visits['Month'].map(month_labels)

    fig_month = px.bar(monthly_visits,
                   x='MonthName', y='VisitCount',
                   title="📆 Monthly Visit Patterns",
                   labels={'VisitCount': 'Number of Visits', 'MonthName': 'Month'},
                   color='VisitCount',
                   color_continuous_scale='sunset')
    

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_year, use_container_width=True)
    with col2:
        st.plotly_chart(fig_month, use_container_width=True)


if page == "💬 Attraction Ratings Analysis":
   
    st.subheader("💬 Attraction Ratings Analysis")
    st.markdown("Understand rating distribution and user satisfaction levels across locations")
    
    fig_rating_dist = px.histogram(
       df, 
       x="Rating",
       nbins=10,
       title="📊 Distribution of Attraction Ratings",
       color_discrete_sequence=["teal"],
       labels={"Rating": "Rating Score"},
       width=300,    
       height=450 
    )
    
    # Average Rating by Country or City    

    avg_rating_country = df.groupby("Country")["Rating"].mean().reset_index()
    avg_rating_country = avg_rating_country.sort_values(by="Rating", ascending=False).head(25)

    fig_avg_country = px.bar(
       avg_rating_country, 
       x="Rating", 
       y="Country", 
       orientation="h",
       title="🌍 Average Rating by Country",
       color="Rating",
       color_continuous_scale="viridis",
       labels={"Rating": "Average Rating", "Country": "Country"}
    )
    

    col1, col2 = st.columns(2)
    with col1:
      st.plotly_chart(fig_rating_dist, use_container_width=True)
    with col2:
      st.plotly_chart(fig_avg_country, use_container_width=True)


#Top-Rated Attractions
    top_attractions = df.groupby(['Attraction', 'CityName'])['Rating'].mean().reset_index()
    top_attractions = top_attractions.sort_values(by='Rating', ascending=False).head(25)

    fig_top_attr = px.bar(
       top_attractions, 
       x='Rating', 
       y='Attraction', 
       orientation='h',
       color='Rating',
       color_continuous_scale='Viridis',
       title='🏆 Top-Rated Attractions',
       labels={'Rating': 'Average Rating', 'Attraction': 'Attraction'},
       hover_data={'CityName': True} 
    )
    st.plotly_chart(fig_top_attr, use_container_width=True)

if page == "🧑‍🤝‍🧑 User Behavior Analysis":

    st.subheader("🧑‍🤝‍🧑 User Behavior Analysis")
    st.markdown("Understand user diversity and engagement patterns")

#Distribution of visit counts per user
    
    visits_per_user = df.groupby('UserId')['AttractionId'].count().reset_index(name='VisitCount')

    fig_visits = px.histogram(
       visits_per_user, 
       x='VisitCount',
       nbins=50,
       title="📊 Distribution of visit counts per user",
       labels={'VisitCount': 'Number of Visits'},
       color_discrete_sequence=["indianred"]
    )
    

    # Group and filter for repeat visits
    repeat_visits = df.groupby(['UserId', 'Attraction'])['TransactionId'].count().reset_index()
    repeat_visits.columns = ['UserId', 'Attraction', 'VisitCount']
    repeat_visits = repeat_visits[repeat_visits['VisitCount'] > 1]

# Sort by most visited attractions
    top_repeats = repeat_visits.groupby('Attraction')['VisitCount'].sum().reset_index()
    top_repeats = top_repeats.sort_values(by='VisitCount', ascending=False).head(10)

 
    fig_repeat = px.bar(
        top_repeats,
        x='VisitCount',
        y='Attraction',
        orientation='h',
        title='🔄 Most Revisited Attractions',
        color='VisitCount',
        color_continuous_scale='Tealgrn',
        labels={'VisitCount': 'Repeat Visits'}
    )
   
    col1, col2 = st.columns(2)
    with col1:
      st.plotly_chart(fig_visits, use_container_width=True)
    with col2:
      st.plotly_chart(fig_repeat, use_container_width=True)

# --- 1. Avg Rating per User ---
    avg_rating = df.groupby('UserId')['Rating'].mean().reset_index(name='AvgRating')
    fig_avg = px.histogram(
       avg_rating,
       x='AvgRating',
       nbins=20,
       title="⭐ Avg Rating per User",
       color_discrete_sequence=['royalblue']
    )
    fig_avg.update_layout(xaxis_title="Avg Rating", yaxis_title="Users")

# --- 2. Variance per User ---
    var_rating = df.groupby('UserId')['Rating'].var().reset_index(name='RatingVariance').dropna()
    fig_var = px.histogram(
       var_rating,
       x='RatingVariance',
       nbins=20,
       title="📉 Rating Variance per User",
       color_discrete_sequence=['indianred']
    )
    fig_var.update_layout(xaxis_title="Variance", yaxis_title="Users")

# --- 3. Rating Trend Over Years ---
    trend_df = df.groupby('VisitYear')['Rating'].mean().reset_index(name='AvgRating')
    fig_trend = px.line(
      trend_df,
      x='VisitYear',
      y='AvgRating',
      title="📅 Yearly Rating Trend",
      markers=True,
      line_shape='linear'
    )
    fig_trend.update_layout(xaxis_title="Year", yaxis_title="Avg Rating")

# --- Layout: Side-by-side ---
    col1, col2 = st.columns(2)
    with col1:
       st.plotly_chart(fig_avg, use_container_width=True)
    with col2:
       st.plotly_chart(fig_var, use_container_width=True)
    
# Group by Region and VisitMode
    visitmode_region = (
        df.groupby(['Region', 'VisitMode'])['TransactionId']
        .count()
        .reset_index()
        .rename(columns={'TransactionId': 'VisitCount'})
    )

# Plot using grouped bar chart
    fig = px.bar(
       visitmode_region,
       x='Region',
       y='VisitCount',
       color='VisitMode',
       barmode='group',
       title='🚗✈️Visit Frequency by Visit Mode per Region',
       labels={'VisitCount': 'Number of Visits'},
       height=500
    )
    
    col1, col2 = st.columns(2)
    with col1:
      st.plotly_chart(fig, use_container_width=True)
    with col2:
      st.plotly_chart(fig_trend, use_container_width=True)


    st.subheader("📋 Repeat Visits (User-Attraction Level)")
    st.dataframe(repeat_visits.sort_values(by='VisitCount', ascending=False))
 
if page == "🧭 Exploring Attraction Categories":
    st.subheader("🔍 Understanding Tourist Preferences by Type")

 # --- Top Filters ---
    col1, col2 = st.columns(2)

    with col1:
        years = ['All'] + sorted(df['VisitYear'].dropna().unique().tolist())
        selected_year = st.selectbox("Select Year", years, key="year_filter")

    with col2:
        countries = ['All'] + sorted(df['Country'].dropna().unique().tolist())
        selected_country = st.selectbox("Select Country", countries, key="country_filter")

# Filter data based on selection
    filtered_df = df.copy()
    if selected_year != 'All':
      filtered_df = filtered_df[filtered_df['VisitYear'] == selected_year]
    if selected_country != 'All':
      filtered_df = filtered_df[filtered_df['Country'] == selected_country]
# --- 1. Popularity: Count of Each Attraction Type ---
    type_counts = filtered_df['AttractionType'].value_counts().reset_index()
    type_counts.columns = ['AttractionType', 'Count']
# --- 2. Average Rating per Type & Top City in Hover ---
    type_rating = (
        filtered_df.groupby('AttractionType')
        .agg(
           Rating=('Rating', 'mean'),
           TopCity=('CityName', lambda x: x.value_counts().index[0])  # most common city
        )
        .reset_index()
        .sort_values(by='Rating', ascending=True)
    )
# --- 🎯 Count Plot ---
    fig_count = px.bar(
        type_counts.sort_values(by='Count', ascending=True),
        x='Count', y='AttractionType',
        orientation='h',
        title="🎯 Popularity of Attraction Types",
        color='Count',
        color_continuous_scale='tealgrn',
        labels={'AttractionType': 'Attraction Type'},
    )
# --- ⭐ Rating Plot with Top City Hover ---
    fig_rating = px.bar(
        type_rating,
        x='Rating', y='AttractionType',
        orientation='h',
        color='Rating',
        color_continuous_scale='sunsetdark',
        title="⭐ Average Rating by Attraction Type",
        hover_data=['TopCity'],
        labels={'TopCity': 'Most Visited City'}
    )
# --- 🥧 Pie Chart (optional) ---
    fig_pie = px.pie(
       type_counts,
       values='Count',
       names='AttractionType',
       title="🥧 Distribution of Attraction Types",
       color_discrete_sequence=px.colors.qualitative.Safe
    )

      # --- 📑 Layout in Tabs ---
    tab1, tab2 = st.tabs(["📊 Bar Charts", "🥧 Pie Chart"])

    # Tab 1: Bar Charts Side-by-Side
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_count, use_container_width=True)
        with col2:
            st.plotly_chart(fig_rating, use_container_width=True)

    # Tab 2: Pie Chart
    with tab2:
        st.plotly_chart(fig_pie, use_container_width=True)

   
if page == "🗓️ Predictor":
   
    predictor_option = st.sidebar.radio(
        "Choose Prediction Type",
        ["📈 Regression", "📊 Classification", "🎯 Recommendation"]
    )
    
    if predictor_option == "📈 Regression":
        st.subheader("📈 Predict Rating with Regression")

    # --- Load Pickled CatBoost Regression Model ---
        @st.cache_resource
        def load_regression_model():
            with open("catboost_model_regression.pkl", "rb") as f:
               return pickle.load(f)

        model = load_regression_model()
        st.markdown("**📋 Input Features for Prediction**") 
    # --- Input Fields ---
        col1, col2= st.columns(2)
        with col1:
            continent_id = st.number_input("ContinentId", min_value=0, value=1)
            city_id = st.number_input("CityId", min_value=0, value=1)
            avg_user_rating = st.slider("Avg_user_rating", 0.0, 5.0, 3.0)
            attraction_avg_rating = st.slider("Attraction_avg_rating", 0.0, 5.0, 4.0)

        with col2:
            country_id = st.number_input("CountryId", min_value=0, value=1)
            attraction_type_id = st.number_input("AttractionTypeId", min_value=0, value=1)
            user_visit_count = st.slider("User_visit_count", 0, 100, 10)
            attraction_popularity = st.slider("Attraction_popularity", 0, 10000, 500)

    # --- Predict ---
        if st.button("🎯 Predict Rating"):
            input_df = pd.DataFrame([{
                'ContinentId': continent_id,
                'CountryId': country_id,
                'CityId': city_id,
                'AttractionTypeId': attraction_type_id,
                'Avg_user_rating': avg_user_rating,
                'User_visit_count': user_visit_count,
                'Attraction_avg_rating': attraction_avg_rating,
                'Attraction_popularity': attraction_popularity
            }])

            prediction = model.predict(input_df)[0]

            st.subheader("🎯 Predicted Rating")
            st.success(f"Predicted Rating: {prediction:.2f}")

    
    elif predictor_option == "📊 Classification":
        st.subheader("📊 Visit Mode Classification")
# --- Load Pickled Model ---
        @st.cache_resource
        def load_model():
            with open("catboost_model_classification.pkl", "rb") as f:
              model = pickle.load(f)
            return model

        model = load_model()
        st.markdown("**📋 Input Features for Prediction**")  # bold but regular size
        # --- Input Fields ---
        col1, col2= st.columns(2)
        with col1:
            country_id = st.number_input("CountryId", min_value=1, value=1)
            city_id = st.number_input("CityId", min_value=1, value=1)
            attraction_id = st.number_input("AttractionId", min_value=1, value=1)
            attraction_type = st.number_input("AttractionTypeId", min_value=1, value=1)
            attraction_city = st.number_input("AttractionCityId", min_value=1, value=1)

        with col2:
            user_rating = st.slider("Avg_user_rating", min_value=0.0, max_value=5.0, value=3.0)
            user_visit_count = st.slider("User_visit_count", min_value=0, max_value=100, value=5)
            rating = st.slider("Rating", min_value=0, max_value=5, value=3)
            popularity = st.slider("Attraction_popularity", min_value=0, max_value=1000, value=50)


# --- Create DataFrame for Prediction ---
        input_df = pd.DataFrame([{
          'CountryId': country_id,
          'CityId': city_id,
          'AttractionId': attraction_id,
          'AttractionTypeId': attraction_type,
          'AttractionCityId': attraction_city,
          'Rating': rating,
          'User_visit_count': user_visit_count,
          'Avg_user_rating': user_rating,
          'Attraction_popularity': popularity
        }])

# --- Predict ---
        if st.button("🎯 Predict Visit Mode"):
          prediction = model.predict(input_df)[0]
          probs = model.predict_proba(input_df)

          st.subheader("📌 Prediction Result")
          st.write(f"**Predicted VisitModeId:** {prediction}")
    
          st.subheader("🔍 Class Probabilities")
          prob_df = pd.DataFrame(probs, columns=[f"Mode {i}" for i in model.classes_])
          st.dataframe(prob_df.T.rename(columns={0: "Probability"}))

          st.info("Prediction completed using CatBoost Classifier.")

          

    elif predictor_option == "🎯 Recommendation":
        st.header("🎯 Personalized Attraction Recommendations")
        tab1, tab2, tab3 = st.tabs(["📚 Content-Based", "🤝 Collaborative", "🔀 Hybrid"])

    # -------------------- TAB 1: CONTENT-BASED --------------------
        with tab1:
            st.subheader("📚 Content-Based Recommendations")
        
            tourism = df.copy()
            tourism.drop_duplicates(
               subset=['Attraction', 'CityName', 'Country', 'AttractionType', 'Rating'], inplace=True
            )
            tourism.reset_index(drop=True, inplace=True)

            tourism['Attraction_recommendation'] = (
                tourism['Attraction'].astype(str) + ' ' +
                tourism['AttractionType'].astype(str) + ' ' +
                tourism['CityName'].astype(str) + ' ' +
                tourism['Country'].astype(str) + ' ' +
                tourism['Rating'].astype(str)
            )

            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(tourism['Attraction_recommendation'])

            def build_user_profile(user_id, df=tourism, tfidf_matrix=tfidf_matrix):
                visited = df[df['UserId'] == user_id]
                if visited.empty:
                    return None, "User has not visited any attractions."
                user_indices = visited.index.tolist()
                user_profile_vector = tfidf_matrix[user_indices].mean(axis=0)
                return user_profile_vector, visited

            def recommend_for_user(user_id, df=tourism, tfidf_matrix=tfidf_matrix, top_n=5):
                user_profile_vector, visited = build_user_profile(user_id, df, tfidf_matrix)
                if user_profile_vector is None:
                    return visited
                sim_scores = cosine_similarity(np.asarray(user_profile_vector), tfidf_matrix).flatten()
                visited_indices = set(visited.index)
                recommendations = [
                    (i, score) for i, score in enumerate(sim_scores) if i not in visited_indices
                ]
                recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
                top_indices = [i[0] for i in recommendations]
                return df[['Attraction', 'CityName', 'Country', 'AttractionType', 'Rating']].iloc[top_indices]

            user_ids = tourism['UserId'].dropna().unique().astype(int)
            selected_user = st.selectbox("Select User ID", sorted(user_ids), key="cb_user")

            if st.button("🎯 Show Recommendations", key="cb_btn"):
                result = recommend_for_user(selected_user)
                if isinstance(result, str):
                    st.warning(result)
                else:
                    st.subheader("📌 Recommended Attractions")
                    st.dataframe(result)

    # -------------------- TAB 2: COLLABORATIVE --------------------
        with tab2:
            st.subheader("🤝 Collaborative Filtering")

            collab_type = st.radio("Choose Collaborative Method", ["Item-Based", "User-Based"], horizontal=True)


            if collab_type == "Item-Based":
                st.markdown("### 🧩 Item-Based Collaborative Filtering")

                reader = Reader(rating_scale=(1, 5))
                data = Dataset.load_from_df(df[['UserId', 'AttractionId', 'Rating']], reader)
                trainset = data.build_full_trainset()
                algo = SVD()
                algo.fit(trainset)

                raw_to_inner = trainset.to_inner_iid
                valid_ids = [i for i in df['AttractionId'].unique() if raw_to_inner(i) < len(algo.qi)]
                item_factors = np.array([algo.qi[raw_to_inner(i)] for i in valid_ids])

                item_sim = cosine_similarity(item_factors)
                sim_df = pd.DataFrame(item_sim, index=valid_ids, columns=valid_ids)

                def get_similar_attractions(attraction_id, top_n=5):
                    if attraction_id not in sim_df.index:
                        return pd.DataFrame()
                    scores = sim_df[attraction_id].sort_values(ascending=False)[1:top_n + 1]
                    return pd.DataFrame({
                        "AttractionId": scores.index,
                        "Attraction": [df[df['AttractionId'] == i]['Attraction'].iloc[0] for i in scores.index],
                        "Similarity Score": scores.values
                    })

                attraction_options = df[['AttractionId', 'Attraction']].drop_duplicates().sort_values(by='Attraction')
                attraction_dict = dict(zip(attraction_options['Attraction'], attraction_options['AttractionId']))
                selected_attraction_name = st.selectbox("Select an Attraction", list(attraction_dict.keys()))
                selected_attraction_id = attraction_dict[selected_attraction_name]
                top_n = st.slider("Top N Similar Attractions", 1, 10, 5)

                similar_df = get_similar_attractions(selected_attraction_id, top_n=top_n)
                if not similar_df.empty:
                    st.subheader(f"🔍 Attractions similar to **{selected_attraction_name}**")
                    st.dataframe(similar_df)
                else:
                    st.warning("⚠️ No similar attractions found.")

            elif collab_type == "User-Based":
                st.markdown("### 👥 User-Based Collaborative Filtering")

                df['UserId'] = df['UserId'].astype(str)
                df['AttractionId'] = df['AttractionId'].astype(str)
                reader = Reader(rating_scale=(1, 5))
                data = Dataset.load_from_df(df[['UserId', 'AttractionId', 'Rating']], reader)
                trainset, testset = train_test_split(data, test_size=0.2)

                algo = SVD()
                algo.fit(trainset)

                user_ids = df['UserId'].unique()
                selected_user = st.selectbox("Select User ID", sorted(user_ids), key="user_cf")
                top_n = st.slider("Top N Recommendations", 1, 10, 5, key="user_cf_topn")

                all_attractions = df['AttractionId'].unique()
                seen = df[df['UserId'] == selected_user]['AttractionId'].tolist()
                predictions = [algo.predict(selected_user, aid) for aid in all_attractions if aid not in seen]
                top_preds = sorted(predictions, key=lambda x: x.est, reverse=True)[:top_n]

                recommendations_df = pd.DataFrame({
                    "AttractionId": [pred.iid for pred in top_preds],
                    "Attraction": [
                        df[df['AttractionId'] == pred.iid]['Attraction'].iloc[0] if not df[df['AttractionId'] == pred.iid].empty else "Unknown"
                        for pred in top_preds
                    ],
                    "Predicted Rating": [round(pred.est, 2) for pred in top_preds]
                })

                st.subheader(f"🎉 Recommendations for User {selected_user}")
                st.dataframe(recommendations_df)
 
    # -------------------- TAB 3: HYBRID --------------------
        with tab3:
            st.subheader("🔀 Hybrid Recommendation System")

            df['Rating'] = df['Rating'].astype(str)
            tourism = df.sample(n=3000, random_state=42).drop_duplicates(
                subset=['Attraction', 'CityName', 'Country', 'AttractionType', 'Rating']
            ).reset_index(drop=True)

            tourism['Attraction_recommendation'] = (
                tourism['Attraction'] + ' ' +
                tourism['AttractionType'] + ' ' +
                tourism['CityName'] + ' ' +
                tourism['Country'] + ' ' +
                tourism['Rating']
            )
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(tourism['Attraction_recommendation'])
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(df[['UserId', 'AttractionId', 'Rating']].astype(float), reader)
            trainset = data.build_full_trainset()
            svd_model = SVD()
            svd_model.fit(trainset)

            def hybrid_recommend(user_id, liked_attraction, city_name, top_n=5, alpha=0.5):
                idx = tourism[(tourism['Attraction'] == liked_attraction) & (tourism['CityName'] == city_name)].index
                if idx.empty:
                    return pd.DataFrame(), f"⚠️ Attraction '{liked_attraction}' in '{city_name}' not found."
                idx = idx[0]
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:50]
                candidate_indices = [i[0] for i in sim_scores]
                hybrid_scores = []
                for i in candidate_indices:
                    attraction_id = tourism.loc[i, 'AttractionId']
                    pred = svd_model.predict(str(user_id), str(attraction_id)).est
                    content_score = cosine_sim[idx][i]
                    hybrid_score = alpha * pred + (1 - alpha) * content_score
                    hybrid_scores.append((i, hybrid_score))
                hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:top_n]
                top_indices = [i[0] for i in hybrid_scores]
                return tourism.loc[top_indices, ['Attraction', 'CityName', 'Country', 'AttractionType', 'Rating']], None
  
            st.subheader("🎯 Hybrid Recommendation Inputs")

# Create two rows of two columns each
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

# Row 1: User ID & Attraction
            with col1:
                user_ids = df['UserId'].astype(int).unique()
                selected_user = st.selectbox("Select User ID", sorted(user_ids), key="hy_user")

            with col2:
                attractions = tourism[['Attraction', 'CityName']].drop_duplicates().sort_values(by='Attraction')
                selected_attr = st.selectbox("Select Attraction", attractions['Attraction'].unique(), key="hy_attr")

# Row 2: City & Alpha
            with col3:
                filtered_cities = attractions[attractions['Attraction'] == selected_attr]['CityName'].unique()
                selected_city = st.selectbox("Select City", filtered_cities, key="hy_city")

            with col4:
                alpha_val = st.slider("Blending α (0=Content, 1=Collaborative)", 0.0, 1.0, 0.5, key="hy_alpha")


            if st.button("Get Hybrid Recommendations"):
                result_df, error = hybrid_recommend(selected_user, selected_attr, selected_city, top_n=5, alpha=alpha_val)
                if error:
                    st.warning(error)
                else:
                    st.success(f"Top Hybrid Recommendations for User {selected_user}")
                    st.dataframe(result_df)

elif page == "🧾Detailed Report":
       
    st.title("📝 Tourism Experience Analytics: Classification, Prediction, and Recommendation System")

# Executive Summary
    st.header("1. Executive Summary")
    st.write("""
        This report presents a comprehensive analysis of tourism data, encompassing geographic insights, time-based trends, attraction ratings, user behaviour, attraction categories, and the performance of predictive and recommendation models.
        Key findings highlight the dominance of nature and relaxation-focused attractions, the impact of the COVID-19 pandemic on visit trends, and the potential for strategic interventions to optimize visitor experiences and marketing efforts.
    """)

# Geographic Insights
    st.header("2. Geographic Insights")
    st.markdown("""
        - **Top Visited Attractions**: Cities such as Singapore, Melbourne, London, and Perth show high visit volumes.
        - **Country-wise Attraction Count**: Indonesia, Singapore, Thailand, and the US lead, especially in Southeast Asia.
        - **Bottom Countries by Attraction Count**: Countries like Azerbaijan and Burkina Faso have fewer attractions.
        - **Attraction Density**: High in North America, Europe, Southeast Asia; lower in parts of Africa, Central Asia.
        - **Cities with Most Attractions**: Jakarta, Singapore, Kuala Lumpur, Bali (25+ attractions).
        - **Cities with Fewest Attractions**: Aachen, Miri, Misina show 1–1.5 attractions — possible emerging markets.
    """)

# Time-Based Analysis
    st.header("3. Time-Based Analysis")
    st.markdown("""
        - **Monthly Visit Patterns**:
            - Peak: July–August (summer season).
            - Off-peak: February and November.
        - **Yearly Visit Trends**:
            - 2013–2016: Rapid growth.
            - 2017–2019: Gradual decline.
            - 2020: COVID impact.
            - 2021–2022: Modest recovery.
    """)

# Attraction Ratings
    st.header("4. Attraction Ratings Analysis")
    st.markdown("""
        - **Top-Rated Attractions**: Sacred Monkey Forest Sanctuary, Tegenungan Waterfall.
        - **Hidden Gems**: Balekambang Beach (high rating, few reviews).
        - **Rating by Country**: Highest - Afghanistan, Uzbekistan; Lowest - Yemen, Moldova.
        - **Recommendations**:
           - Promote high-rated but less visited places.
           - Improve infrastructure in low-rated regions.
           - Support visibility of well-rated but unknown spots.
    """)

# User Behavior
    st.header("5. User Behaviour Analysis")
    st.markdown("""
        - **Yearly Rating Trend**: Stable (~4.2) till 2017, peaked in 2021 (>4.5).
        - **Visit Modes**: Australia leads Family & Business; Southeast Asia strong for Couples.
        - **Rating Variance**: Mostly low, suggesting consistent user behavior.
        - **Visit Frequency**: Mostly 1–2 visits per user.
        - **Repeat Visits**: Merapi Volcano (49x by one user), Sacred Monkey Forest (~3000 revisits).
    """)

# Attraction Categories
    st.header("6. Attraction Categories Analysis")
    st.markdown("""
        - **Top Types**: Nature & Wildlife (25%), Beaches (20.6%), Religious Sites.
        - **Top-Rated Types**: Water Parks, Spas, Caves.
        - **Lowest-Rated Types**: Beaches, Historic Sites.
        - **Insights**:
          - Improve underperforming but popular sites.
          - Promote high-rated, under-visited types like Spas and Caverns.
    """)

# Strategic Recommendations
    st.header("7. Strategic Insights and Recommendations")
    st.dataframe({
        "Aspect": ["Popularity", "User Rating", "Distribution"],
        "High": ["Nature, Beaches, Religious Sites", "Water Parks, Spas, Caverns", "Nature, Beaches"],
        "Medium": ["Water Parks, Volcanoes", "Volcanoes, Museums", "Water Parks, Landmarks"],
        "Low": ["Spas, Museums", "Historic Sites, Beaches", "Ballets, Ruins"]
    })

    st.markdown("""
        **Recommendations:**
        - Promote highly rated but under-visited attractions (e.g., Spas, Caverns).
        - Improve tourist experience at Beaches and Historic Sites.
        - Balance investment between popular and niche attraction types.
    """)

# Predictive Systems
    st.header("8. Predictive and Recommendation Systems")

    st.subheader("🔍 Rating Prediction (CatBoost Regression)")
    st.markdown("""
        - **MSE**: 0.23
        - **R² Score**: 0.75
        - **Interpretation**: Accurately predicts 75% of the variance in user ratings.
    """)

    st.subheader("🧭 Visit Mode Classification (CatBoost Classifier)")
    st.markdown("""
        - **Accuracy**: 45.25%
        - **Note**: Good for majority classes, but struggles with underrepresented classes due to imbalance.
    """)

    st.subheader("🎯 Recommendation Systems")
    st.markdown("""
        - **Content-Based Filtering (CBF)**:
          - Uses metadata like AttractionType, City, Country, Rating.
          - Method: TF-IDF + Cosine Similarity.

        - **Collaborative Filtering (CF)**:
          - **Item-Based**: SVD learns item embeddings, computes similarity for recommendations.
          - **User-Based**: Predicts unseen items for each user based on past ratings.

        - **Hybrid Approach**:
          Combines CF and CBF:
          `Hybrid Score = α × Predicted Rating + (1 - α) × Content Similarity`
    """)

# Footer
    st.info("This report provides a holistic view of global tourism behavior and offers strategic insights for experience optimization, marketing, and personalization.")

































