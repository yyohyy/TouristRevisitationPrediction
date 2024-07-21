import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the saved model 
model_path = "model/random_forest_model.pkl"
model = joblib.load(model_path)

categorical_features = ['Gender', 'Region', 'Season_of_Visit', 'Type_of_Accommodation',
       'Purpose_of_Visit', 'Weather_Conditions',
       'Participation_in_Local_Events', 'Interaction_with_Locals']

numerical_features = ['Age', 'Travel_Group_Size', 'Duration_of_Stay',
       'Number_of_Previous_Visits', 'Spending_per_Day',
       'Distance_from_Home_Country', 'Satisfaction_Rating',
       'Major_Attractions_Visited', 'Recommendation_Likelihood',
       'Social_Media_Shares']

expected_columns = [
       'Gender_Female', 'Gender_Male', 'Gender_Other', 'Region_Asian-Others',
       'Region_Asian-SAARC', 'Region_Europe', 'Region_North American',
       'Region_Oceania', 'Region_South American', 'Season_of_Visit_Fall',
       'Season_of_Visit_Monsoon', 'Season_of_Visit_Spring',
       'Season_of_Visit_Summer', 'Season_of_Visit_Winter',
       'Type_of_Accommodation_HomeStay', 'Type_of_Accommodation_Hostel',
       'Type_of_Accommodation_Hotel', 'Type_of_Accommodation_Resort',
       'Purpose_of_Visit_Business', 'Purpose_of_Visit_Education',
       'Purpose_of_Visit_Family', 'Purpose_of_Visit_Tourism',
       'Weather_Conditions_Cloudy', 'Weather_Conditions_Rainy',
       'Weather_Conditions_Snowy', 'Weather_Conditions_Sunny',
       'Participation_in_Local_Events_No', 'Participation_in_Local_Events_Yes',
       'Age', 'Travel_Group_Size',
       'Duration_of_Stay', 'Number_of_Previous_Visits', 'Spending_per_Day',
       'Distance_from_Home_Country', 'Satisfaction_Rating',
       'Major_Attractions_Visited', 'Interaction_with_Locals',
       'Recommendation_Likelihood', 'Social_Media_Shares'
]

# Define preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_features)
    ]
)

def preprocess_data(data):

    data_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=False)
    data_encoded = data_encoded.reindex(columns=expected_columns, fill_value=0)
    
    return data_encoded

def main():

    with st.container():

        with st.expander('Tourist Information Form', expanded=True):
            st.subheader('Enter Tourist Information')
            col1, col2 = st.columns(2)

            with col1:

                gender = st.selectbox("Gender", ['Female', 'Male', 'Other'])

            with col2:

                 region = st.selectbox("Region", ['Asian-Others', 'Asian-SAARC', 'Europe', 'North American', 'Oceania', 'South American'])
            age = st.slider('Age', 0, 100, 25)
            
            col1, col2 = st.columns(2)

            with col1:
            
                travel_group_size = st.number_input('Travel Group Size', min_value=1)
                duration_of_stay = st.number_input('Duration of Stay (days)', min_value=1)
                season_of_visit = st.selectbox('Season of Visit', ['Spring', 'Summer', 'Fall', 'Winter'])
                type_of_accommodation = st.selectbox('Type of Accommodation', ['HomeStay', 'Hostel', 'Hotel', 'Resort'])
            with col2:
                purpose_of_visit = st.selectbox('Purpose of Visit', ['Business', 'Education', 'Family', 'Tourism'])
                number_of_previous_visits = st.number_input('Number of Previous Visits', min_value=0)
                spending_per_day = st.number_input('Spending per Day', min_value=100.0)
                distance_from_home_country = st.number_input('Distance from Home Country (km)', min_value=630)
            satisfaction_rating = st.slider('Satisfaction Rating', 0, 10, 0)
            col1, col2 = st.columns(2)

            with col1:
                 weather_conditions = st.selectbox('Weather Conditions', ['Sunny', 'Cloudy', 'Rainy', 'Snowy'])
                 major_attractions_visited = st.number_input('Major Attractions Visited', min_value=1)
            with col2:
                participation_in_local_events = st.selectbox('Participation in Local Events', ['No', 'Yes'])
           
            # Mapping for Interaction with Locals
                interaction_map = {'Extensive': 1.0, 'Some': 0.5, 'None': 0.0}
                interaction_with_locals = st.selectbox('Interaction with Locals', list(interaction_map.keys()))
                interaction_value = interaction_map[interaction_with_locals]

            recommendation_likelihood = st.slider('Recommendation Likelihood', 0, 10, 0)
            social_media_shares = st.number_input('Social Media Shares', min_value=0)

            # Convert categorical inputs to numerical
            gender = {'Female': 1, 'Male': 0, 'Other': 2}.get(gender, 0)
            region = {'Asian-Others': 0, 'Asian-SAARC': 1, 'Europe': 2, 'North American': 3, 'Oceania': 4, 'South American': 5}.get(region, 0)
            season_of_visit = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}.get(season_of_visit, 0)
            type_of_accommodation = {'HomeStay': 0, 'Hostel': 1, 'Hotel': 2, 'Resort': 3}.get(type_of_accommodation, 0)
            purpose_of_visit = {'Business': 0, 'Education': 1, 'Family': 2, 'Tourism': 3}.get(purpose_of_visit, 0)
            weather_conditions = {'Sunny': 0, 'Cloudy': 1, 'Rainy': 2, 'Snowy': 3}.get(weather_conditions, 0)
            participation_in_local_events = 1 if participation_in_local_events == 'Yes' else 0

            # Prepare input data as a DataFrame
            input_data = pd.DataFrame({
                'Gender': [gender],
                'Region': [region],
                'Season_of_Visit': [season_of_visit],
                'Type_of_Accommodation': [type_of_accommodation],
                'Purpose_of_Visit': [purpose_of_visit],
                'Weather_Conditions': [weather_conditions],
                'Participation_in_Local_Events': [participation_in_local_events],
                'Interaction_with_Locals': [interaction_value],
                'Age': [age],
                'Travel_Group_Size': [travel_group_size],
                'Duration_of_Stay': [duration_of_stay],
                'Number_of_Previous_Visits': [number_of_previous_visits],
                'Spending_per_Day': [spending_per_day],
                'Distance_from_Home_Country': [distance_from_home_country],
                'Satisfaction_Rating': [satisfaction_rating],
                'Major_Attractions_Visited': [major_attractions_visited],
                'Recommendation_Likelihood': [recommendation_likelihood],
                'Social_Media_Shares': [social_media_shares]
            })

            # Preprocess the input data
            processed_input = preprocess_data(input_data)

    with st.container():
        st.markdown('<div class="centered-button-container">', unsafe_allow_html=True)
        if st.button('Predict'):
            
            prediction = model.predict(processed_input)
            probability = model.predict_proba(processed_input)[0][1]
                    
            st.write(f'The model predicts that the tourist will {"revisit" if prediction[0] == 1 else "not revisit"}.')
            st.write(f'Probability of Revisitation: {probability:.2f}')

            if prediction[0] == 1:
                st.success(f"The tourist is likely to revisit. Consider maintaining current services and engagement strategies.")
            else:
                st.error(f"The tourist is unlikely to revisit. Evaluate and improve the aspects that could enhance the experience.")

if __name__ == '__main__':
    main()
