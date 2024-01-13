import streamlit as st
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Function to label encode columns with handling of unseen labels
def label_encode_columns(data, columns_to_encode):
    label_encoder = preprocessing.LabelEncoder()
    for column in columns_to_encode:
        if column != 'Accident_severity':
            label_encoder.fit(data[column])
            data[column] = label_encoder.transform(data[column])

# Load the data
data = pd.read_csv('./Fatality_app/Road.csv/Road.csv')
data.dropna(inplace=True)

# Modifying Time to required format
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S')
data['hour'] = data['Time'].dt.hour
data['minute'] = data['Time'].dt.minute
data = data.drop(['Time', 'Educational_level', 'Vehicle_driver_relation', 'Owner_of_vehicle',
                  'Service_year_of_vehicle', 'Number_of_casualties', 'Work_of_casuality'], axis=1)

# Label encode the categorical columns
columns_to_encode = [
    'Day_of_week', 'Age_band_of_driver', 'Sex_of_driver', 'Driving_experience',
    'Type_of_vehicle', 'Defect_of_vehicle', 'Area_accident_occured',
    'Lanes_or_Medians', 'Road_allignment', 'Types_of_Junction',
    'Road_surface_type', 'Road_surface_conditions', 'Light_conditions',
    'Weather_conditions', 'Type_of_collision', 'Number_of_vehicles_involved',
    'Vehicle_movement', 'Casualty_class', 'Sex_of_casualty',
    'Age_band_of_casualty', 'Casualty_severity', 'Fitness_of_casuality',
    'Pedestrian_movement', 'Cause_of_accident', 'Accident_severity'
]

label_encode_columns(data, columns_to_encode)

# Create X and Y
X = data.drop(['Accident_severity'], axis=1)
Y = data['Accident_severity']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=13)

# Train RandomForestClassifier
RForest_clf = RandomForestClassifier(n_estimators=100)
RForest_clf.fit(X_train, y_train)

# Train Support Vector Machine (SVM)
model = SVC()
model.fit(X_train, y_train)

# Streamlit app
def main():
    st.title("Fatality Prediction App")

    # Display preprocessing steps
    st.header("Preprocessing Steps")
    st.subheader("Original Data")
    st.write(data.head())

    st.subheader("Label Encoded Data")
    label_encode_columns(data, columns_to_encode)
    st.write(data.head())

    # User input for prediction
    st.header("User Input for Prediction")
    user_input_df = get_user_input()
    st.write(user_input_df)

    # Button to trigger prediction
    if st.button("Predict"):
        # Apply label encoding to the user input
        label_encode_columns(user_input_df, columns_to_encode)

        # Predict using RandomForestClassifier
        rf_prediction = RForest_clf.predict(user_input_df)
        st.subheader("Random Forest Prediction")
        st.write(f'Prediction: {rf_prediction[0]}')

        # Predict using Support Vector Machine (SVM)
        svm_prediction = model.predict(user_input_df)
        st.subheader("SVM Prediction")
        st.write(f'Prediction: {svm_prediction[0]}')

def get_user_input():
    user_input = {}
    for column in columns_to_encode:
        if column != 'Accident_severity':
            user_input_str = st.text_input(f"Enter {column}:")
            if user_input_str:
                user_input[column] = int(user_input_str)
            else:
                # Handle the case where the input is empty
                user_input[column] = 0

    # Include 'hour' and 'minute' in user input
    user_input['hour'] = st.slider("Select hour:", 0, 23, 12)
    user_input['minute'] = st.slider("Select minute:", 0, 59, 30)

    # Convert the user input to a DataFrame
    user_input_df = pd.DataFrame([user_input])

    return user_input_df

if __name__ == "__main__":
    main()
