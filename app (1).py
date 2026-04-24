import streamlit as st
import pandas as pd
import joblib
st.set_page_config(page_title="Student Details", layout="centered")

st.title("Student Details")
st.write("Enter the total result.")

try:
    # 2. Load the trained model
    model = joblib.load('Student Details.pkl')

    # 3. Create a Layout with Columns for User Input
    col1, col2, col3 = st.columns(3)

    with col1:
        GRE_Score = st.number_input("GRE_Score", min_value=0.0, max_value=1000.0, value=10.0)

    with col2:
        TOEFL_Score = st.number_input("TOEFL_Score", min_value=0.0, max_value=100.0, value=37.8)

    with col3:
        University_Rating = st.number_input("University_Rating", min_value=0.0, max_value=200.0, value=69.2)

    # 4. Create a 'Predict' button
    if st.button("Calculate Prediction"):
        # Create a DataFrame from the dynamic user input
        user_input = pd.DataFrame([{
            'GRE Score': GRE_Score,
            'TOEFL Score': TOEFL_Score,
            'University Rating': University_Rating
        }])

        # Get prediction
        prediction = model.predict(user_input)

        # 5. Display Result in a nice box
        st.divider()
        st.subheader("Results")
        st.metric(label="Student Details", value=f"{prediction[0]:.2f}")

        # Show how the input compares
        st.bar_chart(user_input.T)

except Exception as e:
    st.error(f"Model Error: {e}")
