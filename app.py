import streamlit as st
import pandas as pd
import joblib

st.title("Iris Species Predictor")

with st.sidebar:
    st.header('Data requirements')
    st.caption('To inference the model you need to upload a dataframe in csv format with four columns/features (columns names are not important)')
    with st.expander('Data format'):
        st.markdown(' - utf-8')
        st.markdown(' - separated by coma')
        st.markdown(' - delimited by "."')
        st.markdown(' - first row - header')       
    st.divider() 
    st.caption("<p style = 'text-align:center'>Developed by Anas Akhssas</p>", unsafe_allow_html = True)

if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click = clicked, args = [1])
if st.session_state.clicked[1]:
    uploaded_file = st.file_uploader("Choose a file", type='csv')
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, low_memory=True)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
        else:
            st.header('Uploaded data sample')
            st.write(df.head())
            try:
                with st.spinner('Loading model and generating predictions...'):
                    model = joblib.load('model.joblib')

                    # Attempt to align input columns with model expectations to avoid sklearn feature name warning
                    X_input = df
                    if hasattr(model, 'feature_names_in_'):
                        missing = [c for c in model.feature_names_in_ if c not in df.columns]
                        extra = [c for c in df.columns if c not in model.feature_names_in_]
                        if missing:
                            st.warning(f"Uploaded data is missing expected columns: {missing}. Prediction may fail.")
                        if extra:
                            st.info(f"Ignoring extra columns: {extra}")
                        # Reorder & subset
                        present_features = [c for c in model.feature_names_in_ if c in df.columns]
                        X_input = df[present_features]
                    else:
                        # Model trained without feature names; fall back to numpy values to suppress warning
                        X_input = df.values

                    pred = model.predict_proba(X_input)
                    pred = pd.DataFrame(pred, columns=['setosa_probability', 'versicolor_probability', 'virginica_probability'])
            except FileNotFoundError:
                st.error('model.joblib not found in the app directory.')
            except Exception as e:
                st.error(f'Prediction failed: {e}')
            else:
                st.header('Predicted values (first 5 rows)')
                st.write(pred.head())
                pred_bytes = pred.to_csv(index=False).encode('utf-8')
                st.download_button(
                    'Download full prediction CSV',
                    pred_bytes,
                    'prediction.csv',
                    'text/csv',
                    key='download-csv'
                )

