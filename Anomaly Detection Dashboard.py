# Import necessary libraries
import streamlit as st #type: ignore
import pandas as pd #type: ignore
import numpy as np
import plotly.express as px #type: ignore
import altair as alt #type: ignore
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.metrics import classification_report #type: ignore


# Page configuration
st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("fivethirtyeight")

#-----------------Defining Functions-----------------
def calculate_bounds(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

def inject_anomalies(df, feature, lower, upper):
    n_anomalies = int(0.1 * len(df))  # 10% of the data
    anomalies_below = np.random.uniform(low=0, high=lower, size=n_anomalies // 2)
    anomalies_above = np.random.uniform(low=upper, high=14, size=n_anomalies // 2)
    anomalies = np.concatenate((anomalies_below, anomalies_above))
    if n_anomalies % 2 != 0:
        additional_anomaly = np.random.uniform(low=0, high=lower) if np.random.rand() < 0.5 else np.random.uniform(low=upper, high=14)
        anomalies = np.append(anomalies, additional_anomaly)
    np.random.shuffle(anomalies)
    return anomalies

def combined_data(df, ph_anomalies, temp_anomalies, turb_anomalies):
    n_anomalies = len(ph_anomalies)
    anomalies = pd.DataFrame({
        'ph': ph_anomalies,
        'temperature': temp_anomalies,
        'turbidity': turb_anomalies,
        'fish': ['anomaly'] * n_anomalies
    })
    data_with_anomalies = pd.concat([df, anomalies], ignore_index=True)
    return data_with_anomalies

def is_anomaly(row, ph_bounds, temp_bounds, turb_bounds):
    if (row['ph'] < ph_bounds[0] or row['ph'] > ph_bounds[1] or
        row['temperature'] < temp_bounds[0] or row['temperature'] > temp_bounds[1] or
        row['turbidity'] < turb_bounds[0] or row['turbidity'] > turb_bounds[1]):
        return 'anomaly'
    else:
        return 'tilapia'

def anomaly_detection_model(anomalous_data):
    X = anomalous_data[['ph', 'temperature', 'turbidity']]
    y = anomalous_data['fish'].apply(lambda x: 1 if x == 'tilapia' else 0)  # 1 for normal, 0 for anomalies

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    report = classification_report(y_test, y_pred_rf, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    return report_df

def matrix_average(df):
    avg_temp = df['temperature'].mean()
    avg_ph = df['ph'].mean()
    avg_turb = df['turbidity'].mean()
    
    st.markdown('#### Avg. Features')
    # Styling for the grey boxes
    box_style = (
        "background-color: #0000; "
        "padding: 20px; "
        "border-radius: 5px; "
        "box-shadow: 2px 2px 5px #888888;"
    )
    
    with st.markdown(f'<div style="{box_style}">'):
        st.metric(label="Avg. Temperature", value=f"{avg_temp:.2f}°C")
    
    with st.markdown(f'<div style="{box_style}">'):
        st.metric(label="Avg. pH", value=f"{avg_ph:.2f}")
 
    with st.markdown(f'<div style="{box_style}">'):
        st.metric(label="Avg. Turbidity", value=f"{avg_turb:.2f}")
 

def extract_accuracy_from_report(report_df):
    accuracy = report_df.loc['accuracy', 'f1-score']
    return accuracy

def make_donut_chart(values, labels, input_color='blue'):
    colors = {
        'blue': ['#29b5e8', '#155F7A'],
        'green': ['#27AE60', '#12783D'],
        'orange': ['#F39C12', '#875A12'],
        'red': ['#E74C3C', '#781F16']
    }.get(input_color, ['#29b5e8', '#155F7A'])
    
    source = pd.DataFrame({
        "Topic": labels,
        "% value": values
    })

    plot_bg = alt.Chart(source).mark_arc(innerRadius=65, cornerRadius=0).encode(
        theta="% value",
        color=alt.Color("Topic:N", scale=alt.Scale(domain=labels, range=colors)),
        tooltip=['% value']
    ).properties(width=200, height=200)

    text = plot_bg.mark_text(align='center', color="red", font="arial", fontSize=32, fontWeight=700, fontStyle="italic").encode(
        text=alt.value(f'{values[0]:.1f} %')
    )
    
    return plot_bg + text

def scatter_tempph(df, temp_bounds, ph_bounds):
    plt.figure(figsize=(10, 8))
    plt.scatter(df['temperature'], df['ph'], alpha=0.3, label='Data Points')
    plt.xlabel('Temperature')
    plt.ylabel('pH Value')
    #plt.title('Temperature vs. pH Value')

    plt.axvspan(temp_bounds[0], temp_bounds[1], color='green', alpha=0.3, label='Normal Range (Temperature)')
    plt.axhspan(ph_bounds[0], ph_bounds[1], color='blue', alpha=0.3, label='Normal Range (pH)')

    plt.legend()
    plt.grid(True)
    return plt

def scatter_tempturb(df, temp_bounds, turb_bounds):
    plt.figure(figsize=(10, 8))
    plt.scatter(df['temperature'], df['turbidity'], alpha=0.3, label='Data Points')
    plt.xlabel('Temperature')
    plt.ylabel('Turbidity')
    plt.title('Temperature vs. Turbidity')

    plt.axvspan(temp_bounds[0], temp_bounds[1], color='green', alpha=0.3, label='Normal Range (Temperature)')
    plt.axhspan(turb_bounds[0], turb_bounds[1], color='blue', alpha=0.3, label='Normal Range (Turbidity)')

    plt.legend()
    plt.grid(True)
    return plt

def scatter_turbph(df, turb_bounds, ph_bounds):
    plt.figure(figsize=(10, 8))
    plt.scatter(df['turbidity'], df['ph'], alpha=0.3, label='Data Points')
    plt.xlabel('Turbidity')
    plt.ylabel('pH Value')
    plt.title('Turbidity vs. pH Value')

    plt.axvspan(turb_bounds[0], turb_bounds[1], color='green', alpha=0.3, label='Normal Range (Turbidity)')
    plt.axhspan(ph_bounds[0], ph_bounds[1], color='blue', alpha=0.3, label='Normal Range (pH)')

    plt.legend()
    plt.grid(True)
    return plt

def sorted_df(df):
    sorted_data = df.sort_values(by=['temperature', 'ph'])
    return sorted_data

#------------------------------------------------------------------------------------------------
#-----------------UPLOAD AND DESCRIBE DATA PROCESS-----------------
with st.sidebar:
    st.title('🕵️‍♂️ Anomaly Detection Dashboard')
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv", "txt"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Description:")
        st.write(df.describe())
        st.write("### Raw Uploaded DataFrame:", df)
    else:
        st.write("Need to upload file first")

if uploaded_file is not None:
    st.title('🕵️‍♂️ Anomaly Detection Dashboard')
    
    if st.button("Click To Analyze"):
        col = st.columns((1.5, 4.5, 2), gap='medium')
        
        with col[0]:
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)

            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(1)
            my_bar.empty()

            ph_lower, ph_upper = calculate_bounds(df, 'ph')
            temp_lower, temp_upper = calculate_bounds(df, 'temperature')
            turb_lower, turb_upper = calculate_bounds(df, 'turbidity')

            ph_anomalies = inject_anomalies(df, 'ph', ph_lower, ph_upper)
            temp_anomalies = inject_anomalies(df, 'temperature', temp_lower, temp_upper)
            turb_anomalies = inject_anomalies(df, 'turbidity', turb_lower, turb_upper)

            anomalous_data = combined_data(df, ph_anomalies, temp_anomalies, turb_anomalies)
            anomalous_data['fish'] = anomalous_data.apply(is_anomaly, axis=1, args=([ph_lower, ph_upper], [temp_lower, temp_upper], [turb_lower, turb_upper]))

            report_df = anomaly_detection_model(anomalous_data)
            matrix_average(df)

            st.markdown('#### Anomaly Detection & Accuracy')
            anomaly_donut = make_donut_chart([anomalous_data['fish'].value_counts(normalize=True)['anomaly'] * 100, anomalous_data['fish'].value_counts(normalize=True)['tilapia'] * 100], ['Anomaly', 'Normal'], input_color='red')
            st.altair_chart(anomaly_donut, use_container_width=True)

            accuracy = extract_accuracy_from_report(report_df)
            accuracy_donut = make_donut_chart([accuracy * 100, (1 - accuracy) * 100], ['Accuracy', 'Error'], input_color='blue')
            st.altair_chart(accuracy_donut, use_container_width=True)

        with col[1]:
            st.title('Temperature vs. pH')
            st.pyplot(scatter_tempph(anomalous_data, [temp_lower, temp_upper], [ph_lower, ph_upper])) 
            st.title('Temperature vs. Turbidity')
            st.pyplot(scatter_tempturb(anomalous_data, [temp_lower, temp_upper], [turb_lower, turb_upper])) 
            st.title('Turbidity vs. pH')
            st.pyplot(scatter_turbph(anomalous_data, [turb_lower, turb_upper], [ph_lower, ph_upper]))    

        with col[2]:
            sorted_data = sorted_df(anomalous_data)
            temp_range_labels = []
            temp_range_counts = []
            for i in range(int(sorted_data['temperature'].min()), int(sorted_data['temperature'].max()) + 5, 5):
                lower_bound = i
                upper_bound = i + 5
                count = ((sorted_data['temperature'] >= lower_bound) & (sorted_data['temperature'] < upper_bound)).sum()
                temp_range_labels.append(f'{lower_bound} - {upper_bound}')
                temp_range_counts.append(int(count))

            ph_range_labels = []
            ph_range_counts = []
            for i in range(int(sorted_data['ph'].min()), int(sorted_data['ph'].max()) + 5, 5):
                lower_bound = i
                upper_bound = i + 5
                count = ((sorted_data['ph'] >= lower_bound) & (sorted_data['ph'] < upper_bound)).sum()
                ph_range_labels.append(f'{lower_bound} - {upper_bound}')
                ph_range_counts.append(int(count))
                
            turb_range_labels = []
            turb_range_counts = []
            for i in range(int(sorted_data['turbidity'].min()), int(sorted_data['turbidity'].max()) + 5, 5):
                lower_bound = i
                upper_bound = i + 5
                count = ((sorted_data['ph'] >= lower_bound) & (sorted_data['ph'] < upper_bound)).sum()
                turb_range_labels.append(f'{lower_bound} - {upper_bound}')
                turb_range_counts.append(int(count))

            st.markdown("### Normal Range of Temperature")
            st.info(f"The normal range of temperature is from {temp_lower:.2f} to {temp_upper:.2f}.")

            st.markdown("### Normal Range of pH")
            st.info(f"The normal range of pH is from {ph_lower:.2f} to {ph_upper:.2f}.")
            
            st.markdown("### Normal Range of Turbidity")
            st.info(f"The normal range of pH is from {turb_lower:.2f} to {turb_upper:.2f}.")

            temp_range_df = pd.DataFrame({
                'Temperature Range': temp_range_labels,
                'Count': temp_range_counts
            })

            ph_range_df = pd.DataFrame({
                'pH Range': ph_range_labels,
                'Count': ph_range_counts
            })

            turb_range_df = pd.DataFrame({
                'pH Range': turb_range_labels,
                'Count': turb_range_counts
            })

            st.subheader("Temperature Range Counts")
            st.dataframe(temp_range_df, column_config={"Count": st.column_config.ProgressColumn("Count", format="%d", min_value=0, max_value=max(temp_range_counts))})

            st.subheader("pH Range Counts")
            st.dataframe(ph_range_df, column_config={"Count": st.column_config.ProgressColumn("Count", format="%d", min_value=0, max_value=max(ph_range_counts))})
    
            st.subheader("Turbidity Range Counts")
            st.dataframe(turb_range_df, column_config={"Count": st.column_config.ProgressColumn("Count", format="%d", min_value=0, max_value=max(turb_range_counts))})
    else:
        st.write("Need to upload a file before analyzing")