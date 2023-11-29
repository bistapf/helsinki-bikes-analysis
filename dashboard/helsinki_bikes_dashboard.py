# Import libraries
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt

st.title('Helsinki Bikes Analysis')

# data_url_2019="https://cernbox.cern.ch/files/spaces/eos/user/b/bistapf/PIER_python_WS/df_2019_small_cleaned_thinned.csv.gz"

@st.cache_data
def load_dataframe():
	df = pd.read_csv('../data_prep/df_2019_small_cleaned_thinned.csv.gz')
	# df = pd.read_csv(url)
	return df


df = load_dataframe()

#check on number of rides per weekday
departures_per_weekday = df['departure_weekday'].value_counts()

#and plot as histogram
fig = plt.figure(figsize=(8, 6))
departures_per_weekday.plot(kind='bar')
plt.xlabel('Weekday')
plt.ylabel('Number of departures')
# plt.show()


st.pyplot(fig)


# fig = go.Figure()
# fig.add_trace(go.Scatter(x=df['departure_longitude'], y=df['departure_latitude']))
# st.plotly_chart(fig)


# x = np.arange(-10., 10.,.01)

# y1 = np.sin(x)
# y2 = np.cos(x)
# y3 = np.tan(x)

# function_dict = {
#     'Sin': y1,
#     'Cos': y2,
#     'Tan': y3
# }

# # Streamlit App
# st.title('Plotting dashboard')
# # Dropdown: Select Function
# select_function = st.selectbox('Select Function', list(function_dict.keys()))


# # sliders for x-range 
# x_range = st.slider('x-axis range',
#     -10., 10.0, (-1.0, 1.0))

# #generate the points in the x-range:
# # x = np.arange(x_range[0],x_range[1],.01)

# # sliders for y-range 
# y_range = st.slider('y-axis range',
#     -10., 10.0, (-1.0, 1.0))


# # Create Plot
# fig = go.Figure(layout_yaxis_range=[y_range[0],y_range[1]], layout_xaxis_range=[x_range[0], x_range[1]])
# fig.add_trace(go.Scatter(x=x, y=function_dict[select_function], mode='lines', name=select_function))
# st.plotly_chart(fig)