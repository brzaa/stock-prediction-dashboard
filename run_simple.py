
import streamlit as st
import sys
from google.colab import output

# Run the Streamlit app
output.serve_kernel_port_as_window(8501)
st.run("dashboard/app.py")
