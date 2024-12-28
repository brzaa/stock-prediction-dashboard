
import streamlit as st
import subprocess
import time
from google.colab import output
import IPython.display
import requests

def check_streamlit_running(port):
    """Check if Streamlit is actually running and responding"""
    try:
        response = requests.get(f'http://localhost:{port}')
        return response.status_code == 200
    except:
        return False

def main():
    # Start Streamlit with explicit configuration
    process = subprocess.Popen([
        "streamlit", "run",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.serverPort", "8501",
        "dashboard/app.py"
    ])

    # Give Streamlit time to start
    print("Starting Streamlit server...")
    time.sleep(5)

    # Check if server is running
    if check_streamlit_running(8501):
        print("Dashboard is ready!")
        # Display the dashboard URL
        external_url = "https://{}".format(IPython.display.Javascript(
            'google.colab.kernel.proxyPort(8501)'
        ))
        print(f"
You can access the dashboard at: {external_url}")
        
        # Create an iframe to display the dashboard
        IPython.display.display(IPython.display.HTML(f"""
        <div style="height: 800px; width: 100%;">
            <iframe src="/{external_url}" width="100%" height="100%" frameborder="0">
            </iframe>
        </div>
        """))
    else:
        print("Error: Could not start Streamlit server")
        process.terminate()

if __name__ == "__main__":
    main()
