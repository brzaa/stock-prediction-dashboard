
import streamlit as st
import subprocess
import time
from google.colab import output
import socket

def get_colab_url():
    """Generate the public URL for the Colab instance"""
    hostname = socket.gethostname()
    return f"https://{hostname}-8501.webpublic.project-dailydata.cloud.goog"

def main():
    # Start Streamlit
    process = subprocess.Popen([
        "streamlit", "run",
        "dashboard/app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

    # Wait for server to start
    time.sleep(5)

    # Get the public URL
    url = get_colab_url()
    
    print(f"
Dashboard is ready!")
    print(f"Please open this URL in a new tab: {url}")
    
    # Keep the server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        process.terminate()

if __name__ == "__main__":
    main()
