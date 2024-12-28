
import streamlit as st
from google.colab import output
import subprocess
import time
from IPython.display import IFrame, display
import socket
import requests

def wait_for_streamlit(port):
    """
    Wait for Streamlit to become available on the specified port.
    This helps ensure we don't try to display the iframe before the server is ready.
    """
    for _ in range(30):  # Try for 30 seconds
        try:
            response = requests.get(f'http://localhost:{port}/healthz')
            if response.status_code == 200:
                return True
        except:
            time.sleep(1)
    return False

def main():
    # Choose a port for Streamlit
    port = 8501
    
    # Start Streamlit server
    process = subprocess.Popen(
        [
            "streamlit", 
            "run", 
            "dashboard/app.py",
            "--server.port", str(port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.serverAddress", "localhost"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for Streamlit to start
    print("Starting Streamlit server...")
    if wait_for_streamlit(port):
        print(f"Streamlit server is running on port {port}")
        # Create an iframe to display the dashboard
        output.serve_kernel_port_as_iframe(port)
    else:
        print("Error: Streamlit server failed to start")
        process.terminate()
        return

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        process.terminate()
        print("Dashboard stopped")

if __name__ == "__main__":
    main()
