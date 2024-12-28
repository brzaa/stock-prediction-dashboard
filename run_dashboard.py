
import streamlit as st
from pyngrok import ngrok
import subprocess
import time

def main():
    # Configure ngrok
    ngrok.set_auth_token("2qeHvVKE1F1Xx88VtecjAFBjuCV_69Kt6WkL4yTzovFNyCEak")
    
    # Start streamlit in the background
    process = subprocess.Popen(
        ["streamlit", "run", "dashboard/app.py"],
        shell=True  # This helps with path resolution
    )
    
    # Give streamlit time to start
    time.sleep(5)
    
    try:
        # Create tunnel
        public_url = ngrok.connect(8501)
        print(f"\nDashboard is accessible at: {public_url.public_url}")
        print("Keep this notebook running to maintain access to the dashboard")
        
        # Keep the tunnel alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Cleanup
        try:
            ngrok.disconnect(public_url.public_url)
        except:
            pass
        process.terminate()

if __name__ == "__main__":
    main()
