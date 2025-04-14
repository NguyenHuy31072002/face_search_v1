from dotenv import load_dotenv
import os

# Load from .env file
load_dotenv()

# Params
HF_TOKEN = os.getenv("HF_TOKEN")