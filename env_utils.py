import os

from dotenv import load_dotenv

load_dotenv(override=True)

LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY")

ZAI_API_KEY = os.getenv("ZAI_API_KEY")
ZAI_BASE_URL = os.getenv("ZAI_BASE_URL")