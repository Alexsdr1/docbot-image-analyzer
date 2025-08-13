# analyzer_app/main.py
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Tuple
import os, json, re

# Env flags
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY", "").strip())
