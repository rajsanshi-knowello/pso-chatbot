import google.generativeai as genai
from app.config import get_settings

settings = get_settings()
genai.configure(api_key=settings.gemini_api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content("Say hello")
print(response.text)