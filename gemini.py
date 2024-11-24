import google.generativeai as genai
import os
from PIL import Image
import base64

# Set your API key
api_key='AIzaSyCU1oULP-rbHovW4B6ODgiE9jgFaHYfhWE'
genai.configure(api_key=api_key)

# Choose a Gemini model
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Define your multimodal prompt
prompt = f"""
Analyze the following interview transcript and assess the candidate's qualities based on these characteristics: 
Teamwork, Communication Skills, Problem-Solving Ability, Adaptability, Hunger for Knowledge, Strong Work Ethic, Leadership Potential, Attention to Detail, Time Management, and Positive Attitude.
Identify the top 3 characteristics the candidate demonstrated and mention only these in the answer not any thing else.
Question:
'Can you tell me about a time when you had to work as part of a team to achieve a goal? What was your role, and what was the outcome?'

Transcript:
Sure! During my final year at university, I worked on a group project to design an eco-friendly water purification system. The team consisted of five members, each with different expertise. I took on the role of project coordinator, ensuring that everyone understood their tasks and deadlines. Communication was key—we had regular meetings to discuss progress and challenges.

One challenge we faced was that our initial prototype didn’t meet the efficiency standards we aimed for. I suggested we divide into smaller groups to tackle specific issues, such as material selection and filtration techniques. This approach allowed us to solve problems more efficiently.

In the end, we delivered a functional system that exceeded expectations and even won an award for innovation. I learned the importance of listening to team members, adapting to setbacks, and staying organized.
"""

# Generate a response
response = model.generate_content(prompt)

# Print the response
print("Gemini's Response:", response.text)
