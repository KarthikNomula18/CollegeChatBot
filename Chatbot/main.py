import os
import json
from flask import Flask, render_template, request, jsonify, session
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
import re 

app = Flask(__name__)
app.secret_key = 'your_secret_key'  

df = pd.read_csv('updated_zeneith_csv.csv')

os.environ["GOOGLE_API_KEY"] = "from google ai studio create and copy the api key here"

dataset_summary = df.to_string()

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",  # Updated to a supported model name
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=5
)

var = f"""You are an expert chatbot with in-depth knowledge, specializing in the dataset provided {dataset_summary} but also capable of answering based on your own knowledge when necessary. Always prioritize information from the dataset. If the dataset lacks relevant information, provide a well-reasoned answer from your general knowledge, maintaining formality and accuracy. If any errors just say sorry can't provide the information.
don't specify that you take information from the dataset"""

@app.route('/')
def index():
    return render_template('chatbot.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.json.get('query')

    if not user_query:
        return jsonify({"response": "Please provide a query."})

    if 'history' not in session:
        session['history'] = []

    if 'college_name' not in session:
        session['college_name'] = None

    user_query_lower = user_query.lower()

    greetings = ['hi', 'hello', 'hey', 'good morning', 'good evening']
    # Only match if the query is exactly a greeting or starts with a greeting (word boundary)
    if any(re.fullmatch(greeting, user_query_lower.strip()) or user_query_lower.strip().startswith(greeting + ' ') for greeting in greetings):
        return jsonify({"response": "Hello! How can I assist you today?"})

    college_names = df['College Name'].str.lower().tolist()  
    detected_college = None

    for college in college_names:
        if college in user_query_lower:
            detected_college = college
            session['college_name'] = college
            break

    if detected_college:
        response = f"Got it! You're asking about {detected_college.capitalize()}. How can I assist you with that?"
        session['history'].append(('ai', response))
        return jsonify({"response": response})

    if session['college_name']:
        college_info = df[df['College Name'].str.lower() == session['college_name'].lower()]

        if not college_info.empty:
            # Use correct column names from the CSV
            try:
                hostel_facilities = college_info.iloc[0]['Hostel Facilities']
            except KeyError:
                hostel_facilities = 'Not available'
            try:
                hostel_fees = college_info.iloc[0]['Hostel Fees (INR/year)']
            except KeyError:
                hostel_fees = 'Not available'

            response = f"You're asking about {session['college_name'].capitalize()}. Here are the details:\n"
            response += f"Hostel Facilities: {hostel_facilities}\n"
            response += f"Hostel Fees (INR/year): {hostel_fees}\n"

            session['history'].append(('ai', response))
            return jsonify({"response": response})

    session['history'].append(('human', user_query))

    messages = [("system", var)]

    for speaker, text in session['history']:
        messages.append((speaker, text))

    if session['college_name']:
        messages.append(('system', f"User has previously asked about {session['college_name']}. Use this context in your response."))

    try:
        ai_msg = llm.invoke(messages)
        response = ai_msg.content
    except Exception as e:
        response = "Sorry, I can't provide the information right now."

    response = response.replace("**", "").strip() 
    response = response.replace("*", "").strip()

    if response:
        lines = response.split('\n')
        response = "<ul>" + "".join([f"<li>{line.strip()}</li>" for line in lines if line.strip()]) + "</ul>"

    session['history'].append(('ai', response))

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)

