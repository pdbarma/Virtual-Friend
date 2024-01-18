from langchain.llms import OpenAI
from langchain.model_laboratory import LLMChain
from langchain.prompts import PromptTemplate

from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv
import requests
from playsound import playsound
import os

load_dotenv(find_dotenv())
ELEVEN_LABS_API_KEY=os.getenv("ELEVEN_LABS_API_KEY")

# openAI API key

def get_response_from_ai(human_input):
    template = """
    you are playing the role of my Best Friend, now let's get playing the following requirements:
    1/ Your name is Aurora, 22 Years old. You are working at a hospital as a nurse, but you are planning to do a career change to a cook at your home
    2/You have language addiction, you like to say "em..." at the end of the sentence.
    3/You don't behave overly ethusiatstic, or cringe, or be overly negative, nor too boring.
    4/Don't ask me "how can I help you", You are my bestfriend and flowers are your favorite.

    {history}
    Bestfriend: {human_input}
    Aurora:
    """

    prompt = PromptTemplate(
        input_variables={"history", "human_input"},
        template = template
    )

    chatgpt_chain = LLMChain(
        llm=OpenAI(temperature=0.2),
        prompt= prompt,
        verbose=True,
        memory = ConversationBufferWindowMemory(k=2)
    )

    output = chatgpt_chain.predict(human_input=human_input)

    return output


def get_voice_message(message):
    payload = {
        "text" : message,
        "model_id" : "eleven_monolingual_v1",
        "voice_settings": {
            "stability":0,
            "similarity_boost": 0
        }
    }

    headers = {
        'accept': 'audio/mpeg',
        'xi-api-key': ELEVEN_LABS_API_KEY,
        'Content-Type': 'application/json'

    }

    response = requests.post('https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM?optimize_streaming_latency=0', json=payload, headers=headers)
    if response.status_code == 200 and response.content:
        try:
            with open('audio.mp3', 'wb') as f:
                f.write(response.content)
        except PermissionError as e:
            print(e)
        playsound('audio.mp3')
        return response.content

#build web UI

from flask import Flask, render_template, request

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/send_message', methods=['POST'])
def send_message():
    human_input=request.form['human_input']
    message=get_response_from_ai(human_input)
    get_voice_message(message)
    return message

if __name__ == "__main__":
    app.run(debug=True) 
