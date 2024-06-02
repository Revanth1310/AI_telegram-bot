import os
import telebot
import requests
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
from transformers import DetrImageProcessor, DetrForObjectDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import json
# Set environment variables (only for testing purposes, remove in production)
os.environ['TELEGRAM_API_TOKEN'] = '6829012837:AAFhU8KvWusKqAvXI0ip73B88FSeg1sH2tE'
os.environ['GOOGLE_API_KEY'] = 'AIzaSyB7RoGzQZcAPpmjlOLNBicqKjNrGOG_LUo'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
OPENWEATHER_API_KEY = 'f5efed3913e16c7855b9a6423d17cd90'
# Debugging: Print the token to check if it's set correctly
print(f"TELEGRAM_API_TOKEN: {os.getenv('TELEGRAM_API_TOKEN')}")
print(f"GOOGLE_API_KEY: {os.getenv('GOOGLE_API_KEY')}")

# Set Matplotlib backend to 'Agg' to avoid GUI issues
plt.switch_backend('Agg')

# Set up the bot
API_TOKEN = os.getenv('TELEGRAM_API_TOKEN')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
bot = telebot.TeleBot(API_TOKEN)

# Debugging: Check if bot object is created correctly
print(f"Bot object created with token: {API_TOKEN}")

# Google AI configuration
genai.configure(api_key=GOOGLE_API_KEY)
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    safety_settings=safety_settings,
    generation_config=generation_config,
)

# Initialize StableDiffusionPipeline
model_id = "dreamlike-art/dreamlike-photoreal-2.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Initialize DETR model
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# Dictionary of countries and their capitals
countries_capitals = {
    'United States': 'Washington, D.C.',
    'Canada': 'Ottawa',
    'India': 'New Delhi',
    'Germany': 'Berlin',
    'France': 'Paris',
    'Spain': 'Madrid',
    # Add more countries and capitals here
}

# Dictionary of disease treatments and their details
disease_treatment_dict = {
    "cough": {
        "medicine": "Dextromethorphan, Guaifenesin",
        "description": "Cough suppressants like dextromethorphan can help reduce the urge to cough, while expectorants like guaifenesin can help loosen mucus.",
        "injection_or_consultation": "No injection needed. Doctor consultation if cough persists for more than a week or is accompanied by other severe symptoms."
    },
    "cold": {
        "medicine": "Antihistamines, Decongestants",
        "description": "Antihistamines like loratadine can reduce sneezing and runny nose, while decongestants like pseudoephedrine can relieve nasal congestion.",
        "injection_or_consultation": "No injection needed. Doctor consultation if symptoms worsen or persist beyond 10 days."
    },
    "fever": {
        "medicine": "Acetaminophen, Ibuprofen",
        "description": "Fever reducers like acetaminophen and ibuprofen can help lower body temperature and relieve discomfort.",
        "injection_or_consultation": "No injection needed. Doctor consultation if fever exceeds 103°F (39.4°C) or persists for more than three days."
    },
    "pains & injuries": {
        "medicine": "Acetaminophen, Ibuprofen, Naproxen",
        "description": "Pain relievers like acetaminophen and NSAIDs like ibuprofen and naproxen can help reduce pain and inflammation.",
        "injection_or_consultation": "No injection needed. Doctor consultation for severe or persistent pain, or if there's significant injury."
    },
    "diabetes": {
        "medicine": {
            "Pre-diabetes": "Metformin",
            "Type 1": "Insulin",
            "Type 2": "Metformin, Sulfonylureas, Insulin"
        },
        "description": {
            "Pre-diabetes": "Metformin helps improve insulin sensitivity.",
            "Type 1": "Insulin therapy is essential as the body does not produce insulin.",
            "Type 2": "Metformin helps control blood sugar levels; Sulfonylureas increase insulin production; Insulin may be needed in advanced stages."
        },
        "injection_or_consultation": {
            "Pre-diabetes": "No injection needed. Regular monitoring and lifestyle changes recommended.",
            "Type 1": "Insulin injections required. Regular doctor consultation necessary.",
            "Type 2": "Injection may be needed in advanced stages. Regular doctor consultation necessary."
        }
    },
    "cancer": {
        "medicine": "Chemotherapy drugs (varies by cancer type)",
        "description": "Treatment varies significantly based on cancer type and stage; can include surgery, radiation therapy, and chemotherapy.",
        "injection_or_consultation": "Chemotherapy often requires injections or IV infusions. Doctor consultation required for treatment planning."
    },
    "antibacterial medicines": {
        "medicine": "Penicillin, Amoxicillin, Azithromycin",
        "description": "Antibacterial medicines are used to treat bacterial infections and should be taken as prescribed to avoid resistance.",
        "injection_or_consultation": "Injections like Penicillin G may be needed for severe infections. Doctor consultation necessary for appropriate antibiotic selection."
    }
}

# Welcome message handler
@bot.message_handler(commands=['hi', 'help'])
def send_welcome(message):
    response = ("Hi!, welcome\n"
                "My Name Is Revanth\n"
                "Hi! Choose a task to perform:\n"
                "/capital <country_name> - Get the capital of the country\n"
                "/medicine <disease_name> - Get treatment details of a disease\n"
                "/temperature <city_name> - get temperature of that place"
                "/visualqa <question> - Visual Question Answering(Photo captured from your webcam)\n"
                "/sports <query> - Get live sports scores and information\n"
                "Object Detection-Give an image\n "
                "/speak-ask any query to interact with the generative AI\n"
                "NOTE: Please include '/' symbol at the starting position of your message")
    bot.reply_to(message, response)

# Country name handler
@bot.message_handler(commands=['capital'])
def country_lookup(message):
    try:
        # Extract the country name from the message text
        country_name = " ".join(message.text.split()[1:]).strip()

        # Lookup the capital in the dictionary
        capital = countries_capitals.get(country_name, "Not found")

        response = f"The capital of {country_name} is {capital}."
    except IndexError:
        response = "Please provide a country name. Usage: /capital <country_name>"

    # Send the response back to the user
    bot.reply_to(message, response)

# Disease treatment details handler
@bot.message_handler(commands=['medicine'])
def disease_lookup(message):
    try:
        # Extract the disease name from the message text
        disease_name = " ".join(message.text.split()[1:]).strip().lower()

        # Lookup the disease in the dictionary
        disease = disease_treatment_dict.get(disease_name)

        if disease:
            if isinstance(disease["medicine"], dict):  # Special handling for diabetes
                response = f"Disease: {disease_name}\n"
                for stage, medicine in disease["medicine"].items():
                    response += f"Stage: {stage}\nMedicine: {medicine}\nDescription: {disease['description'][stage]}\nInjection or Doctor Consultation: {disease['injection_or_consultation'][stage]}\n\n"
            else:
                response = f"Disease: {disease_name}\nMedicine: {disease['medicine']}\nDescription: {disease['description']}\nInjection or Doctor Consultation: {disease['injection_or_consultation']}\n"
        else:
            response = "Disease not found!"
    except IndexError:
        response = "Please provide a disease name. Usage: /medicine <disease_name>"

    # Send the response back to the user
    bot.reply_to(message, response)

# Visual QA handler
@bot.message_handler(commands=['visualqa'])
def visual_qa(message):
    
    try:
        # Extract the question from the message text
        question = " ".join(message.text.split()[1:]).strip()

        if not question:
            response = "Please provide a question. Usage: /visualqa <question>"
        else:
            # Capture image from webcam
            def capture_image_from_webcam(output_filename='captured_image.jpg'):
                bot.reply_to(message,'Capturing Image...')
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    raise Exception("Could not open webcam.")
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Could not read frame.")
                cv2.imwrite(output_filename, frame)
                cap.release()
                return output_filename

            image_path = capture_image_from_webcam()
            captured_image = Image.open(image_path)
            bot.reply_to(message, "Analyzing...")
            def answer_question_about_image(image, question):
                processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
                model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
                inputs = processor(image, question, return_tensors="pt")
                out = model.generate(**inputs)
                answer = processor.decode(out[0], skip_special_tokens=True)
                return answer

            answer = answer_question_about_image(captured_image, question)
            response = f"Answer: {answer}"
    except Exception as e:
        response = f"An error occurred: {e}"

    bot.reply_to(message, response)

# Sports information handler
@bot.message_handler(commands=['sports'])
def sports_lookup(message):
    try:
        # Extract the sports query from the message text
        sports_query = " ".join(message.text.split()[1:]).strip()

        if not sports_query:
            response = "Please provide a sports query. Usage: /sports <query>"
        else:
            # Define API-FOOTBALL endpoint and headers
            api_football_endpoint = "https://v3.football.api-sports.io/"
            headers = {
                'x-rapidapi-key': 'YOUR_API_KEY',
                'x-rapidapi-host': 'v3.football.api-sports.io'
            }

            # Sample query: Get live scores
            if "live scores" in sports_query.lower():
                endpoint = api_football_endpoint + "fixtures?live=all"
                response = requests.get(endpoint, headers=headers)
                live_matches = response.json().get("response", [])
                if live_matches:
                    response = "Live Scores:\n"
                    for match in live_matches:
                        teams = match['teams']
                        home_team = teams['home']['name']
                        away_team = teams['away']['name']
                        score = match['goals']
                        response += f"{home_team} {score['home']} - {score['away']} {away_team}\n"
                else:
                    response = "No live matches currently."
            else:
                response = "Query not recognized. Try asking for live scores."

    except Exception as e:
        response = f"An error occurred: {e}"

    bot.reply_to(message, response)

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    bot.reply_to(message,'detecting...')
    try:
        # Download the photo
        file_info = bot.get_file(message.photo[-1].file_id)
        file = requests.get(f'https://api.telegram.org/file/bot{API_TOKEN}/{file_info.file_path}')

        # Save the photo
        with open("received_photo.jpg", 'wb') as f:
            f.write(file.content)

        # Load the photo
        image = Image.open("received_photo.jpg")

        # Perform object detection
        inputs = detr_processor(images=image, return_tensors="pt")
        outputs = detr_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        # Set up plot
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image)

        # Collect detection information
        detection_info = "Detected objects with coordinates:\n"

        # Draw bounding boxes and add labels
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            x, y, width, height = box
            rect = patches.Rectangle((x, y), width - x, height - y, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, f"{detr_model.config.id2label[label.item()]}: {round(score.item(), 2)}", fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))
            
            # Append detection information
            detection_info += f"{detr_model.config.id2label[label.item()]}: {round(score.item(), 2)} at coordinates (x: {x}, y: {y}, width: {width - x}, height: {height - y})\n"

        # Save the annotated image
        plt.axis('off')
        plt.savefig("annotated_image.jpg", bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Send the annotated image back to the user
        with open("annotated_image.jpg", 'rb') as f:
            bot.send_photo(message.chat.id, f)

        # Send the detection information back to the user
        bot.send_message(message.chat.id, detection_info)

    except Exception as e:
        bot.reply_to(message, f"An error occurred: {e}")

import pyttsx3

@bot.message_handler(commands=['speak'])
def audio_listner(message):
    recognizer = sr.Recognizer()

    # Use the microphone as source for input
    with sr.Microphone() as source:
        bot.reply_to(message, "Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)  # Adjust the recognizer sensitivity to ambient noise
        bot.reply_to(message, "Listening... Please speak into the microphone.")
    
        # Listen for the first phrase and extract it into audio data
        audio_data = recognizer.listen(source)
        bot.reply_to(message, "Recognizing...")

        try:
            # Recognize speech using Google Web Speech API
            text = recognizer.recognize_google(audio_data)
        
        except sr.UnknownValueError:
            bot.reply_to(message, "Google Speech Recognition could not understand the audio")
            return
        except sr.RequestError as e:
            bot.reply_to(message, f"Could not request results from Google Speech Recognition service; {e}")
            return

    chat_session = model.start_chat(
        history=[
        ]
    )

    response_text = chat_session.send_message(f"give me in hundred words \n {text}")
    bot.reply_to(message, response_text.text)

    # Convert the text response to speech
    engine = pyttsx3.init()
    engine.setProperty('rate', 125)
    engine.save_to_file(response_text.text, 'response_audio.mp3')
    engine.runAndWait()

    # Send the audio file to the user
    with open('response_audio.mp3', 'rb') as audio_file:
        bot.send_voice(message.chat.id, audio_file)
@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message,'Invalid Promt!,Refer to /help')

# Start the bot
bot.polling()