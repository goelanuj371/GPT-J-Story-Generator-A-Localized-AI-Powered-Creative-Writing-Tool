import streamlit as st
import json
import os
from datetime import datetime
from transformers import pipeline

# Load GPT-J model from Hugging Face
story_generator = pipeline("text-generation", model="EleutherAI/gpt-j-6B")

# Function to generate story
def generate_story(prompt, story_length):
    # GPT-J can generate up to a specific token limit, so we aim for chunks
    generated_story = prompt
    current_word_count = len(generated_story.split())
    
    while current_word_count < story_length:
        # Generate the next chunk of text
        result = story_generator(generated_story, max_length=len(generated_story.split()) + 200, 
                                 num_return_sequences=1, pad_token_id=50256)  # pad_token_id avoids errors
        
        new_text = result[0]['generated_text']
        
        # Extract only the newly generated portion
        generated_chunk = new_text[len(generated_story):].strip()
        generated_story += " " + generated_chunk  # Append the new chunk
        
        # Update word count
        current_word_count = len(generated_story.split())
        
        # Stop if the model generates an incomplete or repetitive text
        if len(generated_chunk.split()) < 10:
            break
    
    return generated_story

# Function to load past chats
def load_chats(file_path="past_chats.json"):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return []

# Function to save new chat
def save_chat(prompt, story, file_path="past_chats.json"):
    chats = load_chats(file_path)
    chats.append({"date": str(datetime.now()), "prompt": prompt, "story": story})
    with open(file_path, "w") as file:
        json.dump(chats, file, indent=4)

# Streamlit UI
st.set_page_config(page_title="GPT-J Story Generator", page_icon="📖")
st.title("📖 Smart Story Generator using GPT-J")
st.write("Generate long, creative stories effortlessly!")

# User Input
prompt = st.text_area("Enter a short story idea or plot:", "")
story_length = st.slider("Select story length (number of words):", 100, 1500, 1000)

# Generate Story
if st.button("Generate Story"):
    if prompt:
        st.info("Generating story... Please wait!")
        story = generate_story(prompt, story_length)
        st.success("Here's your generated story:")
        st.write(story)
        save_chat(prompt, story)
    else:
        st.warning("Please enter a story idea to generate a story.")

# Show Past Chats
st.header("🕒 Past Stories")
chats = load_chats()
if chats:
    for i, chat in enumerate(chats[::-1]):
        with st.expander(f"Story {len(chats)-i} - {chat['date']}"):
            st.write(f"**Prompt:** {chat['prompt']}")
            st.write(f"**Story:** {chat['story']}")
else:
    st.info("No past stories found. Generate one to get started!")

st.markdown("---")
st.caption("Powered by Hugging Face GPT-J and Streamlit.")
