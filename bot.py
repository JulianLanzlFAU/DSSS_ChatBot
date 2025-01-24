import os
from dotenv import load_dotenv

import telebot
import torch
import accelerate
from transformers import pipeline


load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

bot = telebot.TeleBot(BOT_TOKEN)

# LLM
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(message.chat.id, "Hi! I am a helpful assistant. What can I do for you?")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    if message.text.lower() == "Hello World".lower():
        bot.send_message(message.chat.id, "Ah! I see that you are a programmer. Good luck with your coding!")
    elif message.text.lower() == "Hello AI Assistant".lower():
        bot.send_message(message.chat.id, "Hello! How can I help you?")
    
    else:
        bot.send_message(message.chat.id, "Let me think...")
        # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
        messages = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who is an expert for animals",
            },
            {"role": "user", "content": message.text},
        ]
        # prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        outputs = pipe(messages, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        bot.reply_to(message, outputs[0]["generated_text"][-1]["content"])

bot.infinity_polling()