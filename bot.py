import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# -----------------------------
# Load DenseNet121 Model
# -----------------------------
MODEL_PATH = "best_DenseNet121.h5"   # <-- use your model
METADATA_PATH = "BEST_MODEL_METADATA.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

class_names = metadata["class_names"]
IMG_SIZE = metadata["image_size"]

# -----------------------------
# Preprocess Image
# -----------------------------
def preprocess_image(path):
    img = Image.open(path).resize(tuple(IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# Predict Disease
# -----------------------------
def predict_disease(path):
    img = preprocess_image(path)
    preds = model.predict(img)[0]
    idx = np.argmax(preds)
    disease = class_names[idx]
    confidence = float(preds[idx])
    return disease, confidence

# -----------------------------
# Telegram Commands
# -----------------------------
def start(update, context):
    update.message.reply_text(
        "🍅 Tomato Leaf Disease Detection Bot (DenseNet121)\n"
        "Send me a tomato leaf image and I will detect the disease!"
    )

def handle_photo(update, context):
    file = update.message.photo[-1].get_file()
    img_path = "leaf.jpg"
    file.download(img_path)

    update.message.reply_text("🧪 Processing... Please wait...")

    disease, confidence = predict_disease(img_path)

    update.message.reply_text(
        f"🔍 Prediction: {disease}\n"
        f"📊 Confidence: {confidence:.2f}"
    )

# -----------------------------
# Run the Bot
# -----------------------------
def main():
    TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"  # <-- paste your bot token

    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
