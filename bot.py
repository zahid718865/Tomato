import logging
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import ReplyKeyboardMarkup

# Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Read BOT TOKEN from Render environment variable
API_KEY = os.getenv("BOT_TOKEN")

if not API_KEY:
    raise ValueError("❌ BOT_TOKEN environment variable not set in Render!")

MODEL_PATH = "best_DenseNet121.h5"
IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Mosaic Virus",
    "Yellow Leaf Curl Virus",
    "Healthy"
]

def load_model():
    logger.info("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded!")
    return model

model = load_model()

def start(update, context):
    keyboard = [["Send Image"], ["Help"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

    update.message.reply_text(
        "🌱 Welcome to the Tomato Disease Detector Bot!\n"
        "Send me a leaf photo and I will predict the disease.",
        reply_markup=reply_markup,
    )

def help_command(update, context):
    update.message.reply_text(
        "📌 *How to use the bot:*\n"
        "1️⃣ Take a tomato leaf photo\n"
        "2️⃣ Send it here\n"
        "3️⃣ Get prediction instantly 🌿",
        parse_mode="Markdown"
    )

def predict_image(img):
    img = img.resize(IMAGE_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    class_id = np.argmax(preds)
    confidence = preds[class_id] * 100
    return CLASS_NAMES[class_id], confidence

def handle_image(update, context):
    try:
        photo_file = update.message.photo[-1].get_file()
        img_path = "input.jpg"
        photo_file.download(img_path)

        img = Image.open(img_path).convert("RGB")
        label, confidence = predict_image(img)

        update.message.reply_text(
            f"🔍 *Prediction: {label}*\n"
            f"📊 Confidence: {confidence:.2f}% 🌡️",
            parse_mode="Markdown"
        )

    except Exception as e:
        logger.error(f"Error: {e}")
        update.message.reply_text("❌ Something went wrong. Try again.")

def main():
    updater = Updater(API_KEY, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(MessageHandler(Filters.photo, handle_image))

    updater.start_polling(drop_pending_updates=True)
    logger.info("Bot is running...")
    updater.idle()

if __name__ == "__main__":
    main()
