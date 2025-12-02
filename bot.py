#!/usr/bin/env python3
"""
bot.py
Telegram bot for tomato leaf disease detection using DenseNet121.
"""

import os
import io
import json
import numpy as np
from PIL import Image
import cv2
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

from telegram import Update, InputMediaPhoto
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# ========================= CONFIG =========================
TOKEN = "8467005268:AAF-0lblz28WB3pgCzj_qxkyYwzQRKHDfMo"
MODEL_PATH = "best_DenseNet121.h5"
METADATA_PATH = "BEST_MODEL_METADATA.json"
IMAGE_SIZE = (224, 224)
TOP_K = 3

# ========================= LOGGING =========================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ========================= UTILS =========================
def safe_load_metadata(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)

metadata = safe_load_metadata(METADATA_PATH)
class_names = metadata.get("class_names", [f"Class_{i}" for i in range(10)])

# Load Model
print("Loading model...", MODEL_PATH)
try:
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded.")
except Exception as e:
    print("ERROR loading model:", e)
    model = None

# Override image size if metadata contains one
if "image_size" in metadata:
    IMAGE_SIZE = tuple(metadata["image_size"])

# ========================= IMAGE PREPROCESSING =========================
def preprocess_image_pil(pil_img, target_size=IMAGE_SIZE):
    img = pil_img.convert("RGB").resize(target_size)
    arr = img_to_array(img)
    arr = densenet_preprocess(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr, np.uint8(np.array(img))

def predict_from_image(pil_img):
    if model is None:
        raise RuntimeError("Model not loaded")
    X, raw_uint8 = preprocess_image_pil(pil_img)
    preds = model.predict(X, verbose=0)[0]
    top_indices = preds.argsort()[-TOP_K:][::-1]
    top = [(class_names[int(i)], float(preds[int(i)])) for i in top_indices]
    return top, preds, raw_uint8

# ========================= GRAD-CAM =========================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(heatmap, orig_uint8, alpha=0.4):
    heatmap = cv2.resize(heatmap, (orig_uint8.shape[1], orig_uint8.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(jet, alpha, orig_uint8, 1 - alpha, 0)
    return overlay

# ========================= BOT HANDLERS =========================
def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "🍅 *TomatoLeafBot*\n\n"
        "Send me a photo of a tomato leaf, and I'll detect the disease.\n"
        "To receive a Grad-CAM heatmap, add: `gradcam` in the caption.\n",
        parse_mode="Markdown"
    )

def info(update: Update, context: CallbackContext):
    text = (
        f"Model: {metadata.get('best_model_name', 'DenseNet121')}\n"
        f"Classes: {len(class_names)}\n"
        f"Input Size: {IMAGE_SIZE}\n"
    )
    update.message.reply_text(text)

def handle_photo(update: Update, context: CallbackContext):
    try:
        message = update.message
        caption = (message.caption or "").lower()
        wants_gradcam = "gradcam" in caption

        file = message.photo[-1].get_file()
        data = io.BytesIO()
        file.download(out=data)
        data.seek(0)

        pil_img = Image.open(data)

        topk, probs, orig_uint8 = predict_from_image(pil_img)
        top_label, top_conf = topk[0]

        reply = f"🔍 *Prediction:* {top_label}\n📊 *Confidence:* {top_conf:.4f}\n\n"
        for cls, score in topk:
            reply += f"• {cls}: {score:.4f}\n"

        update.message.reply_text(reply, parse_mode="Markdown")

        if wants_gradcam:
            last_layer = metadata.get("last_conv_layer", "conv5_block16_concat")
            X, _ = preprocess_image_pil(pil_img)
            heatmap = make_gradcam_heatmap(X, model, last_layer)

            overlay = overlay_heatmap(heatmap, orig_uint8)
            _, buff = cv2.imencode(".jpg", overlay)
            img_bytes = io.BytesIO(buff.tobytes())
            img_bytes.name = "gradcam.jpg"
            img_bytes.seek(0)

            update.message.reply_photo(img_bytes, caption="🔥 Grad-CAM")

    except Exception as e:
        logger.exception("Error handling image")
        update.message.reply_text(f"⚠️ Error: {str(e)}")

def error_handler(update: Update, context: CallbackContext):
    logger.error(msg="Exception occurred", exc_info=context.error)

def main():
    if TOKEN.startswith("REPLACE"):
        print("ERROR: Add your Telegram BOT TOKEN!")
        return

    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("info", info))
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))
    dp.add_error_handler(error_handler)

    print("Bot started (polling)...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
