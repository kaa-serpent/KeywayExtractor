from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import logging
import os
import json
import subprocess

# Load API token from token.json
with open("token.json", "r") as f:
    config = json.load(f)
TOKEN = config.get("telegramToken", "")

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# Function to handle start command
async def start(update: Update, context: CallbackContext) -> None:
    tutorial_image = "tutorial.jpg"  # Ensure this file exists in the same directory
    await update.message.reply_text(
        "Welcome! Here’s how to use this bot:\n1 - Take a picture parallel to the keyway\n2 - Modify your picture manually to get the keyway black and the lock white. Crop around the keyway and ensure only the keyway is black\n3 - Send me the picture and I'll process it in the Silca Catalogue to find the best matches")
    await update.message.reply_photo(photo=open(tutorial_image, 'rb'))


# Function to handle help command
async def help_command(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(
        "Commands:\n/start - Get started and receive a tutorial\n/help - Show this help message\nSend an image, and I will process it!")


async def handle_image(update: Update, context: CallbackContext) -> None:
    # Get the best quality photo from the message
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    file_path = "input.jpg"  # Save the file locally
    await file.download_to_drive(file_path)

    # Call the external image processing script (lockAnalyse.py) with the input image
    output = process_image(file_path)

    # Process and send back results to the user
    for match in output:
        img_path, similarity = match  # Extract file path and similarity score
        with open(img_path, 'rb') as image_file:
            await update.message.reply_photo(photo=image_file)
        await update.message.reply_text(f" Silca Ref : {os.path.splitext(os.path.basename(img_path))[0]} with a similarity Score: {similarity*100:.2f} %")

    await update.message.reply_text("Processing complete!")


# Function to process the image using external script
def process_image(image_path):
    # Call lockAnalyse.py script and pass the image path to it
    result = subprocess.run(["python", "lockAnalyse.py", image_path], capture_output=True, text=True)

    # Parse the output which should be the file paths of the top matches with similarity scores
    output_lines = result.stdout.splitlines()

    # Parse each line into (file_path, similarity) tuple
    output_files = []
    for line in output_lines:
        parts = line.split(" - ")
        if len(parts) == 2:
            file_path = parts[0]
            similarity = float(parts[1].split(": ")[1])
            output_files.append((file_path, similarity))

    return output_files


# Main function to set up the bot
def main():
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    application.run_polling()


if __name__ == "__main__":
    main()
