import discord
import keras
from chatbot2 import Encoder, Decoder, BahdanauAttention, evaluate

# Cargar los modelos (si es necesario)
encoder = keras.models.load_model('encoder_model.keras', custom_objects={'Encoder': Encoder, 'BahdanauAttention': BahdanauAttention})
decoder = keras.models.load_model('decoder_model.keras', custom_objects={'Decoder': Decoder, 'BahdanauAttention': BahdanauAttention})

class MyClient(discord.Client):
    async def on_ready(self):
        print(f"Logged on as {self.user}!")

    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return
        if message.content.startswith("@SI "):
            channel = message.channel
            print(f"Message from {message.author}: {message.content}")
            result, _ = evaluate(message.content[4:], encoder=encoder, decoder=decoder)
            await channel.send(result)


intents = discord.Intents.default()
intents.message_content = True

client = MyClient(intents=intents)
client.run("<insert token here>")

