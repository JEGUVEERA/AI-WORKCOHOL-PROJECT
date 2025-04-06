import random
from openai import OpenAI

client = OpenAI()

def generate_posts(platform, topic):
    prompts = [
        f"Create a fun {platform} post about {topic}.",
        f"Write a {platform} post in a witty tone about {topic}.",
        f"Generate a motivational {platform} caption about {topic}."
    ]
    return [client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content for prompt in random.sample(prompts, 3)]
