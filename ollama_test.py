from openai import OpenAI

# Initialize the client with local Ollama settings
client = OpenAI(
    base_url='http://100.91.44.103:11434/v1/',
    api_key='ollama', # Required by the SDK, but ignored by Ollama
)

# Call the local model
chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': 'Explain what quantization is in three sentences.',
        }
    ],
    model='qwen3.6:latest', 
    extra_body={"think": False} 
)

print(chat_completion.choices[0].message)
