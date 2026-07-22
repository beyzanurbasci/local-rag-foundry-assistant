from foundry_local_sdk import FoundryLocalManager, Configuration

# 1. Initialize the configuration and the model manager
config = Configuration(app_name="LocalRAGApp")
manager = FoundryLocalManager(config=config)

# 2. Access the model through the catalog using the alias
target_model = manager.catalog.get_model("smollm3-3b")

# 3. Check if the model exists before proceeding
if target_model is None:
    print("Error: Model 'smollm3-3b' not found in catalog.")
else:
    # 4. Download process: checking if it is cached locally
    if not target_model.is_cached:
        print("Model not found in cache. Download initiated, please wait...")
        target_model.download()
        print("Download completed successfully!")
    else:
        print("Model already cached.")

    # 5. Loading and ChatClient initialization
    print("Loading model into memory, please wait...")
    if not target_model.is_loaded:
        target_model.load()
    print("Model loaded! Fetching response...\n")

    chat_client = target_model.get_chat_client()

    # 6. Send message
    response = chat_client.complete_chat(messages=[{"role": "user", "content": "Hello, world!"}])

    # 7. Extract and print only the content
    print("Model Response:", response.choices[0].message.content)