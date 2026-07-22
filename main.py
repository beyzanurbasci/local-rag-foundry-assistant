from foundry_local_sdk import FoundryLocalManager, Configuration
import time

def main():
    # 1. Sistemin başlatılması
    config = Configuration(app_name="LocalRAGApp")
    FoundryLocalManager.initialize(config=config)
    manager = FoundryLocalManager.instance
    
    # Smollm veya Qwen 0.5B denemek daha güvenli
    target_model_id = "smollm3-3b-generic-cpu:1"
    
    # Katalogdan modeli bul
    target_model = next((m for m in manager.catalog.list_models() if m.id == target_model_id), None)
    
    # 2. İndirme kontrolü (Girintiler tam 4 boşluk)
    if not target_model.is_cached:
        print("Model indirme işlemi başlatıldı. Lütfen bekleyin...")
        target_model.download()
        
        while not target_model.is_cached:
            print(".", end="", flush=True)
            time.sleep(2)
        print("\nİndirme başarıyla tamamlandı!")
    else:
        print("Model zaten sistemde mevcut.")

    # 3. Yükleme ve ChatClient
    if not target_model.is_loaded:
        target_model.load()
        
    chat_client = target_model.get_chat_client()
    response = chat_client.complete_chat(messages=[{"role": "user", "content": "Hello, world!"}])
    print("\nModelin Cevabı:", response)

if __name__ == "__main__":
    main()