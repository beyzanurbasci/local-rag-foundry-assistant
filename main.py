import json
import math
import os
import sqlite3
from foundry_local_sdk import Configuration, FoundryLocalManager

# Veritabanı ve bilgi dosyası adları
DB_NAME = "rag_database.db"
KNOWLEDGE_FILE = "knowledge.txt"


def load_documents_from_file(filename=KNOWLEDGE_FILE):
  """Hocanın istediği gibi: Verileri kod içine gömmek yerine harici bir belgeden (dosyadan) okur."""
  if not os.path.exists(filename):
    print(
        f"⚠️ Uyarı: '{filename}' dosyası bulunamadı! Lütfen proje klasörüne bu"
        " dosyayı ekleyin."
    )
    return []

  with open(filename, "r", encoding="utf-8") as f:
    file_content = f.read()

  # Paragraflara ayırarak ham doküman listesi oluşturuyoruz
  raw_docs = [doc.strip() for doc in file_content.split("\n\n") if doc.strip()]
  return raw_docs


def chunk_text(text, max_chunk_size=300):
  """Uzun dokümanları anlam bütünlüğünü bozmadan küçük parçalara (chunks) ayırır."""
  paragraphs = text.split("\n\n")
  chunks = []

  for p in paragraphs:
    p = p.strip()
    if not p:
      continue
    if len(p) <= max_chunk_size:
      chunks.append(p)
    else:
      words = p.split()
      current_chunk = []
      current_length = 0
      for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        if current_length >= max_chunk_size:
          chunks.append(" ".join(current_chunk))
          current_chunk = []
          current_length = 0
      if current_chunk:
        chunks.append(" ".join(current_chunk))

  return chunks


def init_db():
  """SQLite veritabanını ve doküman tablosunu kurar."""
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            embedding TEXT
        )
    """)
  conn.commit()
  conn.close()


def save_documents_to_db(docs, embeddings):
  """Parçalanmış dokümanları ve embedding'lerini SQLite veritabanına kaydeder."""
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()

  # Veritabanı boşsa kayıt yap
  cursor.execute("SELECT COUNT(*) FROM documents")
  count = cursor.fetchone()[0]

  if count == 0:
    for doc, emb in zip(docs, embeddings):
      cursor.execute(
          "INSERT INTO documents (content, embedding) VALUES (?, ?)",
          (doc, json.dumps(emb)),
      )
    conn.commit()
    print("✅ Harici belgeden okunan chunks ve embedding'ler SQLite'a kaydedildi.")
  else:
    print("ℹ️ Veritabanı zaten dolu, mevcut veriler yükleniyor.")

  conn.close()


def load_documents_from_db():
  """Veritabanındaki tüm chunk'ları ve embedding'leri belleğe yükler."""
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()
  cursor.execute("SELECT content, embedding FROM documents")
  rows = cursor.fetchall()
  conn.close()

  docs = []
  doc_embeddings = []
  for row in rows:
    docs.append(row[0])
    doc_embeddings.append(json.loads(row[1]))

  return docs, doc_embeddings


def cosine_similarity(a, b):
  """İki vektör arasındaki kosinüs benzerliğini hesaplar."""
  dot = sum(x * y for x, y in zip(a, b))
  norm_a = math.sqrt(sum(x * x for x in a))
  norm_b = math.sqrt(sum(x * x for x in b))
  return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def find_relevant(query_embedding, doc_embeddings, top_k=2):
  """Sorgu vektörüne en yakın top-k doküman parçasını bulur."""
  scores = []
  for i, doc_emb in enumerate(doc_embeddings):
    score = cosine_similarity(query_embedding, doc_emb)
    scores.append((i, score))
  scores.sort(key=lambda x: x[1], reverse=True)
  return scores[:top_k]


def main():
  # 1. SQLite Veritabanını Başlat
  init_db()

  # 2. Harici belgeyi (knowledge.txt) sistem üzerinden oku
  print(f"📖 '{KNOWLEDGE_FILE}' belgesi okunuyor...")
  raw_documents = load_documents_from_file(KNOWLEDGE_FILE)
  if not raw_documents:
    print("❌ İşlem durduruldu: Geçerli bir doküman bulunamadı.")
    return

  # 3. Metinleri otomatik olarak parçala (Chunking)
  all_chunks = []
  for raw_doc in raw_documents:
    chunks = chunk_text(raw_doc)
    all_chunks.extend(chunks)

  print(f"Toplam üretilen chunk sayısı: {len(all_chunks)}")

  # 4. Foundry Local SDK ve Modelleri Başlat
  config = Configuration(app_name="hotel_rag_assistant")
  FoundryLocalManager.initialize(config)
  manager = FoundryLocalManager.instance

  # Embedding Modeli
  embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
  embedding_model.download(
      lambda p: print(f"\rEmbedding modeli indiriliyor: {p:.1f}%", end="", flush=True)
  )
  print()
  embedding_model.load()
  embedding_client = embedding_model.get_embedding_client()

  # 5. Vektörleri üret ve SQLite'a kaydet (Eğer veritabanı boşsa)
  docs, doc_embeddings = load_documents_from_db()
  if not docs:
    print("🔄 Metinler vektörleştiriliyor...")
    response = embedding_client.generate_embeddings(all_chunks)
    doc_embeddings = [item.embedding for item in response.data]
    save_documents_to_db(all_chunks, doc_embeddings)
    docs, doc_embeddings = load_documents_from_db()

  print(f"SQLite veritabanından {len(doc_embeddings)} parça yüklendi.")

  # Chat Modeli
  chat_model = manager.catalog.get_model("qwen2.5-0.5b")
  chat_model.download(
      lambda p: print(f"\rChat modeli indiriliyor: {p:.1f}%", end="", flush=True)
  )
  print()
  chat_model.load()
  chat_client = chat_model.get_chat_client()

  print(
      "\n✨ Grand Horizon Hotel Asistanı Hazır! Otelle ilgili sorularınızı"
      " sorabilirsiniz."
  )
  print('Çıkış için "quit" yazabilirsiniz.\n')

  # 6. İnteraktif Soru-Cevap Döngüsü
  while True:
    query = input("Soru (Guest Question): ").strip()
    if not query or query.lower() == "quit":
      break

    # Sorguyu vektöre dönüştür
    query_response = embedding_client.generate_embedding(query)
    query_embedding = query_response.data[0].embedding

    # En alakalı doküman parçalarını veritabanından bul
    results = find_relevant(query_embedding, doc_embeddings, top_k=2)
    context = "\n".join(f"- {docs[i]}" for i, _ in results)

    # Model için sistem kuralları ve bağlamı hazırla
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional assistant for the Grand Horizon Hotel."
                " Answer the guest's question using only the provided context"
                " from our hotel document. If the answer cannot be found in"
                " the context, politely inform the guest that you don't have"
                " that information.\n\nContext:\n"
                f"{context}"
            ),
        },
        {"role": "user", "content": query},
    ]

    # Yanıtı ekrana akıt (Streaming)
    print("Assistant: ", end="", flush=True)
    for chunk in chat_client.complete_streaming_chat(messages):
      if hasattr(chunk, "choices") and chunk.choices:
        delta = getattr(chunk.choices[0], "delta", None)
        if delta and hasattr(delta, "content") and delta.content:
          print(delta.content, end="", flush=True)
    print("\n" + "-" * 40 + "\n")

  # Temizlik
  embedding_model.unload()
  chat_model.unload()
  print("Modeller kapatıldı. İyi günler!")


if __name__ == "__main__":
  main()