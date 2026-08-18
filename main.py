import json
import math
import sqlite3
from foundry_local_sdk import Configuration, FoundryLocalManager

# Veritabanı dosya adı
DB_NAME = "rag_database.db"


def load_raw_documents_from_file(filename="knowledge.txt"):
  """Hocanın istediği gibi: Bilgileri kodun içine yazmak yerine harici bir dosyadan (knowledge.txt) okur."""
  try:
    with open(filename, "r", encoding="utf-8") as f:
      file_content = f.read()
    # Paragraflara ayırarak ham doküman listesi oluşturuyoruz
    raw_docs = [doc.strip() for doc in file_content.split("\n\n") if doc.strip()]
    return raw_docs
  except FileNotFoundError:
    print(
        f"⚠️ '{filename}' dosyası bulunamadı! Lütfen proje klasörüne bu"
        " dosyayı oluşturun."
    )
    return []


def chunk_text(text, max_chunk_size=300):
  """Uzun metinleri paragraf veya karakter sınırına göre küçük parçalara (chunks) ayırır."""
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
  """id, content ve embedding alanlarına sahip SQLite tablosunu kurar."""
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

  cursor.execute("SELECT COUNT(*) FROM documents")
  count = cursor.fetchone()[0]

  if count == 0:
    for doc, emb in zip(docs, embeddings):
      cursor.execute(
          "INSERT INTO documents (content, embedding) VALUES (?, ?)",
          (doc, json.dumps(emb)),
      )
    conn.commit()
    print("✅ Chunks and embeddings successfully saved to SQLite.")
  else:
    print("ℹ️ Documents already exist in SQLite database.")

  conn.close()


def load_documents_from_db():
  """Veritabanındaki tüm chunk'ları ve embedding'leri yükler."""
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
  """Compute cosine similarity between two vectors."""
  dot = sum(x * y for x, y in zip(a, b))
  norm_a = math.sqrt(sum(x * x for x in a))
  norm_b = math.sqrt(sum(x * x for x in b))
  return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def find_relevant(query_embedding, doc_embeddings, top_k=2):
  """Return the indices and scores of the top-k most similar chunks."""
  scores = []
  for i, doc_emb in enumerate(doc_embeddings):
    score = cosine_similarity(query_embedding, doc_emb)
    scores.append((i, score))
  scores.sort(key=lambda x: x[1], reverse=True)
  return scores[:top_k]


def main():
  # 1. SQLite Veritabanını Başlat
  init_db()

  # 2. Harici dosyadan (knowledge.txt) ham dokümanları oku
  raw_documents = load_raw_documents_from_file("knowledge.txt")
  if not raw_documents:
    return

  # 3. Dokümanları otomatik olarak küçük parçalara (chunks) ayır
  all_chunks = []
  for raw_doc in raw_documents:
    chunks = chunk_text(raw_doc)
    all_chunks.extend(chunks)

  print(f"Total chunks generated: {len(all_chunks)}")

  # Initialize the SDK
  config = Configuration(app_name="foundry_local_rag")
  FoundryLocalManager.initialize(config)
  manager = FoundryLocalManager.instance

  # Load the embedding model
  embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
  embedding_model.download(
      lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True)
  )
  print()
  embedding_model.load()
  embedding_client = embedding_model.get_embedding_client()

  # 4. Embed all chunks (Eğer veritabanında yoksa üretip SQLite'a kaydediyoruz)
  docs, doc_embeddings = load_documents_from_db()
  if not docs:
    response = embedding_client.generate_embeddings(all_chunks)
    doc_embeddings = [item.embedding for item in response.data]
    save_documents_to_db(all_chunks, doc_embeddings)
    docs, doc_embeddings = load_documents_from_db()

  print(f"Indexed {len(doc_embeddings)} chunks from SQLite database.")

  # Load the chat model
  chat_model = manager.catalog.get_model("qwen2.5-0.5b")
  chat_model.download(
      lambda p: print(f"\rDownloading chat model: {p:.1f}%", end="", flush=True)
  )
  print()
  chat_model.load()
  chat_client = chat_model.get_chat_client()

  print("\nModels loaded. Ready for questions.")
  print('\nType "quit" to exit.\n')

  # Interactive query loop
  while True:
    query = input("Question: ").strip()
    if not query or query.lower() == "quit":
      break

    # Embed the query
    query_response = embedding_client.generate_embedding(query)
    query_embedding = query_response.data[0].embedding

    # Retrieve the most relevant chunks from SQLite-backed database
    results = find_relevant(query_embedding, doc_embeddings, top_k=2)
    context = "\n".join(f"- {docs[i]}" for i, _ in results)

    # Build the prompt with retrieved context and system rules
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the user's question using only the provided context. "
                "If the context doesn't contain enough information, say so.\n\n"
                f"Context:\n{context}"
            ),
        },
        {"role": "user", "content": query},
    ]

    # Stream the response safely
    print("Answer: ", end="", flush=True)
    for chunk in chat_client.complete_streaming_chat(messages):
      if hasattr(chunk, "choices") and chunk.choices:
        delta = getattr(chunk.choices[0], "delta", None)
        if delta and hasattr(delta, "content") and delta.content:
          print(delta.content, end="", flush=True)
    print("\n")

  # Clean up
  embedding_model.unload()
  chat_model.unload()
  print("Models unloaded. Done!")


if __name__ == "__main__":
  main()