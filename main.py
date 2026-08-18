import json
import math
import sqlite3
from foundry_local_sdk import Configuration, FoundryLocalManager

# Veritabanı dosya adı
DB_NAME = "rag_database.db"

# Hocanın Week 3 için istediği ham / uzun doküman havuzu
# (İleride bunu bir dosyadan okutarak da yapabilirsin)
raw_documents = [
    (
        "Foundry Local runs AI models directly on your device without cloud"
        " connectivity. It ensures complete data privacy and low latency for"
        " enterprise applications."
    ),
    (
        "The Foundry Local SDK supports multiple programming languages including"
        " Python, C#, JavaScript, and Rust. Developers can easily integrate"
        " local models into their existing stacks."
    ),
    (
        "Embedding models convert text into numerical vectors for similarity"
        " search. This is the foundational step for building robust"
        " Retrieval-Augmented Generation (RAG) systems."
    ),
    (
        "Foundry Local uses ONNX Runtime for efficient model inference on"
        " CPUs and GPUs. This maximizes hardware utilization on local"
        " machines."
    ),
    (
        "The model catalog provides pre-optimized models that you can download"
        " and run locally. You can easily select models based on your"
        " performance requirements."
    ),
    (
        "Retrieval-augmented generation grounds model responses in your own"
        " data, reducing hallucinations and providing accurate, context-aware"
        " answers."
    ),
    (
        "Vector similarity search finds documents that are semantically close"
        " to a query by measuring the distance between their embedding vectors."
    ),
    (
        "Chat completions generate natural language responses from a prompt and"
        " context, allowing for conversational interactions with your data."
    ),
]


def chunk_text(text, max_chunk_size=300):
  """Hocanın istediği: Uzun metinleri paragraf veya karakter sınırına göre küçük parçalara (chunks) ayırır."""
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
    print("Chunks and embeddings successfully saved to SQLite.")
  else:
    print("Documents already exist in SQLite database.")

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

  # 2. Ham dokümanları otomatik olarak küçük parçalara (chunks) ayır
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

  # 3. Embed all chunks (Eğer veritabanında yoksa üretip SQLite'a kaydediyoruz)
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