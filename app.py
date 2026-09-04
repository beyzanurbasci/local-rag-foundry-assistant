import json
import math
import sqlite3
import streamlit as st
from foundry_local_sdk import Configuration, FoundryLocalManager
from pypdf import PdfReader

DB_NAME = "rag_database.db"

# Streamlit Sayfa Yapılandırması
st.set_page_config(
    page_title="Grand Horizon Hotel Assistant", page_icon="🏨", layout="centered"
)

st.title("🏨 Grand Horizon Hotel Assistant")
st.markdown(
    "Welcome! Ask anything about our hotel services, rooms, spa, or"
    " policies."
)


# Veritabanı ve Modelleri Önbellekleme (Cache)
@st.cache_resource
def init_db():
  conn = sqlite3.connect(DB_NAME, timeout=10.0)
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


@st.cache_resource
def load_db_data():
  init_db()
  conn = sqlite3.connect(DB_NAME, timeout=10.0)
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


@st.cache_resource
def init_sdk_and_models():
  config = Configuration(app_name="foundry_local_rag_ui")
  try:
    FoundryLocalManager.initialize(config)
  except Exception:
    pass  # Singleton zaten başlatıldıysa hatayı yoksay

  manager = FoundryLocalManager.instance

  # Embedding Modeli
  emb_model = manager.catalog.get_model("qwen3-embedding-0.6b")
  emb_model.download(lambda p: None)
  emb_model.load()
  emb_client = emb_model.get_embedding_client()

  # Chat Modeli
  chat_model = manager.catalog.get_model("qwen2.5-0.5b")
  chat_model.download(lambda p: None)
  chat_model.load()
  chat_client = chat_model.get_chat_client()

  return emb_client, chat_client


# Kaynakları Yükle
with st.spinner("🤖 Asistan hazırlanıyor, lütfen bekleyin..."):
  docs, doc_embeddings = load_db_data()
  embedding_client, chat_client = init_sdk_and_models()


def cosine_similarity(a, b):
  dot = sum(x * y for x, y in zip(a, b))
  norm_a = math.sqrt(sum(x * x for x in a))
  norm_b = math.sqrt(sum(x * x for x in b))
  return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# Optimize Edilmiş Arama ve Eşik (Threshold) Filtresi
def find_relevant(query_embedding, doc_embeddings, top_k=2, threshold=0.25):
  scores = []
  for i, doc_emb in enumerate(doc_embeddings):
    score = cosine_similarity(query_embedding, doc_emb)
    if score >= threshold:
      scores.append((i, score))
  scores.sort(key=lambda x: x[1], reverse=True)
  return scores[:top_k]


# --- PDF İŞLEME YARDIMCISI ---
def extract_text_from_pdf(uploaded_file):
  reader = PdfReader(uploaded_file)
  text = ""
  for page in reader.pages:
    extracted = page.extract_text()
    if extracted:
      text += extracted + "\n"
  return text


# Sohbet Hafızası Yönetimi
if "messages" not in st.session_state:
  st.session_state.messages = []

# Sidebar (Yan Menü) Kontrolleri ve Dosya Yükleme
with st.sidebar:
  st.header("⚙️ Chat Control")
  if st.button("🗑️ Clear Conversation History"):
    st.session_state.messages = []
    st.rerun()

  st.markdown("---")
  st.header("📂 Document Upload")
  uploaded_pdf = st.file_uploader(
      "Upload a PDF document to expand knowledge base", type=["pdf"]
  )

  if uploaded_pdf is not None:
    pdf_text = extract_text_from_pdf(uploaded_pdf)
    if pdf_text.strip():
      chunks = [pdf_text[i : i + 500] for i in range(0, len(pdf_text), 400)]

      conn = sqlite3.connect(DB_NAME, timeout=10.0)
      cursor = conn.cursor()
      for chunk in chunks:
        if len(chunk.strip()) > 20:
          emb_resp = embedding_client.generate_embedding(chunk)
          chunk_emb = emb_resp.data[0].embedding
          cursor.execute(
              "INSERT INTO documents (content, embedding) VALUES (?, ?)",
              (chunk, json.dumps(chunk_emb)),
          )
      conn.commit()
      conn.close()

      st.cache_resource.clear()
      st.success(
          f"✨ PDF başarıyla yüklendi ve {len(chunks)} parça bilgi sisteme"
          " eklendi! Lütfen sayfayı yenileyin."
      )
    else:
      st.warning("⚠️ PDF dosyasından okunabilir metin çıkarılamadı.")

  st.markdown("---")
  st.info(
      "Optimized RAG pipeline with PDF Uploader & Threshold Filtering, powered"
      " by Foundry Local SDK."
  )

# Geçmiş Mesajları Ekranda Listele
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# Kullanıcı Girdi Alanı
if prompt := st.chat_input("How can I help you today?"):
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  # Vektör Arama ve Optimize Edilmiş Context Filtrelemesi
  query_response = embedding_client.generate_embedding(prompt)
  query_embedding = query_response.data[0].embedding
  results = find_relevant(
      query_embedding, doc_embeddings, top_k=2, threshold=0.25
  )

  if results:
    context = "\n".join(f"- {docs[i]}" for i, _ in results)
  else:
    context = "No relevant hotel information found in the database."

  messages = [
      {
          "role": "system",
          "content": (
              "You are a professional assistant for the Grand Horizon Hotel."
              " Answer the guest's question using only the provided context."
              " If no relevant context is found, politely inform the guest"
              " that you can only answer questions related to hotel services."
              f"\n\nContext:\n{context}"
          ),
      },
  ]
  for m in st.session_state.messages:
    messages.append({"role": m["role"], "content": m["content"]})

  # Asistan Yanıtı (Streaming)
  with st.chat_message("assistant"):
    response_container = st.empty()
    full_response = ""

    for chunk in chat_client.complete_streaming_chat(messages):
      if hasattr(chunk, "choices") and chunk.choices:
        delta = getattr(chunk.choices[0], "delta", None)
        if delta and hasattr(delta, "content") and delta.content:
          full_response += delta.content
          response_container.markdown(full_response + "▌")

    response_container.markdown(full_response)
  st.session_state.messages.append(
      {"role": "assistant", "content": full_response}
  )