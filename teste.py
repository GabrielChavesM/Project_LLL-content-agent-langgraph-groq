from langchain.embeddings import HuggingFaceEmbeddings
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print(emb.embed_query("teste de embeddings"))
