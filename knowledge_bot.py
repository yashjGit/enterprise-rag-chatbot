import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

knowledge_base = {
    'doc_id': ['HR-001', 'HR-002', 'IT-001', 'IT-002', 'FIN-001'],
    'topic': ['Paid Time Off', 'Remote Work', 'Password Reset', 'Software Installation', 'Travel Reimbursement'],
    'content': [
        'Employees are entitled to 20 days of paid time off (PTO) per year. Unused PTO does not roll over. To request time off, use the Workday portal.',
        'Employees may work remotely up to 2 days a week. Core collaborative hours are 10 AM to 3 PM EST. A one-time $500 stipend is provided for home office setup.',
        'To reset your corporate password, visit password.company.internal. You must use the Microsoft Authenticator app to verify your identity. Passwords expire every 90 days.',
        'Standard software like Office 365 and Slack can be installed via the Self-Service portal. Developer tools require manager approval and an IT ticket.',
        'Business travel expenses must be submitted via Concur within 30 days of the trip. The maximum daily allowance for meals is $75. Receipts are required for anything over $25.'
    ]
}

df_kb = pd.DataFrame(knowledge_base)

print("🧠 Downloading Neural Network & Embedding Knowledge Base... (Takes a moment on first run)")

model = SentenceTransformer('all-MiniLM-L6-v2')

doc_embeddings = model.encode(df_kb['content'].tolist())

def ask_semantic_bot(user_question):
    print(f"\n👤 USER ASKS: '{user_question}'")
    print("-" * 50)
    
    question_embedding = model.encode([user_question])
    
    similarity_scores = cosine_similarity(question_embedding, doc_embeddings)
    
    best_match_index = np.argmax(similarity_scores[0])
    highest_score = similarity_scores[0][best_match_index]
    
    if highest_score > 0.35:
        source_doc = df_kb['doc_id'].iloc[best_match_index]
        topic = df_kb['topic'].iloc[best_match_index]
        factual_answer = df_kb['content'].iloc[best_match_index]
        
        print(f"🤖 BOT ANSWER: {factual_answer}")
        print(f"📎 Source: {source_doc} ({topic}) | Confidence Score: {round(highest_score * 100, 1)}%")
    else:
        print("🤖 BOT ANSWER: I'm sorry, I cannot find the answer to that in the company knowledge base. Please contact HR or IT support.")

ask_semantic_bot("How many vacation days do I get and how do I ask for them?")
ask_semantic_bot("I forgot my password, how do I get a new one?")
ask_semantic_bot("Can I work from home every day?")
ask_semantic_bot("What is the company policy on bringing pets to the office?")