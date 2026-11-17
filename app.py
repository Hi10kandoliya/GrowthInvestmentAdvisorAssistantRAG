import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Growth Investment - Advisor Assistant RAG", layout="wide")

@st.cache_resource
def load_rag_system():
    # Load sample dataset or create mock data if missing
    try:
        df = pd.read_csv("sample_policy_db_comprehensive.csv")
    except FileNotFoundError:
        data = {
            'doc_id': ['DOC-001', 'DOC-002', 'DOC-003'],
            'title': ['EMEA Commission Policy', 'LATAM Commission Policy', 'Executive Compensation Policy'],
            'version': ['1.0', '1.0', '2.0'],
            'status': ['Active', 'Active', 'Active'],
            'review_date_verified': [True, True, True],
            'access_label': ['Advisor', 'Advisor', 'Senior Leader'],
            'tenant_id': ['TENANT-GROWTH-01', 'TENANT-GROWTH-01', 'TENANT-GROWTH-01'],
            'source_link': ['https://internal.growth/policy/emea', 'https://internal.growth/policy/latam', 'https://internal.growth/policy/exec'],
            'document_text': [
                'Mutual funds carry a new investment commission of 1.00% and annual trail of 0.25% in EMEA.',
                'Latin America mutual funds new investment commission is 1.00% subject to CVM and local addenda.',
                'Executive compensation bonuses are strictly restricted to senior leadership roles.'
            ]
        }
        df = pd.DataFrame(data)
        df.to_csv('sample_policy_db.csv', index=False)
    
    return df

class StreamlitAdvisorRAG:
    def __init__(self, df):
        self.df = df
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self._prepare_corpus()

    def _prepare_corpus(self):
        # Enforce governance rules: Active status, verified review dates, allowed labels
        valid_mask = (
            (self.df['status'] == 'Active') & 
            (self.df['review_date_verified'] == True) &
            (self.df['access_label'].isin(['Public', 'Advisor', 'Internal', 'Senior Leader']))
        )
        self.active_df = self.df[valid_mask].reset_index(drop=True)
        if not self.active_df.empty:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.active_df['document_text'])
        else:
            self.tfidf_matrix = None

    def retrieve(self, query, user_role, tenant_id, top_k=3):
        tenant_filtered = self.active_df[self.active_df['tenant_id'] == tenant_id]
        if tenant_filtered.empty or self.tfidf_matrix is None:
            return []

        # Enforce Role-Based Access Control (RBAC) at retrieval time
        role_filtered = tenant_filtered[
            tenant_filtered['access_label'].apply(lambda label: self._check_permission(user_role, label))
        ]
        
        if role_filtered.empty:
            return []

        sub_indices = role_filtered.index.tolist()
        sub_matrix = self.tfidf_matrix[sub_indices]
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, sub_matrix).flatten()
        
        if len(similarities) == 0 or np.max(similarities) == 0:
            return []

        top_indices_local = similarities.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_indices_local:
            score = similarities[idx]
            if score > 0.05:
                doc = role_filtered.iloc[idx]
                results.append({
                    'doc_id': doc['doc_id'],
                    'title': doc['title'],
                    'version': doc['version'],
                    'score': float(score),
                    'text': doc['document_text'],
                    'source_link': doc['source_link'],
                    'access_label': doc['access_label']
                })
        return results

    def _check_permission(self, user_role, access_label):
        hierarchy = {
            'Advisor': ['Advisor', 'Public'], 
            'Senior Leader': ['Advisor', 'Internal', 'Public', 'Senior Leader'], 
            'Admin': ['Advisor', 'Internal', 'Public', 'Senior Leader', 'HR']
        }
        return access_label in hierarchy.get(user_role, ['Public'])

# Streamlit UI Layout
st.title("Growth Investment Knowledge Assistant (RAG Prototype)")
st.markdown("*Permission-aware, metadata-filtered policy assistant complying with regional regulatory controls.*")

df_corpus = load_rag_system()
rag_app = StreamlitAdvisorRAG(df_corpus)

with st.sidebar:
    st.header("User & Session Settings")
    user_role = st.selectbox("Select User Role / Persona", ["Advisor", "Senior Leader", "Admin"])
    tenant_id = st.text_input("Tenant ID", value="TENANT-GROWTH-01")
    st.markdown("---")
    st.markdown("### Governance Controls")
    st.info("Active filters exclude unverified review dates and enforce RBAC prior to model ingestion.")

query = st.text_input("Ask a policy or commission question:", placeholder="e.g., What is the mutual fund commission in EMEA?")

if st.button("Submit Query"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Executing permission-aware retrieval and generating response..."):
            retrieved_docs = rag_app.retrieve(query, user_role, tenant_id)
            
            st.markdown("### Response")
            if not retrieved_docs:
                st.error("I am unable to answer this question based on currently verified and approved policy documents. Escalating to policy owner.")
            else:
                top_doc = retrieved_docs[0]
                st.success(f"**Answer (Grounded & Cited):**\n\nAccording to **{top_doc['title']}** (Version {top_doc['version']}):\n> {top_doc['text']}")
                st.markdown(f"**Source Link:** [{top_doc['source_link']}]({top_doc['source_link']})")
                
                with st.expander("Inspection & Retrieval Details"):
                    st.write(f"**Matched Access Label:** {top_doc['access_label']}")
                    st.write(f"**Relevance Score:** {top_doc['score']:.4f}")
                    st.json(retrieved_docs)
