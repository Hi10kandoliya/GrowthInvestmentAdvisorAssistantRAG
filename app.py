import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Growth Investment - Advisor Assistant RAG", layout="wide")

@st.cache_resource
def load_rag_system():
    # Load comprehensive policy dataset or create it if missing
    try:
        df = pd.read_csv("sample_policy_db_comprehensive.csv")
    except FileNotFoundError:
        data = [
            {
                "doc_id": "DOC-EMEA-01",
                "title": "EMEA Commission-Based Investment Products Policy",
                "region": "EMEA",
                "version": "1.0",
                "status": "Active",
                "effective_date": "2027-01-01",
                "review_date": "2026-11-01",
                "review_date_verified": True,
                "access_label": "Advisor",
                "tenant_id": "TENANT-GROWTH-01",
                "source_link": "https://internal.growth/policy/emea-v1",
                "document_text": "EMEA regional policy framework covering MiFID II, FCA rules, and product commission schedules for mutual funds (1.00%), equity investments ($25 fixed / 0.25%), and fixed deposits (0.50%)."
            },
            {
                "doc_id": "DOC-LATAM-01",
                "title": "Latin America (LATAM) Commission-Based Investment Products Policy",
                "region": "LATAM",
                "version": "1.0",
                "status": "Active",
                "effective_date": "2018-01-01",
                "review_date": "2026-11-01",
                "review_date_verified": True,
                "access_label": "Advisor",
                "tenant_id": "TENANT-GROWTH-01",
                "source_link": "https://internal.growth/policy/latam-v1",
                "document_text": "LATAM regional baseline adapting CVM Resolution 179 (Brazil), CNBV (Mexico), SFC (Colombia), and CMF (Chile) rules. Mutual funds new investment commission reference rate 2.00%."
            },
            {
                "doc_id": "DOC-APAC-01",
                "title": "Asia-Pacific (APAC) Commission-Based Investment Products Policy",
                "region": "APAC",
                "version": "1.0",
                "status": "Active",
                "effective_date": "2024-01-01",
                "review_date": "2026-11-01",
                "review_date_verified": True,
                "access_label": "Advisor",
                "tenant_id": "TENANT-GROWTH-01",
                "source_link": "https://internal.growth/policy/apac-v1",
                "document_text": "APAC regional framework covering Singapore Financial Advisers Act and MAS notices/guidelines, Australia Corporations Act 2001 and ASIC guidance, and Hong Kong SFC Code of Conduct regarding investment advisory services and inducement rules."
            },
            {
                "doc_id": "DOC-EXEC-01",
                "title": "Executive Compensation and Discretionary Bonus Policy",
                "region": "Global",
                "version": "1.0",
                "status": "Active",
                "effective_date": "2026-01-01",
                "review_date": "2026-11-01",
                "review_date_verified": True,
                "access_label": "Senior Leader",
                "tenant_id": "TENANT-GROWTH-01",
                "source_link": "https://internal.growth/policy/exec-comp",
                "document_text": "Executive compensation guidelines and discretionary bonus structures restricted strictly to senior leadership roles and HR audit oversight."
            }
        ]
        df = pd.DataFrame(data)
        df.to_csv('sample_policy_db_comprehensive.csv', index=False)
    
    return df

class StreamlitAdvisorRAG:
    def __init__(self, df):
        self.df = df
        # Upgrade to unigrams and bigrams for better phrase matching (e.g., "Singapore", "APAC")
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self._prepare_corpus()

    def _prepare_corpus(self):
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

    def retrieve(self, query, user_role, tenant_id, top_k=1):
        tenant_filtered = self.active_df[self.active_df['tenant_id'] == tenant_id]
        if tenant_filtered.empty or self.tfidf_matrix is None:
            return []

        # Enforce Role-Based Access Control (RBAC)
        role_filtered = tenant_filtered[
            tenant_filtered['access_label'].apply(lambda label: self._check_permission(user_role, label))
        ]
        
        if role_filtered.empty:
            return []

        sub_indices = role_filtered.index.tolist()
        sub_matrix = self.tfidf_matrix[sub_indices]
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, sub_matrix).flatten()
        
        if len(similarities) == 0:
            return []

        # Keyword boosting logic for precise region targeting
        query_lower = query.lower()
        for i, idx in enumerate(sub_indices):
            doc_text = self.active_df.iloc[idx]['document_text'].lower()
            if "singapore" in query_lower or "apac" in query_lower:
                if "apac" in doc_text or "singapore" in doc_text:
                    similarities[i] += 0.5
            elif "latam" in query_lower or "latin america" in query_lower:
                if "latam" in doc_text or "latin america" in doc_text:
                    similarities[i] += 0.5
            elif "emea" in query_lower:
                if "emea" in doc_text:
                    similarities[i] += 0.5

        top_idx_local = similarities.argmax()
        score = similarities[top_idx_local]
        
        if score < 0.02:
            return [] # Triggers correct abstention

        best_doc_idx = sub_indices[top_idx_local]
        doc = self.active_df.iloc[best_doc_idx]
        
        return [{
            'doc_id': doc['doc_id'],
            'title': doc['title'],
            'version': doc['version'],
            'score': float(score),
            'text': doc['document_text'],
            'source_link': doc['source_link'],
            'access_label': doc['access_label']
        }]

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

query = st.text_input("Ask a policy or commission question:", placeholder="e.g., What regulatory framework and inducement rules apply to investment advisory services in Singapore (APAC)?")

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
