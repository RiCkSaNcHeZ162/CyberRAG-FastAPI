import json
import uuid

import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000/api"

# Set page layout and configuration
st.set_page_config(page_title="PDFGPT Assistant", page_icon="📝", layout="wide")

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    # Add initial greeting message
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your Cyber Security expert.",
        }
    ]


# def fetch_documents():
#     """Fetches the list of uploaded documents from the backend API."""
#     try:
#         response = requests.get(f"{API_BASE_URL}/documents")
#         if response.status_code == 200:
#             return response.json().get("documents", [])
#     except Exception as e:
#         st.error(
#             f"Failed to fetch documents. Make sure the FastAPI server is running. Error: {e}"
#         )
#     return []


# def delete_document(doc_id):
#     """Deletes a document using the backend API."""
#     try:
#         response = requests.delete(f"{API_BASE_URL}/documents/{doc_id}")
#         if response.status_code == 200:
#             st.toast("Document deleted successfully!", icon="🗑️")
#             st.rerun()
#         else:
#             st.error("Failed to delete document.")
#     except Exception as e:
#         st.error(f"Error: {e}")


# Main Layout
st.title("🚀 CyberRAG - GenAI Powered Cyber Security Audit Intelligence")
st.caption("Multimodal RAG assistant for DORA, NIS2 Directive, RBI IT Outsourcing "
           "Guidelines, RBI Operational Resilience Guidance Note, ISO/IEC 27000, "
           "and ISO/IEC 27002 — query across text, documents, and images for "
           "grounded regulatory and cybersecurity answers.")

st.warning("⚠️ For reference only. Not a substitute for legal, regulatory, or professional compliance advice. This tool is intended to support — not replace — professional judgment.")
st.divider()
# # Sidebar - Document Management
# with st.sidebar:
#     st.header("📄 Document Management")

#     # Upload section
#     uploaded_file = st.file_uploader("Upload a new PDF", type=["pdf"])
#     strategy = st.selectbox(
#         "Chunking Strategy", ["semantic", "recursive", "context_aware"], index=0
#     )

#     if st.button("Upload & Process", use_container_width=True):
#         if uploaded_file is not None:
#             with st.spinner("Processing PDF... Creating Embeddings & Chunks..."):
#                 try:
#                     files = {
#                         "file": (
#                             uploaded_file.name,
#                             uploaded_file.getvalue(),
#                             "application/pdf",
#                         )
#                     }
#                     data = {"chunking_strategy": strategy}
#                     response = requests.post(
#                         f"{API_BASE_URL}/documents/upload", files=files, data=data
#                     )

#                     if response.status_code == 200:
#                         res_data = response.json()
#                         st.success(
#                             f"Success! {res_data.get('chunks_created')} chunks & vectors stored."
#                         )
#                         st.rerun()
#                     else:
#                         st.error(f"Upload failed: {response.text}")
#                 except Exception as e:
#                     st.error(
#                         f"Error connecting to server. Is the FastAPI backend running?: {e}"
#                     )
#         else:
#             st.warning("Please select a file first.")

#     st.divider()

#     # List documents
#     st.subheader("📚 Available Documents")
#     docs = fetch_documents()
#     if not docs:
#         st.info("No documents found in the database.")
#     else:
#         for doc in docs:
#             with st.container():
#                 st.write(f"**{doc.get('file_name', 'Unknown')}**")
#                 st.caption(
#                     f"ID: {doc.get('doc_id')} | {doc.get('total_pages', 0)} Pages | Strategy: {doc.get('chunking_strategy', 'N/A')}"
#                 )
#                 if st.button(
#                     "Delete",
#                     key=f"del_{doc.get('doc_id')}",
#                     help="Delete this document and all its vectors",
#                 ):
#                     delete_document(doc.get("doc_id"))
#                 st.divider()

#     st.markdown("---")
#     st.subheader("System Actions")
#     if st.button("🗑️ Clear Chat History", use_container_width=True):
#         st.session_state.messages = [
#             {"role": "assistant", "content": "Memory cleared! How else can I assist?"}
#         ]
#         # Start a new physical session as well
#         st.session_state.session_id = str(uuid.uuid4())
#         st.rerun()

# Chat interface
# Display all prior messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Append the user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Render user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Render assistant placeholder to stream the response into
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Connect to FastAPI streaming endpoint
            with requests.post(
                f"{API_BASE_URL}/query/stream",
                json={"question": prompt, "session_id": st.session_state.session_id},
                stream=True,
            ) as response:
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode("utf-8")
                            if decoded_line.startswith("data: "):
                                try:
                                    data = json.loads(decoded_line[6:])
                                    # Collect and show tokens
                                    if "token" in data and data["token"]:
                                        full_response += data["token"]
                                        message_placeholder.markdown(
                                            full_response + "▌"
                                        )

                                    if data.get("done", False):
                                        break
                                except json.JSONDecodeError:
                                    continue

                    # Display the final complete response without cursor
                    message_placeholder.markdown(full_response)
                else:
                    error_msg = f"Error computing answer: HTTP {response.status_code} - {response.text}"
                    message_placeholder.error(error_msg)
                    full_response = error_msg
        except Exception as e:
            error_msg = (
                f"Connection error: Make sure the FastAPI backend is running! ({e})"
            )
            message_placeholder.error(error_msg)
            full_response = error_msg

        # Finish by appending the full response strictly
        if full_response:
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
