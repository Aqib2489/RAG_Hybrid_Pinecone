import requests
from requests.auth import HTTPBasicAuth
import streamlit as st

st.set_page_config(page_title="🎓 Grad Review Assistant", layout="centered")
st.title("🎓 Graduate Application Reviewer (GPT-4o mini)")

# --------------------------
# SETTINGS
# --------------------------
DEFAULT_API = "http://127.0.0.1:8000/review"
api_url = DEFAULT_API
with st.expander("🔧 Settings", expanded=False):
    api_url = st.text_input("Backend API URL", value=api_url)

# --------------------------
# LOGIN
# --------------------------
if "auth" not in st.session_state:
    st.session_state.auth = None

st.subheader("🔐 Login to Access Tool")
username = st.text_input("Username", key="username")
password = st.text_input("Password", type="password", key="password")

if st.button("Sign in"):
    if not username or not password:
        st.error("Please enter both username and password.")
    else:
        st.session_state.auth = HTTPBasicAuth(username, password)
        st.success("✅ Logged in successfully!")

if st.session_state.auth is None:
    st.stop()

# --------------------------
# REVIEW TOOL
# --------------------------
st.subheader("📁 Upload and Review Application")
applicant_id = st.text_input("Applicant ID", value="A-0001")
uploaded = st.file_uploader("Upload applicant PDFs (CV, essay, etc.)", type=["pdf"], accept_multiple_files=True)

if st.button("🚀 Run Review", type="primary"):
    if not uploaded:
        st.error("Please upload at least one PDF.")
    else:
        files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded]
        data = {"applicant_id": applicant_id}

        with st.spinner("Analyzing and preparing DOCX report..."):
            try:
                resp = requests.post(api_url, data=data, files=files, auth=st.session_state.auth, timeout=180)

                if resp.status_code == 200:
                    # Expecting a DOCX file now (binary content)
                    ctype = resp.headers.get("content-type", "")
                    if "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in ctype:
                        st.success("✅ Review completed! Your DOCX report is ready.")
                        st.divider()
                        st.markdown("### 💾 Download DOCX Report")
                        st.download_button(
                            label="⬇️ Download Report (.docx)",
                            data=resp.content,
                            file_name=f"{applicant_id}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            type="primary"
                        )
                    else:
                        # If backend returned something unexpected
                        st.error("Unexpected response format from server.")
                        st.caption(f"Content-Type: {ctype}")
                        st.code(resp.text[:1000])
                elif resp.status_code == 401:
                    st.error("❌ Unauthorized: wrong username or password.")
                else:
                    st.error(f"⚠️ Error {resp.status_code}")
                    st.code(resp.text[:2000])
            except Exception as e:
                st.error(f"Request failed: {e}")

