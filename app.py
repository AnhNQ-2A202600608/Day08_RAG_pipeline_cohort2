import os
import sys
import warnings
import logging

# Tắt toàn bộ log cảnh báo phiền phức từ HuggingFace/transformers
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Đảm bảo đường dẫn import hoạt động
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

load_dotenv()

# Import các hàm từ RAG pipeline cá nhân
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import reorder_for_llm, format_context, local_heuristic_generation

# =============================================================================
# OPTIMIZATION: CACHING HEAVY MODELS & CORPUS DATA (Tăng tốc độ load trang)
# =============================================================================
@st.cache_resource
def get_cached_semantic_model():
    from sentence_transformers import SentenceTransformer
    import src.task5_semantic_search as t5
    return SentenceTransformer(t5.EMBEDDING_MODEL)

@st.cache_resource
def get_cached_corpus_and_bm25():
    import src.task6_lexical_search as t6
    from src.task4_chunking_indexing import load_documents, chunk_documents
    _docs = load_documents()
    corpus = chunk_documents(_docs)
    bm25_index = t6.build_bm25_index(corpus)
    return corpus, bm25_index

try:
    import src.task5_semantic_search as t5
    t5._SEMANTIC_MODEL = get_cached_semantic_model()
except Exception:
    pass

try:
    import src.task6_lexical_search as t6
    cached_corpus, cached_bm25 = get_cached_corpus_and_bm25()
    t6.CORPUS = cached_corpus
    t6._bm25_index = cached_bm25
except Exception:
    pass

import re

# Helper function to extract metadata details
def get_source_details(source_filename: str, doc_type: str, chunk_content: str):
    """
    Trích xuất chi tiết nguồn thông tin:
    - Tên văn bản/bài viết
    - Phần/Trang/Điều luật liên quan
    - URL liên kết
    - Toàn bộ nội dung văn bản gốc
    """
    display_name = source_filename.replace(".md", "")
    section = "N/A"
    link_url = None
    full_text = ""
    
    # Đọc file markdown gốc từ data/standardized/
    md_path = Path("data/standardized") / doc_type / source_filename
    if md_path.exists():
        try:
            full_text = md_path.read_text(encoding="utf-8")
        except Exception:
            pass
            
    # 1. Nếu là văn bản pháp luật
    if doc_type == "legal":
        # Tìm các ký tự chỉ "Điều XX" hoặc "Khoản XX" trong chunk để định vị phần/trang
        match_dieu = re.search(r"(Điều\s+\d+)", chunk_content)
        match_khoan = re.search(r"(Khoản\s+\d+)", chunk_content)
        
        sections = []
        if match_dieu:
            sections.append(match_dieu.group(1))
        if match_khoan:
            sections.append(match_khoan.group(1))
            
        section = " - ".join(sections) if sections else "Thông tin chung"
            
    # 2. Nếu là bài báo tin tức
    elif doc_type == "news":
        json_name = source_filename.replace(".md", ".json")
        json_path = Path("data/landing/news") / json_name
        if json_path.exists():
            try:
                import json
                data = json.loads(json_path.read_text(encoding="utf-8"))
                link_url = data.get("url")
                display_name = data.get("title", display_name)
            except Exception:
                pass
                
        section = "Tin tức báo chí"
        
    return {
        "display_name": display_name,
        "section": section,
        "link_url": link_url,
        "full_text": full_text
    }


def should_show_citations(answer: str) -> bool:
    """
    Xác định xem có nên hiển thị nguồn trích dẫn hay không.
    Ẩn nguồn nếu câu trả lời là chào hỏi xã giao, không đủ thông tin xác minh,
    hoặc không thực sự trích dẫn tài liệu cụ thể nào.
    """
    if not answer:
        return False
        
    answer_lower = answer.lower()
    
    # 1. Ẩn nếu là lời chào xã giao ngắn
    greetings = ["xin chào", "chào bạn", "chào anh", "chào chị", "hello", "hi", "hey", "chúc một ngày"]
    if any(g in answer_lower for g in greetings) and len(answer.split()) < 25:
        return False
        
    # 2. Ẩn nếu chứa các câu từ thể hiện không tìm thấy/không thể xác minh thông tin
    unverified_phrases = [
        "tôi không thể xác minh",
        "tôi không tìm thấy",
        "không thể xác minh",
        "không tìm thấy thông tin",
        "không có thông tin",
        "không đề cập",
        "không được đề cập",
        "tôi không biết",
        "chưa có thông tin"
    ]
    if any(phrase in answer_lower for phrase in unverified_phrases):
        return False
        
    # 3. Ẩn nếu không chứa bất kỳ thẻ trích dẫn nào trong câu trả lời (dạng [bo-luat...docx] hoặc [article_01.md])
    has_brackets = bool(re.search(r"\[[^\]]+\.(docx|md)\]", answer, re.IGNORECASE))
    if not has_brackets:
        return False
        
    return True


def render_sources(sources, key_prefix="hist", msg_idx=0):
    """
    Hiển thị các nguồn trích dẫn bằng Streamlit Expander.
    """
    if not sources:
        return
        
    with st.expander(f"🔍 Xem nguồn trích dẫn ({len(sources)} đoạn văn bản)"):
        for s_idx, source in enumerate(sources, 1):
            source_name = source.get("metadata", {}).get("source", "Không rõ nguồn")
            doc_type = source.get("metadata", {}).get("type", "unknown")
            score = source.get("score", 0.0)
            retrieval_method = source.get("source", "hybrid")
            
            # Lấy thông tin nguồn chi tiết
            details = get_source_details(source_name, doc_type, source["content"])
            
            st.markdown(f"""
            **[{s_idx}] {details['display_name']}** 
            <span class="badge badge-source">{retrieval_method.upper()}</span>
            <span class="badge badge-score">Score: {score:.3f}</span>
            <span class="badge badge-type">Type: {doc_type.upper()}</span>
            <br>
            📍 **Vị trí/Phần:** `{details['section']}`
            """, unsafe_allow_html=True)
            st.caption(source["content"])
            
            # Hiển thị nút liên kết / xem tài liệu gốc
            col1, col2 = st.columns([1, 1])
            with col1:
                key = f"btn_{key_prefix}_{msg_idx}_{s_idx}"
                if st.button("📖 Xem văn bản gốc", key=key, use_container_width=True):
                    st.session_state.selected_doc = {
                        "title": details["display_name"],
                        "section": details["section"],
                        "chunk_content": source["content"],
                        "full_text": details["full_text"]
                    }
                    st.rerun()
            with col2:
                if details["link_url"]:
                    st.link_button("🔗 Mở link bài viết", url=details["link_url"], use_container_width=True)
            st.markdown("<hr style='margin: 0.4rem 0; border: 0.5px solid #F1F5F9;'>", unsafe_allow_html=True)


# Cấu hình trang Streamlit
st.set_page_config(
    page_title="RAG Chatbot Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện premium
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Font chính */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Cấu hình background toàn trang */
    .stApp {
        background-color: #F8FAFC !important;
    }
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #0B0F19 !important;
        }
    }
    
    /* Cấu hình Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    @media (prefers-color-scheme: dark) {
        [data-testid="stSidebar"] {
            background-color: #0F172A !important;
            border-right: 1px solid #1E293B !important;
        }
    }
    
    /* Tiêu đề chính giống Google Gemini */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4285F4 0%, #9B72CB 30%, #D96570 70%, #F48120 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 0.2rem;
        letter-spacing: -0.04em;
        text-shadow: 0 4px 12px rgba(0, 0, 0, 0.01);
    }
    @media (prefers-color-scheme: dark) {
        .main-header {
            background: linear-gradient(90deg, #66A6FF 0%, #B19FFB 30%, #F5A3B3 70%, #F8B375 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }
    }
    
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    @media (prefers-color-scheme: dark) {
        .sub-title {
            color: #94A3B8;
        }
    }
    
    /* --- KEYFRAME ANIMATIONS (DƯỚI CÙNG TRANG VÀ HOVER EFFECT) --- */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(16px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        to {
            box-shadow: 0 0 0 10px rgba(16, 185, 129, 0);
        }
    }
    
    /* Hiệu ứng nhịp đập cho chấm xanh lá hoạt động */
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.6s infinite cubic-bezier(0.66, 0, 0, 1);
        vertical-align: middle;
        margin-right: 6px;
    }
    
    /* Bong bóng chat slide-up và fade-in */
    .stChatMessage {
        border-radius: 16px !important;
        padding: 1.25rem !important;
        margin-bottom: 1rem !important;
        border: 1px solid #E2E8F0 !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.01) !important;
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.25s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.25s ease;
        animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    
    .stChatMessage:hover {
        transform: translateY(-2px);
        border-color: #3B82F6 !important;
        box-shadow: 0 12px 20px -8px rgba(37, 99, 235, 0.08) !important;
    }
    
    @media (prefers-color-scheme: dark) {
        .stChatMessage {
            border: 1px solid #1E293B !important;
            background-color: #0F172A !important;
        }
        .stChatMessage:hover {
            border-color: #60A5FA !important;
            box-shadow: 0 12px 20px -8px rgba(0, 0, 0, 0.3) !important;
        }
    }
    
    /* Thiết kế ô nhập chat dạng viên thuốc (floating pill-shaped stChatInput) */
    div[data-testid="stChatInput"] {
        border-radius: 9999px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05) !important;
        background-color: #FFFFFF !important;
        padding: 6px 12px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: #3B82F6 !important;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.15) !important;
    }
    @media (prefers-color-scheme: dark) {
        div[data-testid="stChatInput"] {
            border-color: #1E293B !important;
            background-color: #1E293B !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3) !important;
        }
        div[data-testid="stChatInput"]:focus-within {
            border-color: #60A5FA !important;
            box-shadow: 0 10px 25px -5px rgba(96, 165, 250, 0.15) !important;
        }
    }
    
    /* Expander styling */
    .stExpander {
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.01) !important;
        background-color: #F8FAFC !important;
        margin-top: 0.8rem !important;
        animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    @media (prefers-color-scheme: dark) {
        .stExpander {
            border: 1px solid #1E293B !important;
            background-color: #1E293B60 !important;
        }
    }
    
    /* Cấu hình khung hiển thị tài liệu đọc (Tab bên phải) slide-in mượt */
    .doc-viewer-container {
        animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    
    /* Custom tabs for source content */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: transparent;
        border-bottom: 2px solid #E2E8F0;
    }
    @media (prefers-color-scheme: dark) {
        .stTabs [data-baseweb="tab-list"] {
            border-bottom: 2px solid #1E293B;
        }
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px !important;
        background-color: transparent !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: #64748B !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.2s ease !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #2563EB !important;
    }
    @media (prefers-color-scheme: dark) {
        .stTabs [data-baseweb="tab"]:hover {
            color: #60A5FA !important;
        }
    }
    
    .stTabs [aria-selected="true"] {
        color: #2563EB !important;
        border-bottom: 2px solid #2563EB !important;
    }
    @media (prefers-color-scheme: dark) {
        .stTabs [aria-selected="true"] {
            color: #60A5FA !important;
            border-bottom: 2px solid #60A5FA !important;
        }
    }
    
    /* Highlight văn bản trích dẫn */
    mark {
        background-color: rgba(245, 158, 11, 0.12) !important;
        color: #D97706 !important;
        border-bottom: 2px solid #F59E0B;
        padding: 2px 4px;
        font-weight: 600;
        border-radius: 4px;
    }
    @media (prefers-color-scheme: dark) {
        mark {
            background-color: rgba(245, 158, 11, 0.25) !important;
            color: #FBBF24 !important;
            border-bottom-color: #F59E0B;
        }
    }
    
    /* Button animations (Spring physics feedback) */
    .stButton > button {
        border-radius: 8px !important;
        transition: transform 0.1s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.2s ease !important;
        font-weight: 500 !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) scale(1.01);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
    }
    .stButton > button:active {
        transform: translateY(1px) scale(0.98);
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.05);
    }
    
    /* Badge tags */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 6px;
        margin-right: 0.4rem;
        letter-spacing: 0.02em;
    }
    .badge-source {
        background-color: #DBEAFE;
        color: #1E40AF;
    }
    .badge-score {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .badge-type {
        background-color: #E0F2FE;
        color: #0369A1;
    }
    .badge-fallback {
        background-color: #FEF3C7;
        color: #92400E;
    }
</style>
""", unsafe_allow_html=True)



# =============================================================================
# CONVERSATIONAL QUERY CONTEXTUALIZATION (Query Rewriting)
# =============================================================================

def is_query_out_of_scope(query: str) -> bool:
    """
    Kiểm tra xem câu hỏi có nằm ngoài phạm vi hỗ trợ hay không.
    Phạm vi hỗ trợ: Luật phòng chống ma túy, tin tức ma túy/nghệ sĩ liên quan ma túy, hoặc câu chào.
    """
    query_lower = query.lower().strip()
    
    # 1. Nếu là câu chào xã giao ngắn
    greetings = ["hello", "hi", "xin chào", "chào bạn", "chào", "chao ban", "chao", "hey", "chúc một ngày"]
    if query_lower in greetings or (any(g in query_lower for g in greetings) and len(query.split()) <= 4):
        return False
        
    # 2. Danh sách từ khóa chặn nhanh (Chặn các chủ đề ngoại lệ rõ ràng như lập trình, nấu ăn, thời tiết...)
    out_of_scope_keywords = [
        "viết code", "viết script", "lập trình", "code python", "code c#", "code java", "code javascript",
        "công thức nấu", "nấu ăn", "nấu món", "cách làm bánh", "cách nấu", "hướng dẫn nấu", "món ăn",
        "dự báo thời tiết", "thời tiết hôm nay", "giá vàng hôm nay", "chơi game", "tải game", "tải phần mềm"
    ]
    if any(keyword in query_lower for keyword in out_of_scope_keywords):
        return True

    # 3. Danh sách các từ khóa liên quan đến phạm vi hỗ trợ
    in_scope_keywords = [
        "ma túy", "ma tuy", "cai nghiện", "cai nghien", "tàng trữ", "tang tru", 
        "vận chuyển", "van chuyen", "mua bán", "mua ban", "sử dụng", "su dung", 
        "hình phạt", "hinh phat", "phạt tù", "phat tu", "bộ luật", "bo luat", 
        "luật", "luat", "điều", "dieu", "khoản", "khoan", "nghệ sĩ", "nghe si", 
        "ca sĩ", "ca si", "diễn viên", "dien vien", "người mẫu", "nguoi mau",
        "chi dân", "an tây", "andrea", "hữu tín", "lệ hằng", "châu việt cường", 
        "chất cấm", "chat cam", "heroin", "cocaine", "ketamine", "thuốc lắc", 
        "thuoc lac", "cần sa", "can sa", "bắt", "khởi tố", "tạm giữ", "án tù", 
        "bo-luat", "luat-phong-chong", "article_"
    ]
    
    if any(keyword in query_lower for keyword in in_scope_keywords):
        return False
        
    # 3. Sử dụng OpenAI GPT để phân loại chính xác nếu có API key
    api_key = os.getenv("OPENAI_API_KEY", "")
    use_openai = api_key and not api_key.startswith("sk-xxx") and len(api_key) > 15
    if use_openai:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompt = (
                "Phân loại câu hỏi sau có liên quan đến chủ đề luật phòng chống ma túy, "
                "chất ma túy, hành vi liên quan đến ma túy, tin tức ma túy, nghệ sĩ liên quan đến ma túy, "
                "hoặc câu chào hỏi hay không?\n"
                f"Câu hỏi: \"{query}\"\n"
                "Chỉ trả lời duy nhất chữ 'YES' (nếu liên quan) hoặc 'NO' (nếu không liên quan)."
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5
            )
            decision = response.choices[0].message.content.strip().upper()
            if "YES" in decision:
                return False
            elif "NO" in decision:
                return True
        except Exception:
            pass
            
    # Mặc định nếu không khớp từ khóa và không có API key thì coi như out of scope
    return True


def contextualize_query_local(query: str, history: list) -> str:
    """
    Sử dụng OpenAI hoặc quy tắc heuristic để viết lại câu hỏi follow-up
    thành câu hỏi độc lập đầy đủ ngữ cảnh dựa trên lịch sử hội thoại.
    """
    if not history:
        return query
        
    api_key = os.getenv("OPENAI_API_KEY", "")
    use_openai = api_key and not api_key.startswith("sk-xxx") and len(api_key) > 15
    
    # Chỉ viết lại nếu câu hỏi quá ngắn hoặc chứa từ chỉ định
    query_lower = query.lower()
    pronouns = ["nó", "họ", "anh ấy", "cô ấy", "ông này", "bà này", "đó", "này", "ở đâu", "bao nhiêu", "như thế nào", "tại sao"]
    is_follow_up = len(query.split()) < 5 or any(p in query_lower for p in pronouns)
    
    if not is_follow_up:
        return query
        
    if use_openai:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            # Xây dựng lịch sử tóm tắt
            history_text = ""
            for msg in history[-3:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"
                
            prompt = (
                "Given the following conversation history and a follow-up question, "
                "rewrite the follow-up question to be a standalone question in Vietnamese that "
                "contains all necessary context. Do NOT answer the question, just rewrite it.\n\n"
                f"History:\n{history_text}\n"
                f"Follow-up: {query}\n"
                "Standalone Question:"
            )
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100
            )
            rewritten = response.choices[0].message.content.strip()
            print(f"Contextualized query: '{query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            print(f"Failed to contextualize query via API: {e}")
            
    # Heuristic Fallback: Ghép với chủ đề câu hỏi trước đó nếu phát hiện follow-up
    last_user_msg = next((msg["content"] for msg in reversed(history) if msg["role"] == "user"), "")
    if last_user_msg:
        # Nếu câu hỏi trước hỏi về 1 nghệ sĩ cụ thể
        for name in ["Chi Dân", "An Tây", "Andrea", "Hữu Tín", "Lệ Hằng", "Châu Việt Cường"]:
            if name.lower() in last_user_msg.lower() and name.lower() not in query_lower:
                return f"{query} liên quan đến {name}"
        # Ghép mặc định với câu hỏi trước
        return f"{query} ({last_user_msg})"
        
    return query


# =============================================================================
# MULTI-TURN GENERATION WITH CUSTOM PIPELINE CONFIGS
# =============================================================================

def generate_answer_with_history(
    query: str,
    retrieved_chunks: list,
    history: list,
    temperature: float = 0.3,
    top_p: float = 0.9
) -> str:
    """
    Sinh câu trả lời có trích dẫn nguồn, hỗ trợ đa lượt hội thoại (multi-turn)
    bằng cách gửi kèm lịch sử chat trước đó tới OpenAI.
    """
    # 1. Định dạng ngữ cảnh trích dẫn
    context = format_context(retrieved_chunks)
    
    # 2. Kiểm tra API Key
    api_key = os.getenv("OPENAI_API_KEY", "")
    use_openai = api_key and not api_key.startswith("sk-xxx") and len(api_key) > 15
    
    if not use_openai:
        # Fallback local heuristic
        return local_heuristic_generation(query, retrieved_chunks)
        
    # 3. Gửi kèm lịch sử hội thoại cho OpenAI
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        system_prompt = (
            "Answer the following question comprehensively in Vietnamese using only the provided context.\n"
            "For every statement of fact or claim, immediately insert a citation in brackets "
            "linking to the specific source (e.g., [luat-phong-chong-ma-tuy-2021.docx] or [article_01.md]).\n"
            "If the information is not explicitly stated in the provided context, state "
            "'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than guessing.\n"
            "Maintain conversation flow based on the history provided."
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Thêm lịch sử (giới hạn 6 tin nhắn gần nhất)
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        # Thêm câu hỏi hiện tại kèm ngữ cảnh được chèn vào
        user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
        messages.append({"role": "user", "content": user_message})
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            timeout=15
        )
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        return local_heuristic_generation(query, retrieved_chunks)


# =============================================================================
# STREAMLIT UI LAYOUT
# =============================================================================

# Header ứng dụng giống Gemini
st.markdown(
    '<div style="display: flex; align-items: baseline; flex-wrap: wrap; gap: 12px; margin-bottom: 0.2rem;">'
    '  <div class="main-header" style="margin-bottom: 0;">⚖️ RAG Chatbot Assistant</div>'
    '  <div style="display: inline-flex; align-items: center; background-color: rgba(16, 185, 129, 0.08); padding: 4px 10px; border-radius: 9999px; border: 1px solid rgba(16, 185, 129, 0.2);">'
    '    <span class="pulse-dot"></span>'
    '    <span style="font-size: 0.75rem; font-weight: 700; color: #10B981; letter-spacing: 0.05em; text-transform: uppercase;">RAG Pipeline Online</span>'
    '  </div>'
    '</div>',
    unsafe_allow_html=True
)
st.markdown('<div class="sub-title">Trợ lý RAG thông minh lấy cảm hứng từ cấu trúc đa tác vụ Google Gemini</div>', unsafe_allow_html=True)

# SIDEBAR: Cấu hình RAG & Trạng thái hệ thống
st.sidebar.title("⚙️ Cấu Hình RAG Pipeline")

# Chọn chế độ so sánh / Config
config_mode = st.sidebar.selectbox(
    "Lựa chọn Cấu hình (A/B Test)",
    options=["Config A (Hybrid + Reranking)", "Config B (Retriever-only)", "Tùy chỉnh cá nhân"]
)

# Kích hoạt HyDE (Luôn khả dụng làm tùy chọn Bonus nâng cao)
use_hyde = st.sidebar.checkbox(
    "Kích hoạt HyDE cho Retrieval (+5đ Bonus)",
    value=False,
    help="Sinh câu trả lời giả định trước khi tìm kiếm để tăng tính chính xác."
)

# Áp dụng cấu hình tương ứng
if config_mode == "Config A (Hybrid + Reranking)":
    use_reranking = True
    rerank_method = "cross_encoder"
    score_threshold = 0.3
    top_k = 5
    st.sidebar.info("💡 Đang kích hoạt **Config A**: Sử dụng gộp RRF và Reranking với ngưỡng fallback là 0.3.")
elif config_mode == "Config B (Retriever-only)":
    use_reranking = False
    rerank_method = "rrf"
    score_threshold = 0.3
    top_k = 5
    st.sidebar.info("💡 Đang kích hoạt **Config B**: Bỏ qua bước Reranking, lấy trực tiếp kết quả RRF.")
else:
    # Cho phép tinh chỉnh thủ công các tham số
    use_reranking = st.sidebar.checkbox("Kích hoạt Reranking", value=True)
    rerank_method = st.sidebar.selectbox("Rerank Method", options=["cross_encoder", "mmr", "rrf"], index=0)
    score_threshold = st.sidebar.slider("Ngưỡng điểm Fallback (Threshold)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    top_k = st.sidebar.slider("Số lượng tài liệu (Top K)", min_value=1, max_value=10, value=5)

# Cấu hình Generation
st.sidebar.subheader("🧠 Cấu Hình LLM")
temperature = st.sidebar.slider("Temperature (Độ sáng tạo)", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.1)

# Trạng thái tính năng Bonus
st.sidebar.subheader("✨ Tính Năng Nâng Cao (Bonus)")
st.sidebar.markdown("""
<div style="font-size: 0.85rem; line-height: 1.6;">
🔹 <b>Conversation Memory:</b> <span style="color: #10B981; font-weight: bold;">Hoạt động 🧠</span><br>
🔹 <b>UI/UX Source Highlights:</b> <span style="color: #10B981; font-weight: bold;">Hoạt động 🖍️</span><br>
🔹 <b>PageIndex Fallback:</b> <span style="color: #10B981; font-weight: bold;">Sẵn sàng 📡</span> (Khi score < 0.3)
</div>
""", unsafe_allow_html=True)

# Thuyết minh Bonus Lexical vs BM25
with st.sidebar.expander("💡 Thuyết minh Lexical vs BM25 (+5đ)"):
    st.markdown("""
    **BM25 (Best Matching 25)** cải tiến vượt bậc so với **TF-IDF**:
    1. **Bão hòa Tần suất (TF Saturation):** TF-IDF tăng điểm số vô hạn theo số lần xuất hiện của từ. BM25 sử dụng tham số $k_1$ để giới hạn trần điểm TF, tránh việc lặp từ khóa "spam" gây nhiễu.
    2. **Chuẩn hóa Độ dài (Length Normalization):** BM25 sử dụng tham số $b$ để phạt độ dài văn bản dài một cách hợp lý dựa trên trung bình độ dài tập tài liệu ($avgdl$).
    """)

# Hướng dẫn Deploy (+4đ Bonus)
with st.sidebar.expander("🚀 Deploy Chatbot Online (+4đ)"):
    st.markdown("""
    **Quy trình 3 bước:**
    1. Tạo Space mới trên [Hugging Face Spaces](https://huggingface.co/spaces) (chọn SDK **Streamlit**).
    2. Upload toàn bộ mã nguồn của project lên Space.
    3. Cấu hình biến môi trường `OPENAI_API_KEY`, `JINA_API_KEY` trong mục Settings của Space.
    """)

# Check trạng thái API Keys
st.sidebar.subheader("📡 Trạng Thái Kết Nối API")
api_keys_status = {
    "OpenAI API": os.getenv("OPENAI_API_KEY", ""),
    "Jina AI Reranker": os.getenv("JINA_API_KEY", ""),
    "PageIndex SDK": os.getenv("PAGEINDEX_API_KEY", "")
}

for name, key in api_keys_status.items():
    if key and not key.startswith("sk-xxx") and not key.startswith("jina_xxx") and not key.startswith("pi_xxx") and len(key) > 10:
        st.sidebar.success(f"✔️ {name}: **Đang hoạt động**")
    else:
        st.sidebar.warning(f"⚠️ {name}: **Chưa cấu hình (Sử dụng Fallback Local)**")

# Nút Xóa lịch sử chat
if st.sidebar.button("🗑️ Xóa Lịch Sử Chat"):
    st.session_state.messages = []
    st.session_state.selected_doc = None
    st.session_state.last_sources = []
    st.rerun()

# Khởi tạo các session state cần thiết
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# Chia giao diện chính thành 2 cột: Cột Trò chuyện (trái) và Cột Tài liệu nguồn (phải)
chat_col, doc_col = st.columns([5, 4])

with chat_col:
    # Hiển thị lịch sử chat
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Hiển thị các nguồn trích dẫn
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                if should_show_citations(message["content"]):
                    render_sources(message["sources"], key_prefix="hist", msg_idx=idx)

    # Sinh câu trả lời của Trợ lý nếu tin nhắn cuối cùng trong lịch sử là của người dùng
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        latest_prompt = st.session_state.messages[-1]["content"]
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Kiểm tra câu hỏi ngoài phạm vi hỗ trợ (Ngoại lệ)
            if is_query_out_of_scope(latest_prompt):
                answer = (
                    "Xin lỗi, tôi là RAG Chatbot Assistant. Tôi chỉ có thể giải đáp các thắc mắc "
                    "liên quan đến Luật phòng chống ma túy Việt Nam và tin tức liên quan đến các nghệ sĩ vi phạm pháp luật "
                    "về ma túy. Câu hỏi của bạn nằm ngoài phạm vi hỗ trợ của tôi."
                )
                message_placeholder.markdown(answer)
                st.session_state.last_sources = []
                st.session_state.selected_doc = None
                
                # Lưu vào lịch sử hội thoại
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": []
                })
                st.rerun()
            
            # Rewrite câu hỏi dựa trên lịch sử nếu có
            contextualized = contextualize_query_local(latest_prompt, st.session_state.messages[:-1])
            
            with st.spinner("🔍 Đang truy xuất thông tin pháp lý & báo chí..."):
                # Chạy Retrieval Pipeline (Task 9)
                chunks = retrieve(
                    contextualized,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    use_reranking=use_reranking,
                    use_hyde=use_hyde
                )
                
                # Sắp xếp lại tránh lost in the middle (Task 10)
                reordered_chunks = reorder_for_llm(chunks)
                
            with st.spinner("✍️ Đang tổng hợp câu trả lời và ghi trích dẫn..."):
                # Gọi LLM sinh câu trả lời
                answer = generate_answer_with_history(
                    contextualized,
                    reordered_chunks,
                    st.session_state.messages[:-1],
                    temperature=temperature,
                    top_p=top_p
                )
                
            message_placeholder.markdown(answer)
            
            # Lưu các nguồn được truy xuất và tự động đặt tài liệu đầu tiên làm mặc định hiển thị
            if chunks and should_show_citations(answer):
                st.session_state.last_sources = chunks
                
                # Tự động chọn tài liệu đầu tiên
                first_source = chunks[0]
                s_name = first_source.get("metadata", {}).get("source", "Không rõ nguồn")
                d_type = first_source.get("metadata", {}).get("type", "unknown")
                details = get_source_details(s_name, d_type, first_source["content"])
                st.session_state.selected_doc = {
                    "title": details["display_name"],
                    "section": details["section"],
                    "chunk_content": first_source["content"],
                    "full_text": details["full_text"]
                }
                
                render_sources(chunks, key_prefix="live", msg_idx=999)
            else:
                st.session_state.last_sources = []
                st.session_state.selected_doc = None
                        
            # Lưu vào lịch sử hội thoại
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": chunks
            })
            
            # Reload lại trang để cột bên cạnh hiển thị tài liệu gốc ngay lập tức
            st.rerun()

# Ô nhập tin nhắn ghim ở cuối trang (ngoài chat_col để tránh giật/nhảy vị trí viết)
if prompt := st.chat_input("Hãy hỏi về luật ma túy hoặc tin tức nghệ sĩ..."):
    # 1. Lưu tin nhắn của người dùng vào session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 2. Reload lại trang để kích hoạt nhánh xử lý sinh câu trả lời của trợ lý
    st.rerun()

with doc_col:
    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 1.5rem; font-weight: 700; color: #1E3A8A;">📄 Chi tiết Tài liệu Nguồn</div>', unsafe_allow_html=True)
    
    if st.session_state.last_sources:
        # Nhúng class CSS để kích hoạt animation slide-up cho khu vực xem tài liệu
        st.markdown('<div class="doc-viewer-container">', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["📖 Nội dung tài liệu gốc", "🔍 Danh sách nguồn trích dẫn"])
        
        with tab1:
            if st.session_state.selected_doc:
                doc = st.session_state.selected_doc
                st.markdown(f"### {doc['title']}")
                if doc['section'] and doc['section'] != "N/A":
                    st.info(f"📍 Phần trích dẫn: **{doc['section']}**")
                    
                st.markdown("**Nội dung văn bản gốc (Phần trích dẫn được bôi màu vàng bên dưới):**")
                
                highlighted_text = doc['full_text']
                if doc['chunk_content'] and doc['chunk_content'] in doc['full_text']:
                    # Sử dụng thẻ mark thuần để CSS custom bôi màu mềm mại hơn
                    mark_tag = f'<mark>{doc["chunk_content"]}</mark>'
                    highlighted_text = doc['full_text'].replace(doc['chunk_content'], mark_tag)
                
                with st.container(height=550):
                    st.markdown(highlighted_text, unsafe_allow_html=True)
            else:
                st.info("💡 Hãy click nút **'📖 Xem văn bản gốc'** của các nguồn trích dẫn trong phần chat hoặc tab bên cạnh để đọc tài liệu gốc tại đây.")
                
        with tab2:
            st.markdown("Các đoạn trích dẫn được hệ thống RAG tìm thấy cho câu hỏi hiện tại:")
            for s_idx, source in enumerate(st.session_state.last_sources, 1):
                source_name = source.get("metadata", {}).get("source", "Không rõ nguồn")
                doc_type = source.get("metadata", {}).get("type", "unknown")
                score = source.get("score", 0.0)
                retrieval_method = source.get("source", "hybrid")
                
                # Lấy thông tin nguồn chi tiết
                details = get_source_details(source_name, doc_type, source["content"])
                
                st.markdown(f"""
                **[{s_idx}] {details['display_name']}** 
                <span class="badge badge-source">{retrieval_method.upper()}</span>
                <span class="badge badge-score">Score: {score:.3f}</span>
                <span class="badge badge-type">Type: {doc_type.upper()}</span>
                <br>
                📍 **Vị trí/Phần:** `{details['section']}`
                """, unsafe_allow_html=True)
                st.caption(source["content"])
                
                # Nút chuyển tài liệu gốc
                col1, col2 = st.columns([1, 1])
                with col1:
                    key = f"btn_tab2_{s_idx}"
                    if st.button("📖 Xem văn bản gốc", key=key, use_container_width=True):
                        st.session_state.selected_doc = {
                            "title": details["display_name"],
                            "section": details["section"],
                            "chunk_content": source["content"],
                            "full_text": details["full_text"]
                        }
                        st.rerun()
                with col2:
                    if details["link_url"]:
                        st.link_button("🔗 Mở link bài viết", url=details["link_url"], use_container_width=True)
                st.markdown("<hr style='margin: 0.4rem 0; border: 0.5px solid #F1F5F9;'>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("💡 Chưa có tài liệu nguồn nào được truy xuất cho câu trả lời này (hoặc câu trả lời là lời chào / thông tin không xác minh được).")
