import streamlit as st
import os
import warnings
import oss2 # 引入 OSS SDK

# 💡 关键修复：必须在引入任何 HuggingFace 相关库之前设置镜像环境变量
# 这样才能确保 sentence-transformers 和 transformers 使用镜像站
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ---------------------------------------------------------
# 1. 警告抑制与环境配置 (优化后台日志)
# ---------------------------------------------------------
# 设置环境变量，消除 HuggingFace Tokenizers 并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 忽略 BingImageCreator 的 pkg_resources 警告
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
# 忽略 LangChain 的 HuggingFaceEmbeddings 弃用警告
warnings.filterwarnings("ignore", message=".*HuggingFaceEmbeddings was deprecated.*")
# 忽略一般性的 LangChainDeprecationWarning
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass

import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # 引入本地 Embeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from openai import OpenAI # 引入 OpenAIError 基类
# from BingImageCreator import ImageGen # 原版引入
from bing_debug import ImageGen # 引入调试版 ImageGen
import db_manager # 引入数据库管理器

# 加载环境变量
load_dotenv()

# 设置页面配置 (必须是第一个 Streamlit 命令)
st.set_page_config(page_title="🎓 校园知识库助手 (RAG + 🎨)", layout="wide")

# 初始化数据库
db_manager.init_db()

# 初始化 Session State
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "user_config" not in st.session_state:
    st.session_state.user_config = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "current_docs" not in st.session_state: # 用于存储当前文档内容用于摘要
    st.session_state.current_docs = None

# --- 辅助函数：OSS 上传 ---
def upload_file_to_oss(file_obj, filename, config):
    """
    上传文件到阿里云 OSS
    :param file_obj: Streamlit UploadedFile 对象
    :param filename: 文件名
    :param config: 用户配置字典 (包含 OSS 凭证)
    :return: (bool, str) -> (是否成功, 消息/URL)
    """
    endpoint = config.get('oss_endpoint')
    access_key_id = config.get('oss_access_key_id')
    access_key_secret = config.get('oss_access_key_secret')
    bucket_name = config.get('oss_bucket_name')

    # 简单校验
    if not all([endpoint, access_key_id, access_key_secret, bucket_name]):
        return False, "OSS 配置不完整"

    try:
        import time
        # 1. 认证
        auth = oss2.Auth(access_key_id, access_key_secret)
        # 2. 获取 Bucket
        # 确保 endpoint 不带 http/https (或者带了也行，oss2比较智能)
        # 规范化：oss2.Bucket 需要 http://...
        if not endpoint.startswith('http'):
            endpoint = 'http://' + endpoint
            
        bucket = oss2.Bucket(auth, endpoint, bucket_name)
        
        # 3. 构造云端路径 (例如 uploads/20231027/timestamp_filename.pdf)
        date_str = time.strftime("%Y%m%d")
        # 使用 int(time.time()) 防止重名
        cloud_path = f"uploads/{date_str}/{int(time.time())}_{filename}"
        
        # 4. 上传
        # UploadedFile.getvalue() 返回 bytes
        bucket.put_object(cloud_path, file_obj.getvalue())
        
        # 5. 成功提示
        return True, f"已备份至 OSS: {cloud_path}"
        
    except Exception as e:
        return False, str(e)

# --- 辅助函数：处理 API 错误 ---
def handle_api_error(e):
    """统一处理 API 调用错误，给出友好提示"""
    error_str = str(e)
    if "429" in error_str or "insufficient_quota" in error_str:
        return (
            "⚠️ **API 余额不足或配额已耗尽**\n\n"
            "原因：你使用的 API Key 似乎没有余额了 (OpenAI 免费额度通常已过期)。\n\n"
            "👉 **解决方案**：\n"
            "1. **推荐 (学生党首选)**：注册 [DeepSeek](https://platform.deepseek.com/)，它非常便宜且不需要魔法。记得在左侧设置里将 Base URL 改为 `https://api.deepseek.com`。\n"
            "2. **检查设置**：如果你已经买了 DeepSeek，请确认左侧 Base URL 填写正确，而不是默认的 `openai.com`。\n"
            "3. **充值**：给你的 OpenAI 账户充值 (需要国外信用卡)。"
        )
    elif "401" in error_str or "invalid_api_key" in error_str:
        return "⚠️ **API Key 无效**\n请检查左侧设置中的 API Key 是否复制正确，注意不要多复制空格。"
    else:
        return f"❌ **发生错误**: {error_str}"

# --- 登录/注册页面 ---
def auth_page():
    st.title("🎓 校园知识库助手 - 登录")
    
    tab1, tab2 = st.tabs(["登录", "注册"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("用户名")
            password = st.text_input("密码", type="password")
            submit = st.form_submit_button("登录")
            
            if submit:
                user_id, msg = db_manager.login_user(username, password)
                if user_id:
                    st.success(msg)
                    st.session_state.user_id = user_id
                    st.session_state.username = username
                    # 加载用户配置
                    st.session_state.user_config = db_manager.get_user_config(user_id)
                    st.rerun()
                else:
                    st.error(msg)
    
    with tab2:
        with st.form("register_form"):
            new_user = st.text_input("用户名")
            new_pass = st.text_input("密码", type="password")
            new_pass_confirm = st.text_input("确认密码", type="password")
            submit_reg = st.form_submit_button("注册")
            
            if submit_reg:
                if new_pass != new_pass_confirm:
                    st.error("两次输入的密码不一致")
                elif not new_user or not new_pass:
                    st.error("用户名和密码不能为空")
                else:
                    success, msg = db_manager.register_user(new_user, new_pass)
                    if success:
                        st.success(f"{msg}，请切换到登录标签页登录。")
                    else:
                        st.error(msg)

# --- 主应用逻辑 ---
def main_app():
    # 标题
    st.title(f"🤖 校园知识库助手 (欢迎, {st.session_state.username})")
    st.markdown("上传文档，支持 **智能问答** 和 **创意配图生成**！")
    
    # --- 侧边栏：配置与上传 ---
    with st.sidebar:
        st.header("⚙️ 设置")
        
        # 获取默认值 (从 session_state.user_config 中取，如果没有则用默认值)
        cfg = st.session_state.user_config
        
        # API 配置
        api_key = st.text_input("API Key", value=cfg.get('api_key', ''), type="password", help="输入你的 OpenAI 或 DeepSeek API Key", key="input_api_key")
        
        # Base URL: 如果配置为空，则使用默认值
        default_base_url = 'https://api.openai.com/v1'
        saved_base_url = cfg.get('base_url', '')
        if not saved_base_url:
            saved_base_url = default_base_url
            
        base_url = st.text_input("Base URL (LLM)", value=saved_base_url, help="LLM 对话用的 Base URL", key="input_base_url")
        
        # 自动推断模型名称
        model_name = "gpt-3.5-turbo"
        if "deepseek" in base_url:
            model_name = "deepseek-chat"
            # Deep Thinking Toggle
            use_r1 = st.checkbox("🧠 开启深度思考 (DeepSeek R1)", value=False, help="使用 DeepSeek-R1 推理模型，擅长复杂逻辑和数学问题。")
            if use_r1:
                model_name = "deepseek-reasoner"
            
            st.caption(f"🤖 检测到 DeepSeek，已自动切换模型为: `{model_name}`")
        
        st.info("💡 提示：如果你使用 DeepSeek，建议在下方选择 '本地 Embeddings'，因为 DeepSeek 可能不支持 OpenAI 格式的 Embeddings 接口。")
        
        # Embeddings 选择
        embed_options = ["OpenAI / 兼容 API", "本地 HuggingFace (免费/慢)"]
        default_embed_idx = 0
        if cfg.get('embedding_type') in embed_options:
            default_embed_idx = embed_options.index(cfg.get('embedding_type'))
            
        embedding_type = st.selectbox("Embeddings 模型", embed_options, index=default_embed_idx, key="input_embedding_type")
        
        with st.expander("🎨 绘图设置 (可选)"):
            image_provider_opts = ["OpenAI DALL-E 3", "Bing Image Creator (免费)", "SiliconFlow (Flux)"]
            default_img_idx = 0
            if cfg.get('image_provider') in image_provider_opts:
                default_img_idx = image_provider_opts.index(cfg.get('image_provider'))
                
            image_provider = st.selectbox("绘图模型", image_provider_opts, index=default_img_idx, key="input_image_provider")
            
            # 初始化变量，防止未定义
            image_api_key = ""
            bing_cookie = ""
            bing_cookie_srch = ""
            full_cookie_str = ""
            proxy_url = ""
            user_agent = ""
            siliconflow_api_key = ""
            
            if image_provider == "OpenAI DALL-E 3":
                st.info("如果你使用 DeepSeek 等不包含 DALL-E 的模型，请在此输入 OpenAI Key 用于绘图，否则将尝试使用主 Key。")
                image_api_key = st.text_input("OpenAI Key (用于绘图)", value=cfg.get('image_api_key', ''), type="password", help="专门用于 DALL-E 绘图的 Key", key="input_image_api_key")
            
            elif image_provider == "SiliconFlow (Flux)":
                st.markdown("""
                **🚀 推荐方案 (稳定且高质量)**
                使用硅基流动 (SiliconFlow) 提供的 Flux.1 模型。
                1. 注册 [SiliconFlow](https://cloud.siliconflow.cn/i/Ia3z5C8s) (通常有免费额度)
                2. 创建 API Key 并填入下方
                """)
                siliconflow_api_key = st.text_input("SiliconFlow API Key", value=cfg.get('siliconflow_api_key', ''), type="password", help="sk-cn-...", key="input_siliconflow_api_key")

            elif image_provider == "Bing Image Creator (免费)":
                bing_cookie = st.text_input("Bing Cookie (_U)", value=cfg.get('bing_cookie', ''), type="password", help="Bing Image Creator 的 _U Cookie", key="input_bing_cookie")
                bing_cookie_srch = st.text_input("Bing Cookie (SRCHHPGUSR)", value=cfg.get('bing_cookie_srch', ''), type="password", help="Bing Image Creator 的 SRCHHPGUSR Cookie (必须填写以避免重定向错误)", key="input_bing_cookie_srch")
                
                # 新增：完整 Cookie 字符串支持
                st.caption("👇 如果上面两个 Cookie 仍然报错，请尝试粘贴完整的 Cookie 字符串")
                full_cookie_str = st.text_area("完整 Cookie 字符串 (可选)", value=cfg.get('full_cookie_str', ''), help="在浏览器 F12 -> Network -> 刷新页面 -> 点击任意 bing.com 请求 -> Request Headers -> 复制整个 Cookie 值", key="input_full_cookie_str", height=100)

                # 新增：User-Agent 设置
                st.caption("🕵️ 浏览器伪装 (User-Agent)")
                default_ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
                user_agent_input = st.text_input("User-Agent (可选)", value=cfg.get('user_agent', default_ua), help="建议使用与你获取 Cookie 的浏览器一致的 UA", key="input_user_agent")
                
                # 安全清理: 去除首尾空格 (包括 \t, \n 等)
                if user_agent_input:
                    user_agent = user_agent_input.strip()
                else:
                    user_agent = default_ua

                st.caption("🌐 网络设置 (如果遇到重定向错误请尝试)")
                proxy_url = st.text_input("HTTP 代理 (可选)", value=cfg.get('proxy_url', ''), placeholder="http://127.0.0.1:7890", help="如果你在中国大陆，通常需要配置代理才能连接 Bing。常见代理端口: 7890 (Clash), 10809 (v2ray) 等。", key="input_proxy_url")
                
                st.markdown("""
                **🔥 终极方案：使用插件 (最简单，不用找 F12)**

                既然开发者工具看不到，我们直接用插件，只需 3 步：

                1. **安装**：在 Chrome/Edge 商店搜索并安装 **"Cookie-Editor"** 扩展。
                2. **打开**：在 Bing 页面点击浏览器右上角的 Cookie-Editor 图标。
                3. **导出**：
                   - 点击右下角的 **"Export"** (导出) 按钮。
                   - 选择 **"Export as Header String"** (导出为字符串)。
                   - **直接粘贴** 到上面的 "完整 Cookie 字符串" 框里。
                
                *(如果插件导出的是 JSON 格式也没关系，直接粘进去，程序会自动尝试解析)*
                """)
        
        st.divider()

        # OSS 配置
        with st.expander("☁️ 云存储设置 (阿里云 OSS)"):
            st.caption("配置后，上传的文档将自动备份到阿里云 OSS，类似 PicGo。")
            oss_endpoint = st.text_input("Endpoint (地域节点)", value=cfg.get('oss_endpoint', ''), placeholder="oss-cn-hangzhou.aliyuncs.com", key="input_oss_endpoint")
            oss_access_key_id = st.text_input("AccessKey ID", value=cfg.get('oss_access_key_id', ''), type="password", key="input_oss_access_key_id")
            oss_access_key_secret = st.text_input("AccessKey Secret", value=cfg.get('oss_access_key_secret', ''), type="password", key="input_oss_access_key_secret")
            oss_bucket_name = st.text_input("Bucket Name (存储桶名称)", value=cfg.get('oss_bucket_name', ''), key="input_oss_bucket_name")

        # 保存配置按钮
        if st.button("💾 保存当前配置"):
            current_config = {
                'api_key': api_key,
                'base_url': base_url,
                'embedding_type': embedding_type,
                'image_provider': image_provider,
                'image_api_key': image_api_key,
                'bing_cookie': bing_cookie,
                'bing_cookie_srch': bing_cookie_srch,
                'full_cookie_str': full_cookie_str,
                'user_agent': user_agent,
                'proxy_url': proxy_url,
                'siliconflow_api_key': siliconflow_api_key,
                'oss_endpoint': oss_endpoint,
                'oss_access_key_id': oss_access_key_id,
                'oss_access_key_secret': oss_access_key_secret,
                'oss_bucket_name': oss_bucket_name
            }
            if db_manager.save_user_config(st.session_state.user_id, current_config):
                st.session_state.user_config = current_config
                st.success("配置已保存！下次登录会自动加载。")
            else:
                st.error("保存失败，请检查日志。")
                
        # 测试 Bing 连接按钮 (仅当选择了 Bing 时显示)
        if image_provider == "Bing Image Creator (免费)":
            if st.button("🧪 测试 Bing 连接 (检查 Cookie)"):
                # 构造临时 ImageGen 对象进行测试
                try:
                    # 智能解析逻辑 (复用)
                    final_u = bing_cookie
                    final_srch = bing_cookie_srch
                    all_cookies_list = []
                    
                    if full_cookie_str:
                        # Clean input
                        full_cookie_str = full_cookie_str.strip()
                        if full_cookie_str.lower().startswith("cookie:"):
                            full_cookie_str = full_cookie_str[7:].strip()
                            
                        # 尝试解析 JSON
                        if full_cookie_str.startswith('[') and full_cookie_str.endswith(']'):
                            import json
                            json_cookies = json.loads(full_cookie_str)
                            for item in json_cookies:
                                if 'name' in item and 'value' in item:
                                    all_cookies_list.append({'name': item['name'], 'value': item['value']})
                                    if item['name'] == "_U":
                                        final_u = item['value']
                                    elif item['name'] == "SRCHHPGUSR":
                                        final_srch = item['value']
                        else:
                            # key=value
                            for item in full_cookie_str.split(';'):
                                if '=' in item:
                                    k, v = item.strip().split('=', 1)
                                    all_cookies_list.append({'name': k, 'value': v})
                                    if k.strip() == "_U":
                                        final_u = v
                                    elif k.strip() == "SRCHHPGUSR":
                                        final_srch = v
                    
                    if not final_u:
                         st.error("❌ 无法找到 _U Cookie，请先填写配置！")
                    else:
                        if not final_srch: final_srch = final_u
                        
                        test_gen = ImageGen(
                            auth_cookie=final_u, 
                            auth_cookie_SRCHHPGUSR=final_srch, 
                            all_cookies=all_cookies_list,
                            quiet=False,
                            user_agent=user_agent
                        )
                        
                        # 设置代理
                        if proxy_url:
                            test_gen.session.proxies = {"http": proxy_url, "https": proxy_url}
                            
                        with st.spinner("正在验证 Bing 连接..."):
                            if test_gen.validate_session():
                                st.success("✅ Bing 连接成功！Cookie 有效，且未检测到登录跳转。")
                            else:
                                st.error("❌ Bing 连接验证失败：Cookie 可能失效，或 IP 被重定向到登录页。请检查日志。")
                                
                except Exception as e:
                    st.error(f"测试出错: {e}")

        if st.button("🚪 退出登录"):
            # 清除所有 Session State，确保登出彻底
            st.session_state.clear()
            st.rerun()

        st.divider()
        
        # 文件上传
        st.header("📂 文档上传")
        uploaded_file = st.file_uploader("上传 PDF 或 TXT 文件", type=["pdf", "txt"])
        
        if uploaded_file and st.button("开始处理文档"):
            if not api_key:
                st.error("请先输入 API Key！")
            else:
                with st.spinner("正在处理文档，请稍候..."):
                    try:
                        # 1. 保存临时文件
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name

                        # --- OSS 备份 ---
                        if cfg.get('oss_endpoint') and cfg.get('oss_bucket_name'):
                            with st.spinner("正在备份文件到阿里云 OSS..."):
                                success, msg = upload_file_to_oss(uploaded_file, uploaded_file.name, cfg)
                                if success:
                                    st.toast(msg, icon="☁️")
                                else:
                                    # 仅显示警告，不打断流程
                                    print(f"OSS Upload Warning: {msg}")
                                    if "OSS 配置不完整" not in msg:
                                        st.warning(f"OSS 备份失败: {msg}")
                        # ----------------

                        # 2. 加载文档
                        if uploaded_file.name.endswith(".pdf"):
                            loader = PyPDFLoader(tmp_path)
                        elif uploaded_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(tmp_path)
                        else:
                            loader = TextLoader(tmp_path)
                        docs = loader.load()
                        st.session_state.current_docs = docs # 保存文档引用
                        
                        # 3. 切分文档
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        splits = text_splitter.split_documents(docs)
                        
                        # 4. 向量化并存储
                        if embedding_type == "本地 HuggingFace (免费/慢)":
                            with st.spinner("正在加载本地 Embedding 模型 (首次运行需要下载)..."):
                                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                        else:
                            embeddings = OpenAIEmbeddings(
                                model="text-embedding-3-small", # 显式指定模型，防止兼容性问题
                                api_key=api_key,
                                base_url=base_url if "openai" not in base_url else None
                            )
                        
                        # 5. 创建向量数据库 (每个用户独立或共享？这里暂时是内存式 Session State，所以其实是隔离的)
                        # 如果需要持久化到磁盘且区分用户，persist_directory 应该加上 user_id
                        # 但这里为了简单，我们还是用 Session 里的 vector_store，刷新就没了
                        # 如果用 persist_directory="./chroma_db"，会混用。
                        # 改进：使用临时目录或者不持久化到磁盘(默认内存)，或者每个用户一个文件夹
                        
                        # 这里我们改用内存模式 (不传 persist_directory) 或者每个用户独立目录
                        user_db_dir = f"./chroma_db_{st.session_state.user_id}"
                        
                        vector_store = Chroma.from_documents(
                            documents=splits, 
                            embedding=embeddings,
                            persist_directory=user_db_dir 
                        )
                        
                        st.session_state.vector_store = vector_store
                        st.success(f"成功处理 {len(splits)} 个文本片段！现在可以提问或生成配图了。")
                        
                        # 清理临时文件
                        os.remove(tmp_path)
                        
                    except Exception as e:
                        st.error("❌ 文档处理发生错误")
                        with st.expander("查看详细错误信息 (请复制并发送给开发者)"):
                            st.code(str(e))
                            import traceback
                            st.code(traceback.format_exc())
                        
                        if "404" in str(e) or "not found" in str(e).lower():
                            st.warning("💡 **可能的原因**：你正在使用 DeepSeek 或其他模型，但它们不支持 OpenAI 格式的 Embedding 接口。\n👉 **建议**：请在左侧设置中将 'Embeddings 模型' 切换为 **'本地 HuggingFace'** 再试一次。")

    # === Tab 1: 智能问答 ===
    tab1, tab2 = st.tabs(["💬 智能问答", "🎨 创意配图"])

    with tab1:
        # 显示历史消息
        for message in st.session_state.messages:
            if message.get("type") != "image": # 只显示文本消息
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # 处理用户输入
        if prompt := st.chat_input("关于文档内容，你想知道什么？"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if st.session_state.vector_store is None:
                with st.chat_message("assistant"):
                    response = "请先在左侧上传文档并点击“开始处理文档”哦！👋"
                    st.markdown(response)
            else:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    try:
                        llm = ChatOpenAI(
                            model=model_name, 
                            temperature=0, 
                            api_key=api_key,
                            base_url=base_url
                        )
                        retriever = st.session_state.vector_store.as_retriever()
                        system_prompt = (
                            "你是一个乐于助人的校园助手。请根据下面的上下文（Context）回答用户的问题。"
                            "如果上下文中没有答案，请诚实地说你不知道。\n\nContext: {context}"
                        )
                        prompt_template = ChatPromptTemplate.from_messages([
                            ("system", system_prompt),
                            ("human", "{input}"),
                        ])
                        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                        
                        response = rag_chain.invoke({"input": prompt})
                        answer = response["answer"]
                        
                        message_placeholder.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                    except Exception as e:
                        error_msg = handle_api_error(e)
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # === Tab 2: 创意配图 ===
    with tab2:
        st.header("🎨 文档灵感配图")
        st.markdown("基于文档内容，自动生成一张创意封面或插图。")
        
        if st.session_state.vector_store is None:
            st.warning("请先上传并处理文档！")
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                style = st.selectbox("选择绘画风格", ["油画 (Oil Painting)", "水彩 (Watercolor)", "赛博朋克 (Cyberpunk)", "素描 (Sketch)", "写实 (Realistic)"])
                generate_btn = st.button("✨ 生成配图")
            
            if generate_btn:
                with col2:
                    with st.spinner("正在构思画面并绘图 (这可能需要十几秒)..."):
                        try:
                            # 1. 使用 LLM 生成绘画 Prompt
                            llm = ChatOpenAI(
                                model=model_name, 
                                temperature=0.7, 
                                api_key=api_key,
                                base_url=base_url
                            )
                            
                            # 简单获取文档摘要（取前2000字符，避免token溢出）
                            doc_snippet = ""
                            if st.session_state.current_docs:
                                doc_snippet = st.session_state.current_docs[0].page_content[:2000]
                            
                            prompt_gen_prompt = f"""
                            请阅读以下文档片段，提取核心主题和意境，将其转化为一段英文的 DALL-E 绘画提示词 (Prompt)。
                            
                            要求：
                            1. 风格：{style}
                            2. 只要提示词，不要其他废话。
                            3. 提示词要具体、画面感强。
                            
                            文档片段：
                            {doc_snippet}
                            """
                            
                            image_prompt_response = llm.invoke(prompt_gen_prompt)
                            image_prompt = image_prompt_response.content
                            st.info(f"🎨 **AI 设计的提示词**: {image_prompt}")
                            
                            # 2. 调用绘图接口
                            if image_provider == "OpenAI DALL-E 3":
                                # 优先使用专门的 Image Key，否则尝试使用主 Key
                                final_image_key = image_api_key if image_api_key else api_key
                                
                                if not final_image_key:
                                    st.error("缺少用于绘图的 API Key！")
                                else:
                                    client = OpenAI(api_key=final_image_key) # 使用官方 SDK
                                    
                                    response = client.images.generate(
                                        model="dall-e-3",
                                        prompt=image_prompt,
                                        size="1024x1024",
                                        quality="standard",
                                        n=1,
                                    )
                                    
                                    image_url = response.data[0].url
                                    st.image(image_url, caption=f"基于文档生成的 {style} 风格配图 (DALL-E 3)")

                            elif image_provider == "SiliconFlow (Flux)":
                                if not siliconflow_api_key:
                                    st.error("❌ 请先在左侧侧边栏填写 SiliconFlow API Key！")
                                else:
                                    try:
                                        client = OpenAI(
                                            api_key=siliconflow_api_key,
                                            base_url="https://api.siliconflow.cn/v1"
                                        )
                                        
                                        response = client.images.generate(
                                            model="black-forest-labs/FLUX.1-schnell",
                                            prompt=image_prompt,
                                            size="1024x1024",
                                            n=1,
                                        )
                                        
                                        image_url = response.data[0].url
                                        st.image(image_url, caption=f"基于文档生成的 {style} 风格配图 (Flux.1 Schnell)")
                                        
                                    except Exception as e:
                                        st.error(f"❌ SiliconFlow 绘图失败: {e}")
                                    
                            elif image_provider == "Bing Image Creator (免费)":
                                # 检查配置是否已填写 (直接使用 Sidebar 中定义的变量)
                                if not bing_cookie and not full_cookie_str:
                                    st.warning("⚠️ 请先在左侧侧边栏【设置 -> 绘图设置】中填写 Bing Cookie！\n\n👉 **推荐操作**：\n1. 打开侧边栏设置\n2. 找到“完整 Cookie 字符串”\n3. 粘贴刚才复制的一长串 Cookie")
                                else:
                                    # 智能解析逻辑
                                    final_u = bing_cookie
                                    final_srch = bing_cookie_srch
                                    all_cookies_list = []
                                    
                                    if full_cookie_str:
                                        try:
                                            # Clean input
                                            full_cookie_str = full_cookie_str.strip()
                                            # Remove "Cookie:" prefix if present (case insensitive)
                                            if full_cookie_str.lower().startswith("cookie:"):
                                                full_cookie_str = full_cookie_str[7:].strip()
                                                
                                            # 尝试解析 JSON 格式 (针对 Cookie-Editor 插件导出)
                                            if full_cookie_str.strip().startswith('[') and full_cookie_str.strip().endswith(']'):
                                                import json
                                                json_cookies = json.loads(full_cookie_str)
                                                for item in json_cookies:
                                                    if 'name' in item and 'value' in item:
                                                        all_cookies_list.append({'name': item['name'], 'value': item['value']})
                                                        if item['name'] == "_U":
                                                            final_u = item['value']
                                                        elif item['name'] == "SRCHHPGUSR":
                                                            final_srch = item['value']
                                            else:
                                                # 解析完整 Cookie 字符串 (key=value; key2=value2)
                                                for item in full_cookie_str.split(';'):
                                                    if '=' in item:
                                                        k, v = item.strip().split('=', 1)
                                                        all_cookies_list.append({'name': k, 'value': v})
                                                        # 自动提取关键 Cookie
                                                        if k.strip() == "_U":
                                                            final_u = v
                                                        elif k.strip() == "SRCHHPGUSR":
                                                            final_srch = v
                                        except Exception as parse_e:
                                            st.warning(f"Cookie 字符串解析部分失败: {parse_e}")
                                    
                                    if not final_u:
                                         st.error("❌ 无法从完整字符串中找到 _U Cookie，请检查复制是否完整！")
                                    else:
                                        # 如果没有填写 SRCHHPGUSR，尝试使用 _U (兼容旧逻辑)
                                        if not final_srch:
                                            final_srch = final_u

                                        with st.status("正在请求 Bing Image Creator...", expanded=True) as status:
                                            try:
                                                status.write("正在连接 Bing 服务器...")
                                                # 开启调试模式 quiet=False
                                                image_gen = ImageGen(
                                                    auth_cookie=final_u, 
                                                    auth_cookie_SRCHHPGUSR=final_srch, 
                                                    all_cookies=all_cookies_list,
                                                    quiet=False,
                                                    user_agent=user_agent
                                                )
                                                
                                                # 如果用户配置了代理，手动设置到 session 中
                                                if proxy_url:
                                                    image_gen.session.proxies = {
                                                        "http": proxy_url,
                                                        "https": proxy_url
                                                    }
                                                    print(f"DEBUG: Using Proxy: {proxy_url}")
                                                
                                                status.write("正在提交绘画任务...")
                                                print(f"DEBUG: Submitting prompt to Bing: {image_prompt}")
                                                image_urls = image_gen.get_images(image_prompt)
                                                print(f"DEBUG: Received {len(image_urls)} images")
                                                
                                                status.update(label="绘图成功！", state="complete")
                                                
                                                cols = st.columns(2)
                                                for i, url in enumerate(image_urls):
                                                    with cols[i % 2]:
                                                        st.image(url, caption=f"Bing 生成图 {i+1}")
                                                        
                                            except Exception as e:
                                                # 打印详细错误日志到控制台，方便调试
                                                print(f"ERROR generating image: {e}")
                                                import traceback
                                                traceback.print_exc()
                                                
                                                status.update(label="绘图失败", state="error")
                                                error_str = str(e)
                                                if "AuthCookieError" in error_str or "Unauthorized" in error_str:
                                                    st.error("❌ **Cookie 无效或过期**\n请重新获取 _U Cookie 并更新。")
                                                elif "Redirect" in error_str or "30 redirects" in error_str:
                                                    st.error(f"❌ **重定向错误 (Redirect Loop)**\n\n{error_str}\n\n原因：Bing 可能将您的请求重定向到了错误的区域 (如 cn.bing.com)。请尝试更换为美国/日本节点。")
                                                elif "Could not get results" in error_str:
                                                    st.error("❌ **生成超时或无结果**\n\nBing 正在处理任务但未返回结果。这通常是因为：\n1. **网络波动**：连接 Bing 服务器不稳定。\n2. **服务器繁忙**：Bing 免费服务当前负载过高。\n3. **Prompt 违规**：提示词可能触发了审核机制但没明确报错。\n\n👉 **建议**：稍等几秒再试一次，或尝试修改提示词。")
                                                else:
                                                    st.error(f"❌ **Bing 绘图出错**: {error_str}")
                                
                        except Exception as e:
                            st.error(handle_api_error(e))

# --- 程序入口 ---
if not st.session_state.user_id:
    auth_page()
else:
    main_app()
