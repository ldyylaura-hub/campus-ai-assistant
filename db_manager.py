import sqlite3
import hashlib
import os

DB_FILE = "users.db"

def init_db():
    """初始化数据库表"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # 创建用户表
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL
        )
    ''')
    
    # 创建配置表
    # 使用 user_id 作为主键，确保每个用户只有一行配置
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_configs (
            user_id INTEGER PRIMARY KEY,
            api_key TEXT,
            base_url TEXT,
            embedding_type TEXT,
            image_provider TEXT,
            image_api_key TEXT,
            bing_cookie TEXT,
            bing_cookie_srch TEXT,
            full_cookie_str TEXT,
            proxy_url TEXT,
            siliconflow_api_key TEXT,
            oss_endpoint TEXT,
            oss_access_key_id TEXT,
            oss_access_key_secret TEXT,
            oss_bucket_name TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def check_migrations():
    """检查并应用数据库迁移 (添加缺失的列)"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # 1. 检查 user_configs 表是否存在 bing_cookie_srch
    try:
        c.execute("SELECT bing_cookie_srch FROM user_configs LIMIT 1")
    except sqlite3.OperationalError:
        # 列不存在，添加它
        try:
            print("Applying migration: Adding bing_cookie_srch column...")
            c.execute("ALTER TABLE user_configs ADD COLUMN bing_cookie_srch TEXT")
        except Exception as e:
            print(f"Migration error (bing_cookie_srch): {e}")

    # 2. 检查 user_configs 表是否存在 full_cookie_str
    try:
        c.execute("SELECT full_cookie_str FROM user_configs LIMIT 1")
    except sqlite3.OperationalError:
        # 列不存在，添加它
        try:
            print("Applying migration: Adding full_cookie_str column...")
            c.execute("ALTER TABLE user_configs ADD COLUMN full_cookie_str TEXT")
        except Exception as e:
            print(f"Migration error (full_cookie_str): {e}")

    # 3. 检查 user_configs 表是否存在 siliconflow_api_key
    try:
        c.execute("SELECT siliconflow_api_key FROM user_configs LIMIT 1")
    except sqlite3.OperationalError:
        # 列不存在，添加它
        try:
            print("Applying migration: Adding siliconflow_api_key column...")
            c.execute("ALTER TABLE user_configs ADD COLUMN siliconflow_api_key TEXT")
        except Exception as e:
            print(f"Migration error (siliconflow_api_key): {e}")

    # 4. 检查 user_configs 表是否存在 oss_endpoint
    try:
        c.execute("SELECT oss_endpoint FROM user_configs LIMIT 1")
    except sqlite3.OperationalError:
        # 列不存在，添加它们 (批量添加)
        try:
            print("Applying migration: Adding OSS columns...")
            c.execute("ALTER TABLE user_configs ADD COLUMN oss_endpoint TEXT")
            c.execute("ALTER TABLE user_configs ADD COLUMN oss_access_key_id TEXT")
            c.execute("ALTER TABLE user_configs ADD COLUMN oss_access_key_secret TEXT")
            c.execute("ALTER TABLE user_configs ADD COLUMN oss_bucket_name TEXT")
        except Exception as e:
            print(f"Migration error (OSS columns): {e}")

    conn.commit()
    conn.close()

def hash_password(password, salt=None):
    """简单的密码哈希"""
    if salt is None:
        salt = os.urandom(16).hex()
    
    # 使用 PBKDF2 增加安全性
    pwd_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        bytes.fromhex(salt),
        100000
    ).hex()
    
    return pwd_hash, salt

def register_user(username, password):
    """注册新用户"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    try:
        # 检查用户是否存在
        c.execute("SELECT id FROM users WHERE username = ?", (username,))
        if c.fetchone():
            return False, "用户名已存在"
        
        pwd_hash, salt = hash_password(password)
        
        c.execute("INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
                  (username, pwd_hash, salt))
        
        # 初始化空配置
        user_id = c.lastrowid
        c.execute("INSERT INTO user_configs (user_id) VALUES (?)", (user_id,))
        
        conn.commit()
        return True, "注册成功"
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def login_user(username, password):
    """用户登录"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    try:
        c.execute("SELECT id, password_hash, salt FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        
        if not user:
            return None, "用户不存在"
        
        user_id, stored_hash, salt = user
        input_hash, _ = hash_password(password, salt)
        
        if input_hash == stored_hash:
            return user_id, "登录成功"
        else:
            return None, "密码错误"
    finally:
        conn.close()

def save_user_config(user_id, config):
    """保存用户配置"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    try:
        # 使用 ON CONFLICT REPLACE 或者 UPDATE
        c.execute('''
            UPDATE user_configs 
            SET api_key=?, base_url=?, embedding_type=?, 
                image_provider=?, image_api_key=?, bing_cookie=?, 
                bing_cookie_srch=?, full_cookie_str=?, proxy_url=?
            WHERE user_id=?
        ''', (
            config.get('api_key', ''),
            config.get('base_url', ''),
            config.get('embedding_type', ''),
            config.get('image_provider', ''),
            config.get('image_api_key', ''),
            config.get('bing_cookie', ''),
            config.get('bing_cookie_srch', ''),
            config.get('full_cookie_str', ''),
            config.get('proxy_url', ''),
            user_id
        ))
        conn.commit()
        return True
    except Exception as e:
        print(f"Save config error: {e}")
        return False
    finally:
        conn.close()

def get_user_config(user_id):
    """获取用户配置"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    try:
        c.execute('''
            SELECT api_key, base_url, embedding_type, image_provider, 
                   image_api_key, bing_cookie, bing_cookie_srch, full_cookie_str, proxy_url 
            FROM user_configs WHERE user_id = ?
        ''', (user_id,))
        row = c.fetchone()
        
        if row:
            return {
                'api_key': row[0] or "",
                'base_url': row[1] or "",
                'embedding_type': row[2] or "",
                'image_provider': row[3] or "",
                'image_api_key': row[4] or "",
                'bing_cookie': row[5] or "",
                'bing_cookie_srch': row[6] or "",
                'full_cookie_str': row[7] or "",
                'proxy_url': row[8] or ""
            }
        return {}
    finally:
        conn.close()

# 初始化数据库 (如果不存在)
if not os.path.exists(DB_FILE):
    init_db()

# 每次运行时检查迁移 (确保现有数据库拥有新字段)
check_migrations()
