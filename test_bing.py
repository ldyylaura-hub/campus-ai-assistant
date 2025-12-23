import sys
import os
sys.path.append(os.getcwd())
from BingImageCreator import ImageGen
import requests

print("Testing ImageGen initialization with all_cookies...")

# Mock data
auth_cookie = "fake_u"
auth_cookie_srch = "fake_srch"
full_cookie_str = "key1=value1; key2=value2; _U=fake_u; SRCHHPGUSR=fake_srch"

# Parse cookies
all_cookies_list = []
for item in full_cookie_str.split(';'):
    if '=' in item:
        k, v = item.strip().split('=', 1)
        all_cookies_list.append({'name': k, 'value': v})

print(f"Parsed cookies: {all_cookies_list}")

try:
    image_gen = ImageGen(
        auth_cookie=auth_cookie,
        auth_cookie_SRCHHPGUSR=auth_cookie_srch,
        all_cookies=all_cookies_list,
        quiet=True
    )
    print("ImageGen initialized successfully.")
    
    # Verify cookies in session
    print("Session cookies:")
    for cookie in image_gen.session.cookies:
        print(f"  {cookie.name}: {cookie.value}")
        
    # Check if key1 and key2 are present
    if image_gen.session.cookies.get("key1") == "value1":
        print("SUCCESS: key1 cookie set correctly.")
    else:
        print("FAILURE: key1 cookie NOT found.")

except Exception as e:
    print(f"ERROR during initialization: {e}")
    import traceback
    traceback.print_exc()
