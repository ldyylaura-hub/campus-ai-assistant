import os
import random
import time
import regex
import requests
from typing import Dict, List, Union
from functools import partial
import contextlib

BING_URL = os.getenv("BING_URL", "https://www.bing.com")
HEADERS = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "max-age=0",
    "content-type": "application/x-www-form-urlencoded",
    "Referer": "https://www.bing.com/images/create/",
    "origin": "https://www.bing.com",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
}

# Error messages
error_timeout = "Your request has timed out."
error_redirect = "Redirect failed"
error_blocked_prompt = "Your prompt has been blocked by Bing. Try to change any bad words and try again."
error_being_reviewed_prompt = "Your prompt is being reviewed by Bing. Try to change any sensitive words and try again."
error_noresults = "Could not get results"
error_unsupported_lang = "\nthis language is currently not supported by bing"
error_bad_images = "Bad images"
error_no_images = "No images"

# Action messages
sending_message = "Sending request..."
wait_message = "Waiting for results..."
download_message = "\nDownloading images..."

def debug(debug_file, text_var):
    """helper function for debug"""
    with open(f"{debug_file}", "a", encoding="utf-8") as f:
        f.write(str(text_var))
        f.write("\n")

class ImageGen:
    """
    Image generation by Microsoft Bing (Debug Version)
    """

    def __init__(
        self,
        auth_cookie: str,
        auth_cookie_SRCHHPGUSR: str,
        debug_file: Union[str, None] = None,
        quiet: bool = False,
        all_cookies: List[Dict] = None,
        user_agent: str = None,
    ) -> None:
        self.session: requests.Session = requests.Session()
        self.session.headers = HEADERS.copy()
        if user_agent:
            # Clean up user_agent to prevent header injection errors
            self.session.headers["user-agent"] = user_agent.strip()
        
        self.session.cookies.set("_U", auth_cookie)
        self.session.cookies.set("SRCHHPGUSR", auth_cookie_SRCHHPGUSR)
        if all_cookies:
            for cookie in all_cookies:
                self.session.cookies.set(cookie["name"], cookie["value"])
        
        # Debug: Verify critical cookies
        u_cookie = self.session.cookies.get("_U")
        if u_cookie:
            print(f"DEBUG: Cookie _U set successfully (Length: {len(u_cookie)}, Starts with: {u_cookie[:10]}...)")
        else:
            print("DEBUG: WARNING - Cookie _U is MISSING or EMPTY!")
            
        self.quiet = quiet
        self.debug_file = debug_file
        if self.debug_file:
            self.debug = partial(debug, self.debug_file)

    def validate_session(self) -> bool:
        """
        Check if the current session is valid (logged in).
        Returns True if valid, False otherwise.
        """
        try:
            # Try accessing the create page. If logged out, it usually redirects to login.
            # We use allow_redirects=False to catch the 302
            url = f"{BING_URL}/images/create"
            response = self.session.get(url, allow_redirects=False, timeout=10)
            
            # If 200 OK, check content for login buttons
            if response.status_code == 200:
                if "Join & Create" in response.text or "Sign in" in response.text:
                    print("DEBUG: Validation failed - 'Join & Create' or 'Sign in' found in page.")
                    return False
                print("DEBUG: Validation successful - Page loaded without login prompt.")
                return True
            
            # If 302 Redirect, check where it goes
            elif response.status_code == 302:
                redirect_url = response.headers.get("Location", "")
                print(f"DEBUG: Validation redirected to: {redirect_url}")
                if "auth" in redirect_url or "login" in redirect_url:
                    return False
                return True # Redirect to another internal page might be okay?
            
            else:
                print(f"DEBUG: Validation returned status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"DEBUG: Validation error: {e}")
            return False

    def get_images(self, prompt: str) -> list:
        """
        Fetches image links from Bing
        """
        if not self.quiet:
            print(sending_message)
            # Verify Proxy / IP
            try:
                ip_url = "http://ip-api.com/json/"
                ip_resp = self.session.get(ip_url, timeout=5)
                if ip_resp.status_code == 200:
                    data = ip_resp.json()
                    print(f"DEBUG: Current IP Location: {data.get('country')} ({data.get('query')})")
                else:
                    print(f"DEBUG: Failed to check IP: {ip_resp.status_code}")
            except Exception as e:
                print(f"DEBUG: Proxy/IP check failed: {e}")
        
        url_encoded_prompt = requests.utils.quote(prompt)
        payload = f"q={url_encoded_prompt}&qs=ds"
        url = f"{BING_URL}/images/create?q={url_encoded_prompt}&rt=4&FORM=GENCRE"
        
        response = self.session.post(
            url,
            allow_redirects=False,
            data=payload,
            timeout=200,
        )
        
        # Check for various error conditions
        if "this prompt is being reviewed" in response.text.lower():
            raise Exception(error_being_reviewed_prompt)
        if "this prompt has been blocked" in response.text.lower():
            raise Exception(error_blocked_prompt)
        if "we're working hard to offer image creator in more languages" in response.text.lower():
            raise Exception(error_unsupported_lang)
            
        if response.status_code != 302:
            # if rt4 fails, try rt3
            url = f"{BING_URL}/images/create?q={url_encoded_prompt}&rt=3&FORM=GENCRE"
            response = self.session.post(url, allow_redirects=False, timeout=200)
            if response.status_code != 302:
                # Debug: Print detailed response info
                print(f"DEBUG: Redirect failed. Status: {response.status_code}")
                print(f"DEBUG: Final URL: {response.url}")
                print(f"DEBUG: Response Preview: {response.text[:1000]}") # Show more context
                
                if "Join & Create" in response.text or "Sign in" in response.text:
                    raise Exception(f"Cookie Invalid (Status 200): Bing thinks you are NOT logged in. \n1. Ensure you are logged into www.bing.com/images/create \n2. Copy the FULL cookie string (not just _U).")
                if "waitlist" in response.text.lower():
                    raise Exception("Account Restricted: You are on the Bing Image Creator waitlist.")
                
                # Check for other common errors
                if "blocked" in response.text.lower():
                    raise Exception(error_blocked_prompt)

                raise Exception(f"{error_redirect}. Status: {response.status_code}. \nPossible cause: Cookie expired or account restricted.")

        # Get redirect URL
        redirect_url = response.headers["Location"].replace("&nfy=1", "")
        print(f"DEBUG: Redirect Location: {redirect_url}") # Log the redirect location
        
        if "cn.bing.com" in redirect_url:
             raise Exception("Redirected to cn.bing.com. Please use a global proxy (US/Japan) to access Bing Image Creator.")

        if "id=" not in redirect_url:
             raise Exception(f"Failed to extract Request ID from redirect URL: {redirect_url}")

        request_id = redirect_url.split("id=")[-1]
        if redirect_url.startswith("http"):
            self.session.get(redirect_url, allow_redirects=False)
        else:
            self.session.get(f"{BING_URL}{redirect_url}", allow_redirects=False)
            
        polling_url = f"{BING_URL}/images/create/async/results/{request_id}?q={url_encoded_prompt}"
        
        if not self.quiet:
            print(f"{wait_message} (Polling URL: {polling_url})")
            
        start_wait = time.time()
        while True:
            if int(time.time() - start_wait) > 300:
                raise Exception(error_timeout)
                
            if not self.quiet:
                print(".", end="", flush=True)
                
            response = self.session.get(polling_url)
            
            # --- DEBUG LOGGING ADDED ---
            if not self.quiet:
                # Print detailed status for debugging
                # Only print every 5 seconds or if status changes to avoid spamming too much? 
                # Actually, user wants to see what's happening.
                pass
                
            if response.status_code != 200:
                print(f" [Status: {response.status_code}] ", end="", flush=True)
                time.sleep(1)
                continue
            
            if not response.text or response.text.find("errorMessage") != -1:
                print(f" [No Content or Error Message] ", end="", flush=True)
                time.sleep(1)
                continue
            else:
                break
                
        # Parse images
        image_links = regex.findall(r'src="([^"]+)"', response.text)
        normal_image_links = [link.split("?w=")[0] for link in image_links]
        normal_image_links = list(set(normal_image_links))

        bad_images = [
            "https://r.bing.com/rp/in-2zU3AJUdkgFe7ZKv19yPBHVs.png",
            "https://r.bing.com/rp/TX9QuO3WzcCJz1uaaSwQAz39Kb0.jpg",
        ]
        for img in normal_image_links:
            if img in bad_images:
                raise Exception("Bad images")
        
        if not normal_image_links:
            raise Exception(error_no_images)
            
        return normal_image_links
