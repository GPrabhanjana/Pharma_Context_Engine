import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import re
import time

INPUT_EXCEL = "archive/data/drug list.xlsx"
OUTPUT_EXCEL = "archive/data/drug_list_with_ingredients.xlsx"

def setup_driver():
    """Initialize Chrome driver with options."""
    options = webdriver.ChromeOptions()
    # Uncomment below to run headless (no browser window)
    # options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36')
    
    driver = webdriver.Chrome(options=options)
    return driver

def search_drug_on_1mg(driver, drug_name: str) -> str | None:
    """
    Search for drug on 1mg and return URL of matching drug page.
    """
    try:
        print(f"  🔎 Searching on 1mg...")
        
        # Go to 1mg homepage
        driver.get("https://www.1mg.com")
        
        # Wait for search input to be present
        wait = WebDriverWait(driver, 10)
        search_input = wait.until(
            EC.presence_of_element_located((By.ID, "search-medicine"))
        )
        
        # Clear and enter drug name
        search_input.clear()
        search_input.send_keys(drug_name)
        
        # Hit Enter to search
        search_input.send_keys(Keys.RETURN)
        
        # Wait for results to load
        time.sleep(3)
        
        # Find all links with /drugs/ in href
        all_links = driver.find_elements(By.TAG_NAME, "a")
        
        drug_links = []
        for link in all_links:
            href = link.get_attribute("href")
            if href and "/drugs/" in href:
                # Extract the drug name from URL: /drugs/[name]-[id]
                match = re.search(r'/drugs/([\w-]+)-\d+', href)
                if match:
                    url_drug_name = match.group(1)
                    drug_links.append({
                        'url': href,
                        'name': url_drug_name
                    })
        
        if not drug_links:
            print(f"  ✗ No drug links found in search results")
            return None
        
        print(f"  📋 Found {len(drug_links)} drug links")
        
        # Check for containment match
        drug_name_lower = drug_name.lower().replace(" ", "").replace("-", "")
        
        for drug_link in drug_links:
            url_name_lower = drug_link['name'].lower().replace("-", "")
            
            # Check if excel name is contained in URL name or vice versa
            if drug_name_lower in url_name_lower or url_name_lower in drug_name_lower:
                print(f"  ✓ Match found: {drug_link['name']} matches {drug_name}")
                print(f"  🔗 URL: {drug_link['url']}")
                return drug_link['url']
        
        print(f"  ✗ No matching drug found (checked {len(drug_links)} links)")
        return None
        
    except TimeoutException:
        print(f"  ⚠️ Timeout waiting for page elements")
        return None
    except Exception as e:
        print(f"  ⚠️ Error during search: {e}")
        return None

def extract_salt_info(driver, drug_url: str) -> str | None:
    """
    Navigate to drug page and extract salt composition from saltInfo div.
    """
    try:
        driver.get(drug_url)
        
        # Wait for the saltInfo element to be present
        wait = WebDriverWait(driver, 10)
        salt_info_element = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "saltInfo"))
        )
        
        # Get the text content from the element
        salt_text = salt_info_element.text.strip()
        
        if salt_text:
            print(f"  ✓ Found ingredient: {salt_text}")
            return salt_text
        else:
            print(f"  ✗ saltInfo element found but empty")
            return None
        
    except TimeoutException:
        print(f"  ✗ Timeout waiting for saltInfo element")
        return None
    except NoSuchElementException:
        print(f"  ✗ saltInfo element not found on page")
        return None
    except Exception as e:
        print(f"  ⚠️ Error extracting salt info: {e}")
        return None

def main():
    # Read Excel file
    df = pd.read_excel(INPUT_EXCEL)
    drug_column = df.columns[1]
    
    # Setup Selenium driver
    print("🚀 Starting Chrome browser...")
    driver = setup_driver()
    
    results = []
    
    try:
        for idx, drug in enumerate(df[drug_column], 1):
            print(f"\n[{idx}/{len(df)}] 🔍 Processing: {drug}")
            ingredient = None
            
            try:
                drug_url = search_drug_on_1mg(driver, str(drug))
                if drug_url:
                    ingredient = extract_salt_info(driver, drug_url)
            except Exception as e:
                print(f"  ⚠️ Unexpected error for {drug}: {e}")
            
            results.append(ingredient)
            
            # Small delay between searches
            time.sleep(2)
        
    finally:
        # Always close the browser
        print("\n🔒 Closing browser...")
        driver.quit()
    
    # Save results
    df["Active Ingredient"] = results
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\n✅ Done. Saved to {OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()