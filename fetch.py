import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time
import json

INPUT_EXCEL = "archive/data/drug list.xlsx"
OUTPUT_EXCEL = "archive/data/drug_list_with_ingredients.xlsx"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0 Safari/537.36"
    )
}


def clean_parentheses(text: str) -> str:
    """Remove anything inside parentheses."""
    return re.sub(r"\s*\(.*?\)", "", text).strip()


def search_1mg(drug_name: str) -> str | None:
    """
    Search 1mg and return the first valid drug page URL.
    """
    search_url = f"https://www.1mg.com/search/all?name={drug_name}"
    resp = requests.get(search_url, headers=HEADERS, timeout=10)

    if resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # First product card link
    link = soup.select_one("a[href^='/drugs/']")
    if not link:
        return None

    return "https://www.1mg.com" + link["href"]


def extract_salt_info(drug_url: str) -> str | None:
    resp = requests.get(drug_url, headers=HEADERS, timeout=10)
    if resp.status_code != 200:
        return None

    html = resp.text

    # Find embedded JSON containing salt info
    match = re.search(
        r'"saltComposition"\s*:\s*\[(.*?)\]',
        html,
        re.DOTALL
    )

    if not match:
        return None

    block = match.group(1)

    # Extract salt names
    salts = re.findall(r'"name"\s*:\s*"([^"]+)"', block)

    if not salts:
        return None

    # Join multiple salts if present
    raw_text = " + ".join(salts)

    # Remove parentheses just in case
    cleaned = re.sub(r"\s*\(.*?\)", "", raw_text).strip()

    return cleaned

def main():
    df = pd.read_excel(INPUT_EXCEL)

    # Adjust column name if needed
    drug_column = df.columns[1]

    results = []

    for drug in df[drug_column]:
        print(f"🔍 Processing: {drug}")

        ingredient = None

        try:
            drug_url = search_1mg(str(drug))
            if drug_url:
                ingredient = extract_salt_info(drug_url)
        except Exception as e:
            print(f"⚠️ Error for {drug}: {e}")

        results.append(ingredient)

        time.sleep(2)  # polite delay

    df["Active Ingredient"] = results
    df.to_excel(OUTPUT_EXCEL, index=False)

    print(f"\n✅ Done. Saved to {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
