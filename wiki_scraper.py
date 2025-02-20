

import bs4
import sys
import requests

FULL_PATH = "https://kingdom-come-deliverance.fandom.com/wiki/A_Costly_Brawl"
WIK_PREFIX = "https://kingdom-come-deliverance.fandom.com/wiki/"
PAGE_TITLES = ["A_Costly_Brawl", "A_Friend_In_Need...", "A_Man_of_the_Cloth", "A_Place_to_Call_Home"]


def download_page(page_title):
    res = requests.get(WIK_PREFIX + page_title )
    res.raise_for_status()
    wiki = bs4.BeautifulSoup(res.text,"html.parser")

    # open a file named as your wiki page in write mode
    with open(page_title+".txt", "w", encoding="utf-8") as f:
        for i in wiki.select('p'):
            # write each paragraph to the file
            f.write(i.getText())

def main():
    for page in PAGE_TITLES:
        download_page(page)
    print("Done")

if __name__ == "__main__":
    main()
