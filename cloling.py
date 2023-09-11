import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://search.naver.com/search.naver?where=nexearch&sm=tab_etc&mra=bkEw&pkid=68&os=28594807&qvt=0&query=%EB%B2%94%EC%A3%84%EB%8F%84%EC%8B%9C3%20%ED%8F%89%EC%A0%90"

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

score_elements = soup.select(".score_result .star_score")
review_elements = soup.select(".score_result .score_reple p")

data = []

for score_element, review_element in zip(score_elements, review_elements):
    score = score_element.text.strip()
    review = review_element.text.strip()
    data.append({"평점": score, "리뷰": review})

df = pd.DataFrame(data)
df.to_excel("movie_reviews.xlsx", index=False)