import requests
from bs4 import BeautifulSoup
import csv

url = 'https://www.haberler.com'
response = requests.get(url)
html_content = response.content
soup = BeautifulSoup(html_content, 'html.parser')

haberler = soup.find_all('article', class_='p12-col')
for haber in haberler:
    haber_baslik = haber.text.strip()
    haber_link = haber.find('a')['href']
    if not haber_link.startswith('http'):
            haber_link = 'https://www.haberler.com/' + haber_link
    detay_response = requests.get(haber_link)
    detay_soup = BeautifulSoup(detay_response.content, 'html.parser')
    haber_icerik = detay_soup.find_all('p', class_=False)
    icerik = '\n'.join(context.text.strip() for context in haber_icerik)

    with open('haberler_icerik.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([haber_baslik, icerik])
        
print("Veri çekme işlemi tamamlandı!")
