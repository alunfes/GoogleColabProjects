from html.parser import HTMLParser
from urllib.parse import urljoin

class MyHTMLParser(HTMLParser):
    def __init__(self, base_url):
        super().__init__()
        self.base_url = base_url
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for attr, value in attrs:
                if attr == 'href':
                    absolute_url = urljoin(self.base_url, value)
                    self.links.append((value, absolute_url))

    def handle_data(self, data):
        if self.lasttag == 'a':
            self.links[-1] = (self.links[-1][0], self.links[-1][1], data)

html_content = '''<html lang="en">
    ...
</html>'''

base_url = 'https://www.hko.gov.hk/wxinfo/currwx/tc_e.htm'
parser = MyHTMLParser(base_url)
parser.feed(html_content)

for link, absolute_url, text in parser.links:
    if link.startswith('http'):
        print(f'Link: {link}')
        print(f'Absolute URL: {absolute_url}')
        print(f'Text: {text}')
        print()
    else:
        print(f'Relative Link: {link}')
        print(f'Absolute URL: {absolute_url}')
        print(f'Text: {text}')
        print()










from bs4 import BeautifulSoup
from urllib.parse import urljoin

html_content = '''<html lang="en">
    ...
</html>'''

base_url = 'https://www.hko.gov.hk/wxinfo/currwx/tc_e.htm'
soup = BeautifulSoup(html_content, 'html.parser')

for link in soup.find_all('a'):
    href = link.get('href')
    text = link.get_text()
    absolute_url = urljoin(base_url, href)

    if href.startswith('http'):
        print(f'Link: {href}')
        print(f'Absolute URL: {absolute_url}')
        print(f'Text: {text}')
        print()
    else:
        print(f'Relative Link: {href}')
        print(f'Absolute URL: {absolute_url}')
        print(f'Text: {text}')
        print()
