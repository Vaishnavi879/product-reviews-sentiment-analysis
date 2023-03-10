import dateparser as dateparser
import requests
import selectorlib as selectorlib
# from flask import jsonify
import json

class Fetch:
    def __init__(self):
        self.extractor = selectorlib.Extractor.from_yaml_file('selectors.yml')

    def scrape(self,url):
        headers = {
            'authority': 'www.amazon.com',
            'pragma': 'no-cache',
            'cache-control': 'no-cache',
            'dnt': '1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'sec-fetch-site': 'none',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-dest': 'document',
            'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
        }

        # Download the page using requests
        print("Downloading %s" % url)
        r = requests.get(url, headers=headers)
        # Simple check to check if page was blocked (Usually 503)
        if r.status_code > 500:
            if "To discuss automated access to Amazon data please contact" in r.text:
                print("Page %s was blocked by Amazon. Please try using better proxies\n" % url)
            else:
                print("Page %s must have been blocked by Amazon as the status code was %d" % (url, r.status_code))
            return None
        # Pass the HTML of the page and create
        data = self.extractor.extract(r.text, base_url=url)
        reviews = []
        for r in data['reviews']:
            r["product"] = data["product_title"]
            r['url'] = url
            if 'verified_purchase' in r:
                if 'Verified Purchase' in r['verified_purchase']:
                    r['verified_purchase'] = True
                else:
                    r['verified_purchase'] = False
            r['rating'] = r['rating'].split(' out of')[0]
            date_posted = r['date'].split('on ')[-1]
            if r['images']:
                r['images'] = "\n".join(r['images'])
            r['date'] = dateparser.parse(date_posted).strftime('%d %b %Y')
            reviews.append(r)
        histogram = {}
        for h in data['histogram']:
            histogram[h['key']] = h['value']
        data['histogram'] = histogram
        data['average_rating'] = float(data['average_rating'].split(' out')[0])
        data['reviews'] = reviews
        string=(data['number_of_reviews'].split('  customer')[0])
        data['number_of_reviews'] = int(''.join(e for e in string if e.isdigit()))
        return data


    def collect(self,url):
        # url='https://www.amazon.in/Redmi-Storage-Segment-5000mAh-Battery/product-reviews/B0BBN4DZBD/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'
        if url:
            data = self.scrape(url)
            reviewlist=[]
            for i in data["reviews"]:
                reviewlist.append(i["content"])
                # print(i["content"])
            return reviewlist
        else:
            return []

