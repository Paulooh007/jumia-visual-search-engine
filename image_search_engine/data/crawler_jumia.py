import scrapy
from pathlib import Path
from selenium.webdriver import Chrome, ChromeOptions 
from webdriver_manager.chrome import ChromeDriverManager

from scrapy.crawler import CrawlerProcess

from metadata import PRODUCT_BASE_URLS, PRODUCT_CATEGORIES, MAX_PAGE

class ProductsSpider(scrapy.Spider):
    name = 'products'
    product_names = PRODUCT_CATEGORIES
    urls_list = PRODUCT_BASE_URLS

    raw_data_dir = Path(__file__).resolve().parents[2] / "data" / "raw"

    custom_settings = {
    "FEEDS": {
        f"{raw_data_dir}/raw.csv": {"format": "csv"},
        f"{raw_data_dir}/raw.json": {"format": "json"},
    },
    "FEED_EXPORT_ENCODING": 'utf-8'
    }

    def start_requests(self):

        options = ChromeOptions()
        options.headless = True
        driver = Chrome(ChromeDriverManager().install(), options = options)

        MAX_PAGES = MAX_PAGE

        def get_url_for_page(BASE_URL, page_num):
            return f"{BASE_URL}&page={page_num}#catalog-listing"
        
        count = 0
        for product_name, base_url in zip(self.product_names, self.urls_list):
            for page in range(1, MAX_PAGES + 1):
                url = get_url_for_page(base_url, page)
                try:
                    driver.get(url)
                    xpath = "//article[@class='prd _fb col c-prd']//a"
                    product_links = driver.find_elements_by_xpath(xpath=xpath)

                    for link in product_links:
                        href = link.get_attribute("href")
                        yield scrapy.Request(href, callback=self.parse, meta = {"page_no": page, "product_category": product_name.lower(), "count": str(count)})
                        count += 1

                except:
                    print("url not found!!")

        driver.quit()


    def parse(self, response):
        product_category = response.meta["product_category"]
        name = response.xpath("//h1[@class='-fs20 -pts -pbxs']/text()").get()
        discounted_price = response.xpath("//span[@class='-b -ltr -tal -fs24']/text()").get()
        original_price = response.xpath("//span[@class='-tal -gy5 -lthr -fs16']/text()").get()
        product_images = response.xpath("//div[@id='imgs']/a/@href").getall()
        page_no = response.meta["page_no"]
        count = response.meta["count"]



        yield {
            "product_id": f"product_{product_category}_{count}",
            "product_category": product_category,
            "product_name": name,
            "product_url": response.url,
            "discounted_price": discounted_price,
            "original_price": original_price,
            "product_images_url": product_images,
            "page_no": page_no

        }

if __name__ == "__main__":
    process = CrawlerProcess()
    process.crawl(ProductsSpider)
    process.start()

