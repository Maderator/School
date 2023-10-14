import scrapy


class ThesesspiderSpider(scrapy.Spider):
    name = 'ThesesSpider'
    allowed_domains = ['is.cuni.cz/webapps/zzp/search/?______searchform___search=&______facetform___facets___workType%5B%5D=DP&______facetform___facets___faculty%5B%5D=11320&______facetform___facets___defenseYear%5B%5D=2020&lang=en&tab_searchas=basic&PSzzpSearchListbasic=10&SOzzpSearchListbasic=&_sessionId=1835430&______searchform___butsearch=Search&PNzzpSearchListbasic=1']
    start_urls = ['http://is.cuni.cz/webapps/zzp/search/?______searchform___search=&______facetform___facets___workType%5B%5D=DP&______facetform___facets___faculty%5B%5D=11320&______facetform___facets___defenseYear%5B%5D=2020&lang=en&tab_searchas=basic&PSzzpSearchListbasic=10&SOzzpSearchListbasic=&_sessionId=1835430&______searchform___butsearch=Search&PNzzpSearchListbasic=1/']

    def parse(self, response):
        for r in response.css("div.zzp-work-maintitle"):
            item = {
                'title': r.css("a::text").extract_first(), 
                'link' : r.css("a::attr(href)").extract_first()
                }
            yield item
        # extract possible link to the next page
        # if the page exists, we should move to the next page.
    #response.css("div.paginator").css("a::attr(href)").extract_first()