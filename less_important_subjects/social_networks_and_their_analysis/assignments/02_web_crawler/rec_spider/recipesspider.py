import scrapy


class RecipesspiderSpider(scrapy.Spider):
    name = 'recipesspider'
    allowed_domains = ['www.afamilyfeast.com']
    start_urls = ['https://www.afamilyfeast.com/recipe-index/']

    RECNUMBER = 1000
    count = 0

    def parse(self, response):
        links = response.css("a[href*='tasty-recipes-']::attr(href)").extract()
        for l in links:
            yield response.follow(l, callback=self.parse_one_recipe)
            self.count += 1
            if self.count > self.RECNUMBER:
                return
        next = response.css("a.next.page-numbers::attr(href)").extract_first()
        if next:
            yield response.follow(next,callback=self.parse)

    def parse_one_recipe(self, response):
        par_num = len(response.css("div.tasty-recipes-ingredients-body > p").extract())
        if par_num:
            ings = []
            for i in range(1, par_num+1):
                par_strings = response.css("div.tasty-recipes-ingredients-body > p:nth-child("+str(i)+")::text").extract()
                ings.append(''.join(par_strings))
            yield {"name" : response.css("h2.tasty-recipes-title::text").extract_first(), 
                    "ingredients" : ings}
        else:
            li_num = len(response.css("div.tasty-recipes-ingredients-body > ul > li").extract())
            if li_num:
                ings = []
                for i in range(1, li_num+1):
                    li_strings = response.css("div.tasty-recipes-ingredients-body > ul > li:nth-child("+str(i)+")::text").extract()
                    #self.log(li_strings)
                    ings.append(''.join(li_strings))
            yield {"name" : response.css("h2.tasty-recipes-title::text").extract_first(), 
                    "ingredients" : ings}
