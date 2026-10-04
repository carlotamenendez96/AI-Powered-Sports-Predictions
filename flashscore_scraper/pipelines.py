# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import json
import os

class FlashscoreScraperPipeline:
    def process_item(self, item, spider):
        return item

class StandingsPipeline:
    def open_spider(self, spider):
        if spider.name != "standings": return
        
        self.files = {}
        self.base_dir = os.path.join(spider.settings.get('PROJECT_ROOT', '.'), 'data_sets/standings')
        os.makedirs(self.base_dir, exist_ok=True)
        
        # We will accumulate data in memory and write at the end?
        # Or write line by line?
        # User wants "json files" which implies a JSON array.
        # Scrapy calls process_item one by one. I can append to a list in memory.
        self.data_store = {
            'standings_overall': [],
            'standings_home': [],
            'standings_away': [],
            'last_5_matches_overall': [],
            'last_5_matches_home': [],
            'last_5_matches_away': [],
            'last_10_matches_overall': [],
            'last_10_matches_home': [],
            'last_10_matches_away': []
        }

    def process_item(self, item, spider):
        if spider.name != "standings": return item
        
        t = item.get('type')
        if t:
            if t not in self.data_store:
                self.data_store[t] = []
                
            # Flatten structure: The item contains 'table' which is a list of rows.
            # We want the output JSON to look like:
            # [ { 'country': 'ENGLAND', 'league': 'Premier League', 'rank': '1', 'team': 'Arsenal'... }, ... ]
            # So we unwrap the table rows and add metadata.
            
            country = item['country']
            league = item['league']
            
            for row in item['table']:
                enriched_row = row.copy()
                enriched_row['country'] = country
                enriched_row['league'] = league
                self.data_store[t].append(enriched_row)
                
        return item

    def close_spider(self, spider):
        if spider.name != "standings": return
        
        for key, rows in self.data_store.items():
            filepath = os.path.join(self.base_dir, f"{key}.json")
            # Never clobber a previous good scrape with an empty [] — Flashscore
            # flakes often leave home/away empty while overall succeeds. Keep the
            # stale file so predictions degrade gracefully instead of reading zero.
            if not rows and os.path.isfile(filepath):
                spider.logger.warning(
                    f"Skipping write of empty {key}.json — keeping previous file on disk"
                )
                continue
            with open(filepath, 'w') as f:
                json.dump(rows, f, indent=2)
            spider.logger.info(f"Saved {len(rows)} rows to {filepath}")
