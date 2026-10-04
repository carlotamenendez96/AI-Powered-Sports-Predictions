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

        # Merge per (country, league) instead of overwriting. close_spider also
        # runs on a PARTIAL crawl — CLOSESPIDER_TIMEOUT, or a SIGTERM, which
        # Scrapy handles as a graceful shutdown — and a blind overwrite then
        # replaced every league's table with the few it had reached. Measured
        # 2026-09-30: killing duplicate crawls left standings_overall holding 2
        # leagues instead of the full 40-league crawl from minutes earlier, and
        # because the files were now brand new, run_predictions.sh's 12h age
        # gate skipped the refresh and served neutral strength (MLS: "No
        # Standings Data"). Now a league scraped this run replaces its own rows;
        # every other league keeps its last good rows.
        for key, rows in self.data_store.items():
            if not rows:
                spider.logger.info(f"No rows for {key} this run — keeping existing file")
                continue
            filepath = os.path.join(self.base_dir, f"{key}.json")
            scraped = {(r.get('country'), r.get('league')) for r in rows}
            kept = []
            try:
                with open(filepath) as f:
                    existing = json.load(f)
                if isinstance(existing, list):
                    kept = [r for r in existing
                            if (r.get('country'), r.get('league')) not in scraped]
            except (OSError, ValueError):
                pass
            # Atomic write: a kill mid-dump must not leave a truncated file.
            tmp = filepath + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(kept + rows, f, indent=2)
            os.replace(tmp, filepath)
            spider.logger.info(f"Saved {len(rows)} rows ({len(scraped)} leagues) to {filepath}, "
                               f"kept {len(kept)} rows from other leagues")
