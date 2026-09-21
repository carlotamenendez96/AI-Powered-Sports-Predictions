import scrapy
import csv
import os
import json
from scrapy_playwright.page import PageMethod

class StandingsSpider(scrapy.Spider):
    name = "standings"
    
    
    def start_requests(self):
        csv_path = os.path.join(self.settings.get('PROJECT_ROOT', '.'), 'data_sets/standings_form_flashscore_direct_links.csv')
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"Missing {csv_path}. Seed it from "
                "data_sets/standings_form_flashscore_direct_links.template.csv "
                "(bin/update_leagues_data.sh does this automatically)."
            )

        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                meta = {
                    'country': row['COUNTRY'],
                    'league': row['LEAGUE'],
                    'playwright': True,
                    'playwright_include_page': True,
                    "playwright_page_goto_kwargs": {
                        "wait_until": "domcontentloaded",
                        "timeout": 60000
                    }
                }
                
                # Request 1: Standings (Overall -> Home -> Away)
                yield scrapy.Request(
                    url=row['STANDINGS_OVERALL'],
                    callback=self.parse_standings,
                    meta=meta.copy(),
                    dont_filter=True
                )
                
                # Request 2: Form (Overall -> Home -> Away)
                yield scrapy.Request(
                    url=row['FORM_LAST_5_OVERALL'],
                    callback=self.parse_form,
                    meta=meta.copy(),
                    dont_filter=True
                )

                # Request 3: Form Last 10 (Overall -> Home -> Away)
                # Check if column exists avoids key error for old CSVs
                if 'FORM_LAST_10_OVERALL' in row and row['FORM_LAST_10_OVERALL']:
                     meta_10 = meta.copy()
                     meta_10['form_type'] = 'last_10'
                     yield scrapy.Request(
                        url=row['FORM_LAST_10_OVERALL'],
                        callback=self.parse_form,
                        meta=meta_10,
                        dont_filter=True
                    )

    async def parse_standings(self, response):
        page = response.meta["playwright_page"]
        league = response.meta['league']
        country = response.meta['country']
        
        # 1. Overall
        await page.wait_for_selector(".ui-table__row")
        overall_data = await self.extract_table(page, "standings")
        yield {
            'type': 'standings_overall',
            'country': country,
            'league': league,
            'table': overall_data
        }
        
        # 2. Home
        try:
            # Click Home Tab
            # Selectors can be tricky, using text "Home" in the subTabs might work better
            # Or use the href pattern from CSV if we wanted, but clicking is usually safer in SPA
            # XPath: //a[contains(@href, 'standings/home')]
            await page.click("a[href*='standings/home']")
            await page.wait_for_timeout(1000) # Wait for table update
            await page.wait_for_selector(".ui-table__row")
            home_data = await self.extract_table(page, "standings")
            yield {
                'type': 'standings_home',
                'country': country,
                'league': league,
                'table': home_data
            }
        except Exception as e:
            self.logger.error(f"Error extracting Home standings for {league}: {e}")

        # 3. Away
        try:
            await page.click("a[href*='standings/away']")
            await page.wait_for_timeout(1000)
            await page.wait_for_selector(".ui-table__row")
            away_data = await self.extract_table(page, "standings")
            yield {
                'type': 'standings_away',
                'country': country,
                'league': league,
                'table': away_data
            }
        except Exception as e:
            self.logger.error(f"Error extracting Away standings for {league}: {e}")
            
        await page.close()

    async def parse_form(self, response):
        page = response.meta["playwright_page"]
        league = response.meta['league']
        country = response.meta['country']
        # Default to 'last_5' if not set
        form_type = response.meta.get('form_type', 'last_5') 
        
        # 1. Overall Form
        await page.wait_for_selector(".ui-table__row")
        overall_data = await self.extract_table(page, "form")
        yield {
            'type': f'{form_type}_matches_overall',
            'country': country,
            'league': league,
            'table': overall_data
        }
        
        # 2. Home Form
        try:
            await page.click("a[href*='form/home']")
            await page.wait_for_timeout(1000)
            await page.wait_for_selector(".ui-table__row")
            home_data = await self.extract_table(page, "form")
            yield {
                'type': f'{form_type}_matches_home',
                'country': country,
                'league': league,
                'table': home_data
            }
        except: pass

        # 3. Away Form
        try:
            await page.click("a[href*='form/away']")
            await page.wait_for_timeout(1000)
            await page.wait_for_selector(".ui-table__row")
            away_data = await self.extract_table(page, "form")
            yield {
                'type': f'{form_type}_matches_away',
                'country': country,
                'league': league,
                'table': away_data
            }
        except: pass
        
        await page.close()

    async def extract_table(self, page, table_type):
        # Identify rows
        rows = await page.query_selector_all(".ui-table__row")
        self.logger.info(f"Extracting {table_type}: Found {len(rows)} rows.")
        data = []
        for row in rows:
            text = await row.inner_text()
            lines = text.split('\n')
            # Format varies slightly but usually: Rank, Team, MP, W, D, L, Goals, Pts, Form?
            # Standings: 1. \n Team \n MP \n W \n D \n L \n GF:GA \n GD \n Pts \n ? \n Form...
            
            try:
                # Rank
                rank = lines[0].replace('.', '')
                
                # Team (Lines[1], sometimes Lines[2] if promotion marker?)
                # Usually lines[1] is Team.
                team = lines[1]
                
                # MP
                mp = lines[2]
                
                if table_type == "standings":
                    # W, D, L
                    w = lines[3]
                    d = lines[4]
                    l = lines[5]
                    # Goals (28:9)
                    goals = lines[6]
                    # GD (19) or Pts?
                    # In debug output: 28:9 \n 19 \n 33
                    # So lines[7] is GD, lines[8] is Pts.
                    # Verify length
                    
                    item = {
                        "rank": rank,
                        "team_name": team,
                        "matches_played": mp,
                        "wins": w,
                        "draws": d,
                        "losses": l,
                        "goals": goals,
                        "goals_difference": lines[7] if len(lines)>7 else 0,
                        "points": lines[8] if len(lines)>8 else 0
                    }
                    data.append(item)
                    
                else:
                    # Form Table
                    # Rank, Team, MP, W, D, L, Goals, Pts, FormString
                    # In form table, Flashscore shows "Form" column with icons.
                    # Text dump of Last 5 Form usually includes the W D L chars?
                    # Debug output showed empty form results?
                    # Let's extract Form String explicitly via selectors
                    
                    # W, D, L logic is same usually
                    w = lines[3]
                    d = lines[4]
                    l = lines[5]
                    goals = lines[6]
                    pts = lines[7] if ":" not in lines[7] else lines[8] # Careful with goals position
                    
                    # Explicitly get form icons
                    form_icons = await row.query_selector_all(".tableCellFormIcon")
                    form_str = ""
                    if form_icons:
                         texts = [await i.inner_text() for i in form_icons]
                         # Filter out empty, newlines, or '?'
                         texts = [t.strip() for t in texts if t.strip() and t.strip() != '?']
                         form_str = "|".join(texts)
                    
                    item = {
                        "rank": rank,
                        "team_name": team,
                        "matches_played": mp,
                        "last_5_results": form_str, # W|W|L...
                        "goals": goals,
                        "goals_difference": "N/A", # Form table often doesn't show GD explicity in text dump position? 
                        # Actually Flashscore form table: Rank, Team, MP, W, D, L, Goals, Pts, Form
                        # It doesn't show GD usually. It shows "Goals".
                        # User asked for "goals difference". I can calc it from Goals (e.g. 10:2 -> +8)
                        "points": pts
                    }
                    
                    # Calc GD
                    if ":" in goals:
                        gf, ga = goals.split(":")
                        item["goals_difference"] = int(gf) - int(ga)
                        
                    data.append(item)

            except Exception as e:
                # self.logger.warning(f"Error parsing row: {e}")
                continue
                
        return data
