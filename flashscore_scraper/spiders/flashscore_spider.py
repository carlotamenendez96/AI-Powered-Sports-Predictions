import scrapy
from scrapy_playwright.page import PageMethod
from flashscore_scraper.items import MatchItem
from flashscore_scraper.items import MatchItem
import json
import datetime

class FlashscoreSpider(scrapy.Spider):
    name = "flashscore"
    allowed_domains = ["flashscore.com"]
    start_urls = ["https://www.flashscore.com/"]

    def __init__(self, day_diff=None, days_back=None, filter_leagues='false', mode='prediction', live_match_id=None, live_list='false', live_ids=None, *args, **kwargs):
        super(FlashscoreSpider, self).__init__(*args, **kwargs)
        
        # Resolve Date Offset (day_diff)
        # day_diff: 0 (Today), >0 (Future), <0 (Past)
        if day_diff is not None:
             self.day_diff = int(day_diff)
        elif days_back is not None:
             # Legacy support: days_back=1 means 1 day ago (-1)
             self.day_diff = -1 * int(days_back)
        else:
             # Default behavior if nothing specified: Tomorrow (+1)
             self.day_diff = 1
             
        self.filter_leagues_enabled = filter_leagues.lower() == 'true'
        self.mode = mode.lower()
        self.live_match_id = live_match_id
        self.live_list = live_list.lower() == 'true'
        self.live_ids = live_ids # Comma separated string
        self.target_leagues = set()
        
        self.logger.info(f"Spider initialized in {self.mode} mode.")
        if self.live_ids:
             self.logger.info(f"Batch Live IDs provided: {self.live_ids}")
             
        if self.mode == 'verification':
             self.filter_leagues_enabled = False
             self.logger.info("Verification Mode: League filtering DISABLED to capture all results.")

        if self.filter_leagues_enabled:
             try:
                 with open('data_sets/target_leagues.json', 'r') as f:
                     self.target_leagues = set(json.load(f))
             except: pass

    def start_requests(self):
        if self.live_ids:
            self.logger.info(f"Starting BATCH Live Stats for IDs: {self.live_ids}")
            # Use one request to process all sequentially in one browser context
            # Use one request to process all sequentially in one browser context
            yield scrapy.Request(
                url="https://www.flashscore.com/", # Must be allowed domain
                meta={
                    "playwright": True,
                    "playwright_include_page": True,
                     # Reuse context or default off. 'live_batch' context.
                    "playwright_context": f"live_batch_{str(datetime.datetime.now())}", # Force fresh context? No, just fresh page.
                    "live_ids": self.live_ids,
                    "dont_cache": True,
                },
                dont_filter=True,
                callback=self.parse_live_batch
            )
        elif self.live_match_id:
            # Single Match Mode (Legacy)
            url = f"https://www.flashscore.com/match/{self.live_match_id}/#/match-summary/match-statistics/0"
            yield scrapy.Request(
                url=url,
                meta={
                    "playwright": True,
                    "playwright_include_page": True,
                    "playwright_page_goto_kwargs": {
                        "wait_until": "domcontentloaded",
                        "timeout": 30000
                    },
                    "match_id": self.live_match_id,
                    "dont_cache": True,
                },
                dont_filter=True,
                callback=self.parse_live_stats # Legacy method, but we can verify it exists
            )
        elif self.live_list:
             yield scrapy.Request(
                url="https://www.flashscore.com/",
                meta={
                    "playwright": True,
                    "playwright_include_page": True
                },
                callback=self.parse_live_list_page
            )
        else:
            # Default Batch Mode
            yield scrapy.Request(
                url="https://www.flashscore.com/",
                meta={
                    "playwright": True,
                    "playwright_include_page": True,
                    "playwright_context": "match_list",
                },
                callback=self.parse_match_list
            )

    async def parse_live_list_page(self, response):
        page = response.meta["playwright_page"]
        
        try:
            # Click LIVE tab - robust try
            try:
                # Wait for at least one match to appear first to ensure page loaded
                await page.wait_for_selector('[id^="g_1_"]', timeout=15000)
                
                # Then try to click LIVE
                await page.click('.filters__tab:has-text("LIVE")')
                await page.wait_for_timeout(2000) 
            except:
                pass 
            
            # Use Locators (slower but verified to work in debug script)
            matches = []
            rows = page.locator('[id^="g_1_"]')
            count = await rows.count()
            self.logger.info(f"Spider found {count} match elements via Locators.")
            
            for i in range(count):
                row = rows.nth(i)
                id_attr = await row.get_attribute('id')
                if not id_attr: continue
                match_id = id_attr.replace('g_1_', '')
                
                # Try new selector first
                h_loc = row.locator('.event__homeParticipant')
                if await h_loc.count() == 0:
                   h_loc = row.locator('.event__participant--home') # Fallback
                   
                a_loc = row.locator('.event__awayParticipant')
                if await a_loc.count() == 0:
                   a_loc = row.locator('.event__participant--away') # Fallback

                h_text = await h_loc.inner_text() if await h_loc.count() > 0 else None
                a_text = await a_loc.inner_text() if await a_loc.count() > 0 else None
                
                if h_text: h_text = h_text.split('\n')[0]
                if a_text: a_text = a_text.split('\n')[0]
                
                if match_id and h_text and a_text:
                    matches.append({'match_id': match_id, 'home_team': h_text, 'away_team': a_text})
            
            for m in matches:
                # Log first few to verify
                yield m
        except Exception as e:
            self.logger.error(f"Live List Error: {e}")
        finally:
            await page.close()

    async def parse_match_list(self, response):
        page = response.meta["playwright_page"]
        
        # 1. Handle Consent / Cookie Banner
        try:
            if await page.query_selector("button#onetrust-accept-btn-handler"):
                await page.click("button#onetrust-accept-btn-handler")
                self.logger.info("Clicked cookie banner.")
                await page.wait_for_timeout(1000)
        except Exception as e:
            self.logger.warning(f"Cookie banner handling: {e}")


        # 2. Navigate to Target Date
        try:
            # Wait for navigation to be stable
            await page.wait_for_load_state("domcontentloaded")
            
            # self.day_diff: 0 (Today), >0 (Future), <0 (Past)
            self.logger.info(f"Navigating to target date (Offset: {self.day_diff} days)...")
            
            if self.day_diff == 0:
                self.logger.info("Target is Today. No navigation needed.")
                
            elif self.day_diff < 0:
                # PAST (Click Previous Day)
                days_to_go_back = abs(self.day_diff)
                for i in range(days_to_go_back):
                    clicked_prev = False
                    if await page.locator("button[aria-label='Previous day']").count() > 0:
                         await page.locator("button[aria-label='Previous day']").click()
                         self.logger.info(f"Clicked previous day ({i+1}/{days_to_go_back})")
                         clicked_prev = True
                    elif await page.locator(".calendar__navigation--yesterday").count() > 0:
                         await page.locator(".calendar__navigation--yesterday").click()
                         self.logger.info(f"Clicked previous day legacy ({i+1}/{days_to_go_back})")
                         clicked_prev = True
                    
                    if clicked_prev:
                        await page.wait_for_timeout(1000)
                    else:
                        self.logger.warning("Previous day button not found!")
                        break
                        
            elif self.day_diff > 0:
                # FUTURE (Click Next Day)
                days_to_go_fwd = self.day_diff
                for i in range(days_to_go_fwd):
                    clicked_next = False
                    if await page.locator("button[aria-label='Next day']").count() > 0:
                         await page.locator("button[aria-label='Next day']").click()
                         self.logger.info(f"Clicked next day ({i+1}/{days_to_go_fwd})")
                         clicked_next = True
                    elif await page.locator(".calendar__navigation--tomorrow").count() > 0:
                         await page.locator(".calendar__navigation--tomorrow").click()
                         self.logger.info(f"Clicked next day legacy ({i+1}/{days_to_go_fwd})")
                         clicked_next = True
                    
                    if clicked_next:
                        await page.wait_for_timeout(1000)
                    else:
                        self.logger.warning("Next day button not found!")
                        break
            
            # Wait after final navigation
            await page.wait_for_timeout(2000)
            
            # 2a. Click ODDS tab for better listing (Verification Mode)
            # DISABLED: Verification needs SCORES, which are best visible on the main ALL tab.
            # The ODDS tab often changes the layout or hides scores in favor of odds values.
            # if self.mode == 'verification':
            #      try:
            #          await page.click('.filters__tab:has-text("ODDS")')
            #          await page.wait_for_timeout(3000)
            #          self.logger.info("Clicked ODDS tab for verification.")
            #      except:
            #          self.logger.warning("Could not click ODDS tab.")

            # 2b. Scroll and Load All Matches (Lazy Loading) - PageDown Strategy
            # 2b. Scroll and Load All Matches (Lazy Loading) - JS Scroll Strategy
            self.logger.info("Scrolling (JS) to load all matches...")
            
            for i in range(20): # 20 scrolls should be enough with "Show More" handling
                # Scroll to bottom
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(1000)
                
                # Try clicking "Show more" (for past days matches often hidden behind this)
                if await page.locator(".event__more").count() > 0:
                     try:
                        await page.locator(".event__more").click()
                        self.logger.info("Clicked 'Show more' button.")
                        await page.wait_for_timeout(2000)
                     except: pass
                     
                # Check for "pinned" headers or other elements that might block view? No need.
                
                # Check match count
                if i % 5 == 0:
                    count = await page.evaluate("document.querySelectorAll('[id^=\"g_1_\"]').length")
                    self.logger.info(f"Scroll {i}: Found {count} match elements (g_1_*) so far.")
            
            self.logger.info("Finished scrolling.")

        except Exception as e:
            self.logger.error(f"Error navigating to next day: {e}")
            await page.close()
            return

        # 3. Extract Match IDs (and Leagues)
        content = await page.content()
        sel = scrapy.Selector(text=content)
        
        matches_to_scrape = []
        
        # Determine container - usually .sportName
        sport_containers = sel.css('.sportName')
        
        self.logger.info(f"Found {len(sport_containers)} sport containers.")
        
        for container in sport_containers:
            current_league = "Unknown League"
            
            # Iterate over all direct children to verify sequence (Header -> Matches)
            skip_league = False
            
            for child in container.xpath('./*'):
                classes = child.attrib.get('class', '')
                c_id = child.attrib.get('id', '')
                
                # Check for Header
                if 'headerLeague__wrapper' in classes or 'event__header' in classes:
                    # ... (header parsing logic same as before) ...
                    country = child.css('.headerLeague__category-text::text').get() or child.css('.event__title--type::text').get() or ""
                    league = child.css('.headerLeague__title-text::text').get() or child.css('.event__title--name::text').get() or ""
                    country = country.strip()
                    league = league.strip()
                    
                    if not country and not league:
                         all_text = " ".join(child.css("::text").getall())
                         current_league = " ".join(all_text.split())
                    else:
                         current_league = f"{country}: {league}".strip(": ")
                    
                    # self.logger.info(f"Header Found: {current_league}") # Uncomment for deeper debug if needed

                    # 1. Women's League Filter
                    league_lower = current_league.lower()
                    if "women" in league_lower or " w " in " " + league_lower + " " or "(w)" in league_lower:
                        skip_league = True
                        continue

                    # 2. Target League Filter
                    if self.filter_leagues_enabled:
                        if current_league not in self.target_leagues:
                            skip_league = True
                        else:
                            skip_league = False
                    
                elif c_id.startswith('g_1_'):
                    match_id = c_id.replace("g_1_", "")
                    # Log existence
                    # self.logger.info(f"Scanning match {match_id} in {current_league} (Skip: {skip_league})")
                    
                    if skip_league:
                        continue
                    
                    # Debug Filter: If debug_match is set, skip others
                    debug_id = getattr(self, 'debug_match', None)
                    if debug_id and match_id != debug_id:
                        continue

                    # Updated Selectors for Team Names (2025)
                    home_team = child.css('.event__participant--home::text').get()
                    if not home_team:
                        home_team = child.css('.event__homeParticipant').xpath('normalize-space()').get()
                    home_team = home_team or "Unknown Home"

                    away_team = child.css('.event__participant--away::text').get()
                    if not away_team:
                         away_team = child.css('.event__awayParticipant').xpath('normalize-space()').get()
                    away_team = away_team or "Unknown Away"
                    
                    # Capture scores if available (for verification mode)
                    home_score = child.css('.event__score--home::text').get()
                    away_score = child.css('.event__score--away::text').get()

                    # Match stage/status text (e.g. "Finished", "After Pen.",
                    # "AET", "Postp.", or a live minute). Scheduled matches show
                    # only a kickoff time and have no populated stage block.
                    stage_text = ' '.join(child.css('.event__stage ::text').getall()).strip()

                    # In PREDICTION mode we only want not-yet-started fixtures.
                    # A match that has already finished (incl. AET / penalties),
                    # is live, or is postponed/abandoned can still appear in the
                    # day's list with a final score and a stage marker — skip it
                    # so it isn't treated as an upcoming fixture to predict.
                    if self.mode == 'prediction':
                        has_score = (str(home_score or '').strip().isdigit()
                                     and str(away_score or '').strip().isdigit())
                        finished_markers = (
                            'finished', 'after pen', 'aet', 'awarded',
                            'walkover', 'abandoned', 'cancel', 'postp', 'retired',
                        )
                        stage_lower = stage_text.lower()
                        if has_score or any(m in stage_lower for m in finished_markers):
                            self.logger.info(
                                f"Skipping non-upcoming match (Prediction): "
                                f"{home_team} vs {away_team} "
                                f"[{stage_text or f'{home_score}-{away_score}'}]"
                            )
                            continue

                    item = MatchItem()
                    item['match_id'] = match_id
                    item['home_team'] = home_team
                    item['away_team'] = away_team
                    item['league'] = current_league  # Assign captured league
                    item['home_score'] = home_score
                    item['away_score'] = away_score
                    
                    # Optimization: Extract full URL for Direct Navigation
                    full_url = child.css('a.eventRowLink::attr(href)').get()
                    if full_url:
                        # Full URL: https://www.flashscore.com/match/slugs/?mid=ID
                        # Clean it
                        item['base_url'] = full_url.split('?')[0]
                    
                        # Filter Women's Matches (Consistent with Live List)
                        is_women = False
                        check_text = (item['home_team'] + " " + item['away_team']).lower()
                        
                        if " w " in " " + check_text + " ": is_women = True 
                        elif "(w)" in check_text: is_women = True
                        elif "women" in check_text: is_women = True
                        elif item['home_team'].endswith(" W") or item['away_team'].endswith(" W"): is_women = True
                        
                        if is_women:
                            self.logger.info(f"Skipping Women's match (Prediction): {item['home_team']} vs {item['away_team']}")
                            continue

                    matches_to_scrape.append(item)
                    
        self.logger.info(f"Found {len(matches_to_scrape)} matches for next day.")

        # await page.close() # Let Playwright handler close it or do it after yielding if needed, but safer to let it be.
        
        self.logger.info(f"Yielding {len(matches_to_scrape)} requests...")
        # debug limit removed

        for item in matches_to_scrape:
            # Verification Mode Optimization: Yield item directly with score, skip detail page
            # ONLY if we successfully captured names. If names are "Unknown", we MUST visit the page.
            if self.mode == 'verification':
                if item['home_team'] != "Unknown Home" and item['away_team'] != "Unknown Away":
                     self.logger.debug(f"Verification Mode: Yielding result for {item['match_id']} ({item['home_score']}-{item['away_score']})")
                     yield item
                     continue
                else:
                     self.logger.info(f"Verification Mode: Names unknown for {item['match_id']}, visiting detail page...")

            # Request 1X2 Odds
            # URL: .../match/{id}/#/odds-comparison/1x2-odds/full-time
            url = f"https://www.flashscore.com/match/{item['match_id']}/#/odds-comparison/1x2-odds/full-time"
            self.logger.debug(f"Yielding request for match {item['match_id']}")
            yield scrapy.Request(
                url=url,
                meta={
                    "playwright": True,
                    "playwright_include_page": True,
                    "playwright_page_goto_kwargs": {
                        "timeout": 30000,
                        "wait_until": "domcontentloaded",
                    },
                    "item": item,
                    "playwright_context": "odds_context", # reusing context
                },
                callback=self.parse_odds_1x2
            )

    async def parse_odds_1x2(self, response):
        page = response.meta["playwright_page"]
        item = response.meta["item"]
        
        try:
            self.logger.info(f"Processing match {item['match_id']} odds.")

            # Extract Team Names from Title (Robust)
            if item['home_team'] == "Unknown Home":
                 title = await page.title()
                 # Format: "Home - Away | Home v Away ..." or similar
                 # Example: "KAI - OLY | Kairat Almaty v Olympiacos Piraeus ..."
                 if "|" in title:
                     parts = title.split("|")
                     if len(parts) > 1:
                         match_part = parts[1].strip()
                         # Format: "Kairat Almaty v Olympiacos Piraeus" or "Home - Away"
                         if " v " in match_part:
                             teams = match_part.split(" v ")[0:2]
                             item['home_team'] = teams[0].strip()
                             # Cleanup away name (often has extra info)
                             away_full = teams[1]
                             for sep in [" LIVE", " - ", " |"]:
                                 away_full = away_full.split(sep)[0]
                             item['away_team'] = away_full.strip()

            # --- 2. EXTRACT 1X2 ODDS ---
            # Wait for odds
            try:
                await page.wait_for_selector("[class*='oddsValue']", timeout=5000)
            except:
                self.logger.warning(f"Timeout waiting for 1X2 odds for {item['match_id']}")

            odds_elements = page.locator("[class*='oddsValue']")
            odds_texts = await odds_elements.all_text_contents()
            
            if len(odds_texts) >= 3:
                item['interaction_1x2_1'] = odds_texts[0]
                item['interaction_1x2_X'] = odds_texts[1]
                item['interaction_1x2_2'] = odds_texts[2]
            else:
                self.logger.warning(f"1X2 Odds length {len(odds_texts)} insufficient.")

            # --- EXTRACT START TIME ---
            try:
                # Format: "12.08.2023 12:30"
                start_time = await page.locator(".duelParticipant__startTime").inner_text()
                if start_time:
                    item['start_time'] = start_time.strip()
                else:
                    self.logger.warning(f"Start time not found for {item['match_id']}")
            except Exception as e:
                self.logger.warning(f"Error extracting start time: {e}")

            # --- EXTRACT SCORES (If available) ---
            try:
                # Scores are usually in .duelParticipant__score
                # Home: .duelParticipant__home .participant__score
                # Away: .duelParticipant__away .participant__score
                # Flashscore often uses simpler wrapper
                home_score = await page.locator(".duelParticipant__home .participant__score").inner_text()
                away_score = await page.locator(".duelParticipant__away .participant__score").inner_text()
                
                if home_score and away_score:
                    item['home_score'] = home_score
                    item['away_score'] = away_score
            except:
                pass # Expected for upcoming matches

            # --- 3. EXTRACT H2H DATA (Last 5 Matches) ---
            # --- 3. EXTRACT H2H DATA (Last 5 Matches) ---
            try:
                # Use get_by_text for robust selection, similar to O/U tab
                h2h_tab = page.get_by_text("H2H", exact=True)
                
                # Check visibility
                if await h2h_tab.count() > 0:
                     await h2h_tab.first.click()
                     self.logger.info(f"Clicked H2H tab for {item['match_id']}")
                     # Wait for an actual ROW, not just the .h2h__section shell: clicking H2H
                     # re-renders the panel and the section element appears (so a bare
                     # wait_for_selector(".h2h__section") returns) a beat BEFORE its rows are
                     # populated — parse_h2h_section's `sections.count()` then reads 0 and we
                     # silently return empty. The sections also load progressively (home first,
                     # away a moment later), so settle briefly to let section 1 fill in too.
                     await page.wait_for_selector(".h2h__section .h2h__row", timeout=12000)
                     await page.wait_for_timeout(1500)

                     
                     async def parse_h2h_section(section_index):
                         results = []
                         sections = page.locator(".h2h__section")
                         if await sections.count() > section_index:
                             rows = sections.nth(section_index).locator(".h2h__row")
                             count = await rows.count()
                             for i in range(min(count, 5)):
                                 row = rows.nth(i)
                                 try:
                                     # Flashscore renamed the date cell to `wclH2h__date` in the wcl-* redesign
                                     # (legacy `h2h__date` dropped); match either so we survive both DOM versions.
                                     date = await row.locator(".h2h__date, .wclH2h__date").first.inner_text()
                                     home = await row.locator(".h2h__homeParticipant").first.inner_text()
                                     away = await row.locator(".h2h__awayParticipant").first.inner_text()
                                     score = ""
                                     try:
                                         # Pattern: Digit(s) - Digit(s) (handling hyphen, en-dash, spaces, newlines)
                                         import re
                                         score_pattern = re.compile(r"^\s*(\d+)\s*[-–\n]\s*(\d+)\s*$")
                                         
                                         spans = row.locator("span")
                                         count_spans = await spans.count()
                                         
                                         for k in range(count_spans):
                                             txt = await spans.nth(k).inner_text()
                                             match = score_pattern.match(txt)
                                             if match:
                                                 # Normalize to "H-A" format
                                                 score = f"{match.group(1)}-{match.group(2)}"
                                                 break
                                         
                                         # Fallback: Check direct text if no span matched (rare)
                                         if not score:
                                             divs = row.locator("div")
                                             count_divs = await divs.count()
                                             for k in range(count_divs):
                                                 txt = await divs.nth(k).inner_text()
                                                 match = score_pattern.match(txt)
                                                 if match:
                                                     score = f"{match.group(1)}-{match.group(2)}"
                                                     break

                                     except Exception as e:
                                         self.logger.warning(f"Error parsing score: {e}")

                                     results.append({
                                         "date": date.strip(),
                                         "home_team": home.strip(),
                                         "away_team": away.strip(),
                                         "score": score
                                     })
                                 except Exception as row_e:
                                     self.logger.warning(f"Error parsing row {i} in section {section_index}: {row_e}")
                         return results

                     # Section 0: Home Team Last Matches
                     # Section 1: Away Team Last Matches
                     item['last_matches_home'] = await parse_h2h_section(0)
                     item['last_matches_away'] = await parse_h2h_section(1)
                else:
                     self.logger.warning(f"H2H tab not found for {item['match_id']}")

            except Exception as e:
                self.logger.warning(f"Error extracting H2H data: {e}")

            # --- 3. NAVIGATE TO O/U (Optimized Direct Strategy) ---
            try:
                base_url = item.get('base_url')
                if base_url:
                    if base_url.endswith('/'): base_url = base_url[:-1]
                    ou_url = f"{base_url}/odds/over-under/full-time/?mid={item['match_id']}"
                    
                    # Direct Navigation with fast wait
                    await page.goto(ou_url, timeout=30000, wait_until='domcontentloaded')
                    
                    # Wait for generic table to ensure basics loaded
                    try:
                        await page.wait_for_selector('.ui-table', timeout=5000)
                    except:
                        pass
                else:
                    # Fallback: Summary -> User URL Strategy (If base_url missing)
                    summary_url = f"https://www.flashscore.com/match/{item['match_id']}/#/match-summary"
                    await page.goto(summary_url, timeout=30000)
                    await page.wait_for_load_state('domcontentloaded') # Faster wait
                    
                    current_url = page.url
                    if "/match/" in current_url:
                        base_url_fallback = current_url.split('#')[0].split('?')[0]
                        if base_url_fallback.endswith('/'): base_url_fallback = base_url_fallback[:-1]
                        ou_url = f"{base_url_fallback}/odds/over-under/full-time/?mid={item['match_id']}"
                        await page.goto(ou_url, timeout=30000, wait_until='domcontentloaded')

            except Exception as e:
                 self.logger.warning(f"Error navigating O/U via Optimized User URL: {e}")

            # --- 4. EXTRACT O/U 2.5 ODDS (Row-Based Strategy) ---
            # Iterate .ui-table__row elements to find the specific "2.5" line row
            found_odds = False
            import re
            
            # Broad selector for rows - works for both odds and standings, so we must filter by content
            rows = page.locator('.ui-table__row') 
            count = await rows.count()
            
            for i in range(count):
                row = rows.nth(i)
                text = await row.inner_text()
                # Clean text
                clean_text = text.replace("\n", " ").strip()
                
                # Pattern: Start with "2.5" or contain "2.5" distinctly
                # Usually "2.5" is the first text in the row for that line
                if re.search(r'(?:^|[^\d])2\.5(?:[^\d]|$)', clean_text):
                    # This row contains "2.5". Is it the odds row?
                    # Check for odds numbers.
                    nums = re.findall(r'\d+\.\d+', clean_text)
                    
                    # Logic:
                    # If row is "2.5  1.50  2.50", nums=['2.5', '1.50', '2.50']
                    # If row is "2.5  1.50", nums=['2.5', '1.50'] (Missing under?)
                    
                    # We expect at least 3 numbers (Line + 2 Odds) OR 2 numbers if line is integer? No 2.5 is float.
                    if len(nums) >= 3:
                        # Assume first is line, next two are Over/Under
                        # But sometimes order is different?
                        # Usually Flashscore O/U columns: Over | Under
                        item['over_2_5'] = nums[1]
                        item['under_2_5'] = nums[2]
                        found_odds = True
                        self.logger.info(f"O/U 2.5 Found: {item['over_2_5']} / {item['under_2_5']} (Row: {clean_text[:50]}...)")
                        break
                    elif len(nums) == 2:
                         # Maybe 2.5 isn't captured as float? But regex matched 2.5
                         # If nums=['1.50', '2.50'] and line 2.5 was text only?
                         item['over_2_5'] = nums[0]
                         item['under_2_5'] = nums[1]
                         found_odds = True
                         self.logger.info(f"O/U 2.5 Found (2 nums): {item['over_2_5']} / {item['under_2_5']}")
                         break
            
            if not found_odds:
                 self.logger.warning(f"O/U 2.5 row not found for {item['match_id']}")
                
        except Exception as e:
             pass
        finally:
            await page.close()
        
        yield item

    async def parse_live_batch(self, response):
        page = response.meta["playwright_page"]
        ids = response.meta["live_ids"].split(',')
        self.logger.info(f"Starting Batch Live Scraping for {len(ids)} matches...")
        
        for match_id in ids:
            if not match_id: continue
            
            # 0.5 Handle Cookie Banner (Just in case)
            try:
                if await page.query_selector("button#onetrust-accept-btn-handler"):
                    await page.click("button#onetrust-accept-btn-handler")
                    await page.wait_for_timeout(500)
            except: pass

            # 1. Navigate to the match page (no URL fragment — Flashscore's
            # SPA doesn't auto-route #/match-statistics/0 to the stats panel
            # reliably; we click the STATS tab below instead).
            url = f"https://www.flashscore.com/match/{match_id}/"
            try:
                await page.goto(url, timeout=20000, wait_until='domcontentloaded')

                # 2. Wait for Header (Base)
                try:
                    await page.wait_for_selector('.detailScore__wrapper', timeout=5000)
                except: pass

                # 3. Click the STATS tab. Flashscore's tab labels are
                # all-caps in the DOM ("STATS") — case-insensitive has-text
                # handles that. Wait long enough for the stats panel to
                # populate (it lazy-loads via XHR after click).
                try:
                    stats_tab = page.locator('a:has-text("STATS"), button:has-text("STATS")')
                    if await stats_tab.count() > 0:
                        await stats_tab.first.click()
                        await page.wait_for_timeout(3500)
                except: pass

                # Debug: Screenshot the first match to see structure.
                # output/ is gitignored so this doesn't pollute the repo root.
                if match_id == ids[0]:
                    await page.screenshot(path="output/debug_live_batch_stats.png")

                # 3. Extract Data (Regex on Text - Most Robust).
                # Stats appear as 3 lines: home_value \n label \n away_value.
                # Possession is "44%" (with % sign), others are plain numbers.
                data = await page.evaluate(r"""
                    () => {
                        const bodyText = document.body.innerText;
                        const stats = {};

                        function extractStat(label, key) {
                            // home_value [newlines/spaces] LABEL [newlines/spaces] away_value
                            // Values may have a trailing % (possession).
                            const regex = new RegExp(
                                `([\\d\\.]+)%?[\\s\\n]*${label}[^\\n]*[\\s\\n]+([\\d\\.]+)%?`,
                                'i'
                            );
                            const match = bodyText.match(regex);
                            if (match) {
                                stats[key + '_home'] = parseFloat(match[1]);
                                stats[key + '_away'] = parseFloat(match[2]);
                            }
                        }

                        // Core stats (verified labels from live Flashscore DOM).
                        extractStat('Expected goals', 'xg');         // "Expected goals (xG)"
                        extractStat('Total shots', 'shots');
                        extractStat('Goal Attempts', 'shots');       // Alt name (old layout)
                        extractStat('Ball possession', 'possession');
                        extractStat('Corner kicks', 'corners');
                        extractStat('Shots on target', 'shots_on_target');

                        // Extended stats — future-proofs live_history_*.jsonl.
                        // Labels are best-guess based on Flashscore's typical
                        // shot/dominance panel headings. Misses (label not on
                        // page for this match) leave the key absent; the
                        // dashboard handles that with '-' fallback.
                        extractStat('xG on target', 'xgot');         // "xG on target (xGOT)"
                        extractStat('Big chances', 'big_chances');
                        extractStat('Shots inside the box', 'shots_inside_box');
                        extractStat('Shots outside the box', 'shots_outside_box');
                        extractStat('Hit woodwork', 'woodwork');
                        extractStat('Touches in opposition box', 'touches_opp_box');
                        extractStat('Touches in opposition’s box', 'touches_opp_box'); // curly apostrophe
                        extractStat('Saves', 'saves');
                        extractStat('Yellow cards', 'yellow_cards');
                        extractStat('Red cards', 'red_cards');
                        extractStat('Fouls', 'fouls');
                        
                        let scores = document.querySelector('.detailScore__wrapper')?.innerText || '0-0';
                        scores = scores.replace(/\n/g, '').replace(/\s+/g, '').replace(/-/g, '-'); 
                        const hScore = document.querySelector('.detailScore__wrapper span:nth-child(1)')?.innerText;
                        const aScore = document.querySelector('.detailScore__wrapper span:nth-child(3)')?.innerText;
                        if(hScore && aScore) scores = hScore + "-" + aScore;
                        
                        // The live match minute/status lives in .event__stage--block
                        // (e.g. "67'", "45+2'", "Half Time", "Finished"). Use it as
                        // the authoritative source; fall back to the older selectors
                        // only when it's absent.
                        let time = document.querySelector('.event__stage--block')?.innerText
                                 || document.querySelector('.eventTime')?.innerText
                                 || document.querySelector('.detailScore__status')?.innerText
                                 || '0';
                        
                        // Check for 2nd Half Status for time correction
                        const bodyUpper = document.body.innerText.toUpperCase();
                        const is2nd = bodyUpper.includes("2ND HALF") || bodyUpper.includes("SECOND HALF");
                        
                        // Robust time extraction. The primary selector (.eventTime)
                        // gives the live clock; when it's empty (e.g. at Half Time)
                        // .detailScore__status returns a status WORD ("HALF TIME")
                        // with no ':' or "'". Earlier this fell through to a
                        // document.body-wide mm:ss regex — which wrongly matched the
                        // KICKOFF time (e.g. "17:00" in .duelParticipant__startTime),
                        // freezing every match that kicked off at HH:00 at minute 17.
                        // Fix: explicit status words win first, and any clock scan is
                        // scoped to the status element only (never the whole body,
                        // which contains the kickoff / wall-clock time).
                        if (!time || (!time.includes(':') && !time.includes("'"))) {
                             if (bodyUpper.includes("HALF TIME") || bodyUpper.includes("HALFTIME")) {
                                 time = "45";
                             } else if (bodyUpper.includes("FINISHED") || bodyUpper.includes("FULL TIME")) {
                                 time = "90";
                             } else {
                                 const statusEl = document.querySelector(
                                     '.event__stage--block, .detailScore__status, .eventTime, .fixedHeaderDuel__detailStatus');
                                 const statusTxt = statusEl ? statusEl.innerText : '';
                                 const tm = statusTxt.match(/(\d{1,3})'/) || statusTxt.match(/(\d{1,3}):(\d{2})/);
                                 if (tm) time = tm[0]; // e.g. "63:06" or "67'"
                             }
                        }

                        
                        return { score: scores, time: time, stats: stats, is_second_half: is2nd };
                    }
                """)
                
                # 4. Parse Minute
                min_val = 0
                time_str = str(data.get('time', '')).strip().upper()
                is_2nd_half = data.get('is_second_half', False)
                
                # Regex for MM:SS
                import re
                mm_ss = re.search(r'(\d+):(\d+)', time_str)
                if mm_ss:
                    min_val = int(mm_ss.group(1))
                    # Correction: If 2nd Half and time < 45 (e.g. 18' meaning 63'), add 45.
                    # Some views might show relative time.
                    if is_2nd_half and min_val < 45:
                        min_val += 45
                        
                    if min_val > 90 and not "EXTRA" in time_str: min_val = 90 # Cap regular time
                    
                elif "HALF" in time_str:
                    min_val = 45
                    if is_2nd_half or "2ND" in time_str: min_val = 45 # Fallback if no digits found
                elif "FULL" in time_str or "FINISH" in time_str:
                    min_val = 90
                elif "'" in time_str:
                    try:
                        val = int(time_str.split("'")[0])
                        if is_2nd_half and val < 45: val += 45
                        min_val = val
                    except:
                        if "+" in time_str:
                             parts = time_str.split("'")[0].split('+')
                             if len(parts) > 0 and parts[0].isdigit():
                                 min_val = int(parts[0])
                else:
                    if time_str.isdigit():
                        val = int(time_str)
                        if is_2nd_half and val < 45: val += 45
                        min_val = val

                # 4.5 Fallback: Look for time in body text via regex if selector failed (often "2nd Half" selector hides the time)
                # If min_val is 0 or 45 but score suggests game is on.
                if min_val <= 45 and "2ND" in time_str:
                     # Attempt to find "63:06" pattern in body near score? Too risky?
                     pass

                # Log
                self.logger.info(f"Match {match_id}: {data['score']} ({min_val}')")
                
                # VERIFICATION MODE: Yield standard item for evaluation
                if self.mode == 'verification':
                     try:
                         # Extract teams for confirmation (optional but good)
                         teams = await page.evaluate("""() => {
                             const h = document.querySelector('.participant__participantName--home')?.innerText || 
                                       document.querySelector('.participant__participantName')?.innerText || "Home";
                             const a = document.querySelector('.participant__participantName--away')?.innerText || 
                                       document.querySelectorAll('.participant__participantName')[1]?.innerText || "Away";
                             return [h, a];
                         }""")
                         
                         score_parts = data['score'].split('-')
                         h_s = score_parts[0] if len(score_parts) > 0 else '0'
                         a_s = score_parts[1] if len(score_parts) > 1 else '0'
                         
                         item = MatchItem()
                         item['match_id'] = match_id
                         item['home_team'] = teams[0].strip()
                         item['away_team'] = teams[1].strip()
                         item['home_score'] = h_s
                         item['away_score'] = a_s
                         item['league'] = "Verified via ID" # Placeholder
                         
                         yield item
                     except Exception as e:
                         self.logger.error(f"Error extracting verification data for {match_id}: {e}")
                
                else: 
                    # LIVE MODE: Yield stats
                    yield {
                        'match_id': match_id,
                        'live_data': True,
                        'score': data['score'],
                        'minute': min_val,
                        'stats': data['stats']
                    }
                
            except Exception as e:
                self.logger.error(f"Error scraping match {match_id}: {e}")
        
        await page.close()

    async def parse_live_list_page(self, response):
        page = response.meta["playwright_page"]
        
        try:
            # Click LIVE tab - robust try
            try:
                # Wait for at least one match to appear first to ensure page loaded
                await page.wait_for_selector('[id^="g_1_"]', timeout=15000)
                
                # Then try to click LIVE
                await page.click('.filters__tab:has-text("LIVE")')
                await page.wait_for_timeout(2000) 
            except:
                pass 
            
            # Use Locators (slower but verified to work in debug script)
            matches = []
            rows = page.locator('[id^="g_1_"]')
            count = await rows.count()
            self.logger.info(f"Spider found {count} match elements via Locators.")
            
            for i in range(count):
                row = rows.nth(i)
                id_attr = await row.get_attribute('id')
                if not id_attr: continue
                match_id = id_attr.replace('g_1_', '')
                
                # Try new selector first
                h_loc = row.locator('.event__homeParticipant')
                if await h_loc.count() == 0:
                   h_loc = row.locator('.event__participant--home') # Fallback
                   
                a_loc = row.locator('.event__awayParticipant')
                if await a_loc.count() == 0:
                   a_loc = row.locator('.event__participant--away') # Fallback

                h_text = await h_loc.inner_text() if await h_loc.count() > 0 else None
                a_text = await a_loc.inner_text() if await a_loc.count() > 0 else None
                
                if h_text: h_text = h_text.split('\n')[0]
                if a_text: a_text = a_text.split('\n')[0]
                
                if match_id and h_text and a_text:
                    # Filter Women's Matches
                    is_women = False
                    check_text = (h_text + " " + a_text).lower()
                    
                    # Patterns: "arsenal w", "usa (w)", "chelsea women"
                    if " w " in " " + check_text + " ": is_women = True 
                    elif "(w)" in check_text: is_women = True
                    elif "women" in check_text: is_women = True
                    # Common suffixes check
                    elif h_text.endswith(" W") or a_text.endswith(" W"): is_women = True
                    
                    if is_women:
                        self.logger.info(f"Skipping Women's match: {h_text} vs {a_text}")
                        continue

                    matches.append({'match_id': match_id, 'home_team': h_text, 'away_team': a_text})
            
            for m in matches:
                yield m
        except Exception as e:
            self.logger.error(f"Live List Error: {e}")
        finally:
            await page.close()

