import scrapy
import csv
import os


class StandingsSpider(scrapy.Spider):
    name = "standings"

    # Flashscore rate-limits / flakes under the default concurrency of 4.
    custom_settings = {
        "CONCURRENT_REQUESTS": 2,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "DOWNLOAD_DELAY": 2,
    }

    TABLE_SELECTOR = ".ui-table__row"
    TABLE_WAIT_MS = 45_000
    TABLE_ATTEMPTS = 3

    def start_requests(self):
        csv_path = os.path.join(
            self.settings.get("PROJECT_ROOT", "."),
            "data_sets/standings_form_flashscore_direct_links.csv",
        )
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"Missing {csv_path}. Seed it from "
                "data_sets/standings_form_flashscore_direct_links.template.csv "
                "(bin/update_leagues_data.sh does this automatically)."
            )

        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                meta = {
                    "country": row["COUNTRY"],
                    "league": row["LEAGUE"],
                    "playwright": True,
                    "playwright_include_page": True,
                    "playwright_page_goto_kwargs": {
                        "wait_until": "domcontentloaded",
                        "timeout": 60000,
                    },
                }

                yield scrapy.Request(
                    url=row["STANDINGS_OVERALL"],
                    callback=self.parse_standings,
                    meta=meta.copy(),
                    dont_filter=True,
                    errback=self._errback_close_page,
                )

                yield scrapy.Request(
                    url=row["FORM_LAST_5_OVERALL"],
                    callback=self.parse_form,
                    meta=meta.copy(),
                    dont_filter=True,
                    errback=self._errback_close_page,
                )

                if row.get("FORM_LAST_10_OVERALL"):
                    meta_10 = meta.copy()
                    meta_10["form_type"] = "last_10"
                    yield scrapy.Request(
                        url=row["FORM_LAST_10_OVERALL"],
                        callback=self.parse_form,
                        meta=meta_10,
                        dont_filter=True,
                        errback=self._errback_close_page,
                    )

    async def _errback_close_page(self, failure):
        page = failure.request.meta.get("playwright_page")
        if page:
            try:
                await page.close()
            except Exception:
                pass

    @staticmethod
    def _variant_url(base_url: str, variant: str) -> str:
        """Build /standings/home/ or /form/away/ from the overall URL."""
        base = base_url.rstrip("/") + "/"
        if variant == "overall":
            return base
        return f"{base}{variant}/"

    async def _dismiss_consent(self, page):
        """Best-effort cookie / consent dismiss — Flashscore often blocks the table behind it."""
        selectors = (
            "#onetrust-accept-btn-handler",
            "button#onetrust-accept-btn-handler",
            "button:has-text('Accept All')",
            "button:has-text('I Accept')",
            "button:has-text('Accept')",
            ".acceptCookies",
        )
        for sel in selectors:
            try:
                loc = page.locator(sel).first
                if await loc.count() == 0:
                    continue
                if await loc.is_visible(timeout=800):
                    await loc.click(timeout=2000)
                    await page.wait_for_timeout(400)
                    return
            except Exception:
                continue

    async def _wait_for_table(self, page, label: str = ""):
        """Wait for the standings/form table, reloading on transient Flashscore flakes."""
        await self._dismiss_consent(page)
        last_err = None
        for attempt in range(1, self.TABLE_ATTEMPTS + 1):
            try:
                await page.wait_for_selector(
                    self.TABLE_SELECTOR, timeout=self.TABLE_WAIT_MS
                )
                return
            except Exception as e:
                last_err = e
                self.logger.warning(
                    "Table not ready%s (attempt %d/%d): %s",
                    f" [{label}]" if label else "",
                    attempt,
                    self.TABLE_ATTEMPTS,
                    e,
                )
                if attempt < self.TABLE_ATTEMPTS:
                    try:
                        await page.reload(
                            wait_until="domcontentloaded", timeout=60000
                        )
                    except Exception as reload_err:
                        self.logger.warning("Reload failed: %s", reload_err)
                    await self._dismiss_consent(page)
                    await page.wait_for_timeout(2000)
        raise last_err

    async def _goto_and_extract(self, page, url: str, table_type: str, label: str):
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await self._wait_for_table(page, label=label)
        data = await self.extract_table(page, table_type)
        if not data:
            raise RuntimeError(f"Parsed 0 rows from {url}")
        return data

    async def parse_standings(self, response):
        page = response.meta["playwright_page"]
        league = response.meta["league"]
        country = response.meta["country"]
        base_url = response.url

        try:
            await self._wait_for_table(page, label=f"{league} standings/overall")
            overall_data = await self.extract_table(page, "standings")
            if overall_data:
                yield {
                    "type": "standings_overall",
                    "country": country,
                    "league": league,
                    "table": overall_data,
                }
            else:
                self.logger.error(
                    "Empty overall standings for %s — skipping home/away", league
                )
                return

            for variant in ("home", "away"):
                try:
                    url = self._variant_url(base_url, variant)
                    data = await self._goto_and_extract(
                        page, url, "standings", f"{league} standings/{variant}"
                    )
                    yield {
                        "type": f"standings_{variant}",
                        "country": country,
                        "league": league,
                        "table": data,
                    }
                except Exception as e:
                    self.logger.error(
                        "Error extracting %s standings for %s: %s",
                        variant.capitalize(),
                        league,
                        e,
                    )
        finally:
            await page.close()

    async def parse_form(self, response):
        page = response.meta["playwright_page"]
        league = response.meta["league"]
        country = response.meta["country"]
        form_type = response.meta.get("form_type", "last_5")
        base_url = response.url

        try:
            await self._wait_for_table(page, label=f"{league} {form_type}/overall")
            overall_data = await self.extract_table(page, "form")
            if overall_data:
                yield {
                    "type": f"{form_type}_matches_overall",
                    "country": country,
                    "league": league,
                    "table": overall_data,
                }
            else:
                self.logger.error(
                    "Empty overall form (%s) for %s — skipping home/away",
                    form_type,
                    league,
                )
                return

            for variant in ("home", "away"):
                try:
                    url = self._variant_url(base_url, variant)
                    data = await self._goto_and_extract(
                        page, url, "form", f"{league} {form_type}/{variant}"
                    )
                    yield {
                        "type": f"{form_type}_matches_{variant}",
                        "country": country,
                        "league": league,
                        "table": data,
                    }
                except Exception as e:
                    self.logger.error(
                        "Error extracting %s form (%s) for %s: %s",
                        variant,
                        form_type,
                        league,
                        e,
                    )
        finally:
            await page.close()

    async def extract_table(self, page, table_type):
        rows = await page.query_selector_all(self.TABLE_SELECTOR)
        self.logger.info("Extracting %s: Found %d rows.", table_type, len(rows))
        data = []
        for row in rows:
            text = await row.inner_text()
            lines = text.split("\n")

            try:
                rank = lines[0].replace(".", "")
                team = lines[1]
                mp = lines[2]

                if table_type == "standings":
                    w = lines[3]
                    d = lines[4]
                    l = lines[5]
                    goals = lines[6]
                    item = {
                        "rank": rank,
                        "team_name": team,
                        "matches_played": mp,
                        "wins": w,
                        "draws": d,
                        "losses": l,
                        "goals": goals,
                        "goals_difference": lines[7] if len(lines) > 7 else 0,
                        "points": lines[8] if len(lines) > 8 else 0,
                    }
                    data.append(item)
                else:
                    w = lines[3]
                    d = lines[4]
                    l = lines[5]
                    goals = lines[6]
                    pts = lines[7] if ":" not in lines[7] else lines[8]

                    form_icons = await row.query_selector_all(".tableCellFormIcon")
                    form_str = ""
                    if form_icons:
                        texts = [await i.inner_text() for i in form_icons]
                        texts = [
                            t.strip() for t in texts if t.strip() and t.strip() != "?"
                        ]
                        form_str = "|".join(texts)

                    item = {
                        "rank": rank,
                        "team_name": team,
                        "matches_played": mp,
                        "last_5_results": form_str,
                        "goals": goals,
                        "goals_difference": "N/A",
                        "points": pts,
                    }

                    if ":" in goals:
                        gf, ga = goals.split(":")
                        item["goals_difference"] = int(gf) - int(ga)

                    data.append(item)

            except Exception:
                continue

        return data
