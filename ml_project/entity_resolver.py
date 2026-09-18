import json
import os
from rapidfuzz import process, fuzz

class EntityResolver:
    def __init__(self, elo_file="data_sets/elo_ratings.json", mapping_file="data_sets/team_mappings.json"):
        self.elo_file = elo_file
        self.mapping_file = mapping_file
        self.elo_data = {}
        self.mappings = {}
        self.load_data()
        
    def load_data(self):
        # team_mappings.json is gitignored deployment state; seed it from the
        # committed template so a fresh clone starts with the 553 accumulated
        # mappings instead of re-learning every club from scratch.
        try:
            from config_bootstrap import seed_all
            seed_all()
        except Exception:
            try:
                from ml_project.config_bootstrap import seed_all
                seed_all()
            except Exception:
                pass    # degrade to fuzzy matching rather than fail to load

        if os.path.exists(self.elo_file):
            with open(self.elo_file, 'r') as f:
                self.elo_data = json.load(f)
        else:
            print("Warning: ELO file not found. Run elo_scraper.py first.")
            
        if os.path.exists(self.mapping_file):
            with open(self.mapping_file, 'r') as f:
                self.mappings = json.load(f)
                
    def save_mappings(self):
        os.makedirs(os.path.dirname(self.mapping_file), exist_ok=True)
        with open(self.mapping_file, 'w') as f:
            json.dump(self.mappings, f, indent=4)
            
    def get_canonical_name(self, team_name):
        """
        Resolves the scraper team name to the canonical name in our database.
        Returns None if no match found.
        """
        if not team_name:
            return None
            
        # 1. Direct key match
        if team_name in self.elo_data:
            return team_name
            
        # 2. Checked Cached Mapping
        if team_name in self.mappings:
            mapped_name = self.mappings[team_name]
            # Verify it still exists in data? (Optional but good safety)
            return mapped_name
                
        # 3. Fuzzy Match
        choices = list(self.elo_data.keys())
        if not choices:
            return None
            
        match = process.extractOne(team_name, choices, scorer=fuzz.token_set_ratio)
        
        if match and match[1] >= 80:
            best_match = match[0]
            self.mappings[team_name] = best_match
            self.save_mappings()
            return best_match
        else:
            self.mappings[team_name] = None
            self.save_mappings()
            return None

    def get_elo(self, team_name):
        canon = self.get_canonical_name(team_name)
        if canon:
            return self.elo_data.get(canon)
        return None

if __name__ == "__main__":
    # Test
    resolver = EntityResolver()
    test_teams = ["Real Madrid", "Man City", "Barca", "Liverpool FC"]
    for t in test_teams:
        elo = resolver.get_elo(t)
        print(f"{t}: {elo}")
