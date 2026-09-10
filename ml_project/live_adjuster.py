import math
import numpy as np


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k) for X ~ Poisson(lam)."""
    return sum(_poisson_pmf(i, lam) for i in range(k + 1))


class LiveAdjuster:
    """
    Adjusts pre-match probabilities based on live match statistics and game state.
    """
    def __init__(self):
        # Heuristic Weights
        self.WEIGHT_XG = 1.5
        self.WEIGHT_SOG = 0.5
        self.WEIGHT_POSSESSION = 0.05

        # Red-card swing per NET card (one team a man up). Applied to the
        # man-down team's win probability, scaled by remaining time. A
        # red card with the full game left is worth ~0.18 off the
        # disadvantaged team; near full-time it's worth much less (little
        # time for the advantage to matter). Capped at MAX_RED_SHIFT.
        self.WEIGHT_RED_CARD = 0.18
        self.MAX_RED_SHIFT = 0.35
        # Of the probability taken off the man-down team, how much goes
        # to the opponent's WIN vs to the DRAW. A team a man down is most
        # likely to lose, but also parks the bus → draw rises somewhat.
        self.RED_TO_OPPONENT = 0.65
        self.RED_TO_DRAW = 0.35

        # Thresholds
        self.DOMINANCE_THRESHOLD = 1.5 # Significant advantage

        # O/U adjuster tuning. Threshold is 2.5 → over means final ≥ 3 goals.
        self.OU_THRESHOLD = 2.5
        # Cap remaining-xG estimate. Very early-minute xG samples are noisy
        # (xG=1.0 at minute 5 implies a 18-goal full-time pace, which is silly).
        self.OU_MAX_REMAINING_XG = 6.0
        # Blend weight on observed-pace vs pre-match. Early game = trust pre-match,
        # late game = trust the score state. Crossover at ~30 minutes.
        self.OU_PACE_CROSSOVER_MIN = 30

    def adjust_ou_probabilities(self, pre_ou_probs, live_stats, minute, current_score,
                                trace=None):
        """Adjust Over/Under 2.5 probabilities based on score state + xG pace.

        Args:
            pre_ou_probs (dict): {'over': p, 'under': 1-p} from the model.
            live_stats (dict): xg_home, xg_away as observed so far.
            minute (int): 0-90+.
            current_score (str): "1-0", etc.
            trace (dict|None): if given, populated with the intermediate
                quantities behind the result (pace, blend weight, branch
                taken) so a stored snapshot can be audited after the fact.

        Returns:
            dict: {'over': p, 'under': 1-p} adjusted.
        """
        def _t(**kw):
            if trace is not None:
                trace.update(kw)

        try:
            h, a = map(int, current_score.split('-'))
        except (ValueError, AttributeError):
            _t(branch='unparseable_score')
            return dict(pre_ou_probs)

        current_goals = h + a
        # Already past threshold → Over is locked in.
        if current_goals >= 3:
            _t(branch='over_locked', current_goals=current_goals)
            return {'over': 0.99, 'under': 0.01}
        # No time left to score → Under is locked in.
        if minute >= 90:
            _t(branch='under_locked', current_goals=current_goals)
            return {'over': 0.01, 'under': 0.99}

        # Estimate remaining goals from observed xG pace.
        xg_so_far = live_stats.get('xg_home', 0) + live_stats.get('xg_away', 0)
        if minute > 0 and xg_so_far > 0:
            pace_per_min = xg_so_far / minute
            remaining_xg = min(pace_per_min * (90 - minute), self.OU_MAX_REMAINING_XG)
        else:
            # Fall back to a league-average ~2.6 goals/match pace.
            remaining_xg = 2.6 * (90 - minute) / 90

        # P(at least N more goals) where N = 3 - current_goals.
        need = max(0, 3 - current_goals)
        if need == 0:
            p_over_pace = 0.99
        else:
            p_over_pace = 1.0 - _poisson_cdf(need - 1, remaining_xg)

        # Blend with pre-match: early minutes trust pre-match, late minutes trust
        # observed-pace. Smooth crossover via sigmoid centered at OU_PACE_CROSSOVER_MIN.
        # Weight on pace goes 0→1 as minute goes 0→90.
        x = (minute - self.OU_PACE_CROSSOVER_MIN) / 15.0
        pace_weight = 1.0 / (1.0 + math.exp(-x))
        pre_over = float(pre_ou_probs.get('over', 0.5))
        p_over = pace_weight * p_over_pace + (1 - pace_weight) * pre_over

        # Clamp to keep callers safe.
        p_over = max(0.01, min(0.99, p_over))
        _t(branch='pace_blend', current_goals=current_goals,
           xg_so_far=round(xg_so_far, 3), remaining_xg=round(remaining_xg, 3),
           need=need, p_over_pace=round(p_over_pace, 4),
           pace_weight=round(pace_weight, 4), pre_over=round(pre_over, 4))
        return {'over': p_over, 'under': 1.0 - p_over}
        
    def adjust_probabilities(self, pre_probs, live_stats, minute, current_score,
                             trace=None):
        """
        Adjusts probabilities.
        
        Args:
            pre_probs (dict): {'home': 0.45, 'draw': 0.30, 'away': 0.25}
            live_stats (dict): {
                'xg_home': 1.2, 'xg_away': 0.4,
                'shots_home': 10, 'shots_away': 2,
                'possession_home': 60, 'possession_away': 40
            }
            minute (int): Current minute (0-90+)
            current_score (str): "1-0", "0-0", etc.
            
            trace (dict|None): if given, populated with the dominance score and
                the probability vector after each stage, so a stored snapshot
                shows WHICH stage moved the number and by how much. Without it
                the per-stage contributions are unrecoverable after the fact.

        Returns:
            dict: Adjusted probabilities {'home': ..., 'draw': ..., 'away': ...}
        """
        def _stage(name, probs):
            if trace is not None:
                trace.setdefault('stages', []).append(
                    {'stage': name, 'probs': {k: round(v, 4) for k, v in probs.items()}})

        # Parse Score
        try:
            h_score, a_score = map(int, current_score.split('-'))
        except:
            if trace is not None:
                trace['branch'] = 'unparseable_score'
            return pre_probs # Fail safe

        # 1. Calculate Dominance Score (Positive = Home Dominance, Negative = Away)
        dominance = self._calculate_dominance(live_stats)
        if trace is not None:
            trace['branch'] = 'adjusted'
            trace['dominance'] = round(dominance, 4)
        _stage('pre', pre_probs)

        # 2. Base Adjustment on Game State (Time Decay)
        # As time passes, the probability of the CURRENT outcome increases.
        adjusted_probs = self._apply_time_decay(pre_probs, h_score, a_score, minute)
        _stage('time_decay', adjusted_probs)

        # 3. Apply Dominance Modifier
        adjusted_probs = self._apply_dominance_modifier(adjusted_probs, dominance, h_score, a_score, minute)
        _stage('dominance', adjusted_probs)

        # 4. Apply Sterile Possession Penalty
        adjusted_probs = self._apply_sterile_possession(adjusted_probs, live_stats, minute)
        _stage('sterile_possession', adjusted_probs)

        # 5. Apply Red-Card Modifier (man-advantage going forward)
        adjusted_probs = self._apply_red_card_modifier(adjusted_probs, live_stats, minute)
        _stage('red_card', adjusted_probs)

        # Normalize
        total = sum(adjusted_probs.values())
        if total <= 0:
            if trace is not None:
                trace['branch'] = 'all_zero_failsafe'
            return pre_probs  # fail safe — never return all-zeros
        return {k: v/total for k, v in adjusted_probs.items()}

    def _apply_red_card_modifier(self, probs, stats, minute):
        """Shift win probability toward the team with the man advantage.

        Red cards are a forward-looking signal the pre-match model can't
        see: a team down to 10 men is markedly less likely to win and
        somewhat more likely to draw (defensive shell). Effect scales
        with remaining time — a 20th-minute red matters far more than an
        85th-minute one — and is capped so a double-sending-off doesn't
        nuke the distribution.

        `net > 0` ⇒ home has MORE reds ⇒ home is the disadvantaged side.
        """
        red_home = stats.get('red_cards_home', 0) or 0
        red_away = stats.get('red_cards_away', 0) or 0
        try:
            net = int(red_home) - int(red_away)
        except (TypeError, ValueError):
            return probs
        if net == 0:
            return probs

        # More time left → bigger swing. Floor at 0.25 so even a late red
        # nudges the numbers (the man-down team still has to hold on).
        time_left_frac = max(0.0, (90 - minute) / 90.0)
        scale = 0.25 + 0.75 * time_left_frac

        shift = min(abs(net) * self.WEIGHT_RED_CARD * scale, self.MAX_RED_SHIFT)

        new_probs = probs.copy()
        # Disadvantaged team (man down) vs advantaged team.
        down, up = ('home', 'away') if net > 0 else ('away', 'home')

        # Can't take more than what the down-team currently holds.
        taken = min(shift, new_probs.get(down, 0.0))
        new_probs[down] = new_probs.get(down, 0.0) - taken
        new_probs[up] = new_probs.get(up, 0.0) + taken * self.RED_TO_OPPONENT
        new_probs['draw'] = new_probs.get('draw', 0.0) + taken * self.RED_TO_DRAW
        return new_probs
        
    def _calculate_dominance(self, stats):
        xg_diff = stats.get('xg_home', 0) - stats.get('xg_away', 0)
        shot_diff = stats.get('shots_home', 0) - stats.get('shots_away', 0)
        poss_diff = (stats.get('possession_home', 50) - stats.get('possession_away', 50)) / 10 # Scale down
        
        score = (xg_diff * self.WEIGHT_XG) + (shot_diff * self.WEIGHT_SOG) + (poss_diff * self.WEIGHT_POSSESSION)
        return score

    def _apply_time_decay(self, probs, h_score, a_score, minute):
        """
        Shifts probabilities toward the current result.
        """
        # Identify current winning state
        curr_winner = 'draw'
        if h_score > a_score: curr_winner = 'home'
        elif a_score > h_score: curr_winner = 'away'
        
        new_probs = probs.copy()
        
        # 1. Immediate Goal Impact (The "Scoreboard Pressure")
        # Even at minute 0, a goal changes the baseline probability significantly.
        if curr_winner != 'draw':
            # Boost the leader immediately. 
            # If they were 0.33, they become higher.
            # Reduced aggressiveness: 0.25 -> 0.15 -> 0.08
            initial_boost = 0.08
            new_probs[curr_winner] = min(0.95, new_probs[curr_winner] + initial_boost)
            
            # Renormalize immediately to keep math sane before decay
            total = sum(new_probs.values())
            for k in new_probs:
                new_probs[k] /= total

        # 2. Time Decay
        # As time passes, certainty increases
        decay_factor = min(minute / 95.0, 1.0)
        # Target certainty at full time: Reduced 0.98 -> 0.92 to allow late drama potential
        target_prob = 0.92
        
        current_prob = new_probs[curr_winner]
        new_probs[curr_winner] = current_prob + (target_prob - current_prob) * decay_factor
        
        # Reduce others proportionally
        remaining_prob = 1.0 - new_probs[curr_winner]
        other_keys = [k for k in probs.keys() if k != curr_winner]
        sum_others = sum(probs[k] for k in other_keys) # Use original weights for distribution? 
        # Better to use current new_probs ratio essentially, but sum_others of new_probs is just (1-current_prob)
        # So we just scale down the others.
        
        sum_current_others = sum(new_probs[k] for k in other_keys)
        if sum_current_others > 0:
            for k in other_keys:
                new_probs[k] = (new_probs[k] / sum_current_others) * remaining_prob
        else:
            # Edge case where others were 0
            for k in other_keys: new_probs[k] = remaining_prob / len(other_keys)
        
        return new_probs

    def _apply_dominance_modifier(self, probs, dominance, h_score, a_score, minute):
        """
        Boosts probable winner based on dominance if they haven't secured the win yet.
        """
        new_probs = probs.copy()
        
        # --- 1. PRESSURE COOKER (xG divergence) ---
        # If drawing late (60+) but one team has huge xG advantage (> 1.0 diff), 
        # probability of them winning should be significantly higher than standard dominance.
        # "Knocking on the door"
        
        # Calculate raw xG diff roughly from dominance (or pass strictly? Let's infer from dominance approx or just use dominance)
        # Dominance ~ xG*1.5 + Shots*0.5. 
        # A dominance of 3.0 usually means xG diff ~1.0+ and shot diff ~5+.
        
        PRESSURE_THRESHOLD = 2.5
        
        # Case: Drawing but Home Piling Pressure
        if h_score == a_score and dominance > PRESSURE_THRESHOLD and minute > 55:
            # Massive Boost -> Reduced
            boost = 0.15
            new_probs['home'] += boost
            new_probs['draw'] -= (boost * 0.7)
            new_probs['away'] -= (boost * 0.3)
            
        # Case: Drawing but Away Piling Pressure
        elif h_score == a_score and dominance < -PRESSURE_THRESHOLD and minute > 55:
            boost = 0.15
            new_probs['away'] += boost
            new_probs['draw'] -= (boost * 0.7)
            new_probs['home'] -= (boost * 0.3)

        # --- 2. STANDARD DOMINANCE ---
        elif h_score == a_score and dominance > self.DOMINANCE_THRESHOLD:
            # Standard Boost -> Reduced
            boost = 0.07 * (dominance / 2.0)
            new_probs['home'] += boost
            new_probs['draw'] -= (boost / 2)
            new_probs['away'] -= (boost / 2)
            
        elif h_score == a_score and dominance < -self.DOMINANCE_THRESHOLD:
            boost = 0.07 * (abs(dominance) / 2.0)
            new_probs['away'] += boost
            new_probs['draw'] -= (boost / 2)
            new_probs['home'] -= (boost / 2)
            
        # --- 3. LATE EQUALIZER POTENTIAL ---
        # Home Losing 0-1 but Dominating
        if (a_score - h_score) == 1 and dominance > self.DOMINANCE_THRESHOLD and minute > 60:
             new_probs['draw'] += 0.08
             new_probs['away'] -= 0.08
             
        # Away Losing 1-0 but Dominating
        if (h_score - a_score) == 1 and dominance < -self.DOMINANCE_THRESHOLD and minute > 60:
             new_probs['draw'] += 0.08
             new_probs['home'] -= 0.08
             
        # Clamp values 0-1
        for k in new_probs:
            new_probs[k] = max(0.01, min(0.99, new_probs[k]))
            
        return new_probs

    def _apply_sterile_possession(self, probs, live_stats, minute):
        """
        Penalize teams with high possession but low xG (Ineffective).
        """
        if minute < 45: return probs
        
        new_probs = probs.copy()
        
        poss_h = live_stats.get('possession_home', 50)
        poss_a = live_stats.get('possession_away', 50)
        xg_h = live_stats.get('xg_home', 0)
        xg_a = live_stats.get('xg_away', 0)
        
        # Home Sterile: >65% poss, <0.3 xG (at 45m+)
        if poss_h > 65 and xg_h < 0.4:
            penalty = 0.08
            new_probs['home'] -= penalty
            new_probs['draw'] += (penalty * 0.6)
            new_probs['away'] += (penalty * 0.4) # Counter attack risk
            
        # Away Sterile
        if poss_a > 65 and xg_a < 0.4:
            penalty = 0.08
            new_probs['away'] -= penalty
            new_probs['draw'] += (penalty * 0.6)
            new_probs['home'] += (penalty * 0.4)
            
        return new_probs
