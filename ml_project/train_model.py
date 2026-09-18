import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import json
import os
import gc
from data_loader import DataLoader
from feature_engineering import FeatureEngineer, H2H_FEATURES
from model_registry import get_spec, save_meta
import feature_engineering
print(f"DEBUG: Loaded feature_engineering from {feature_engineering.__file__}")

class ModelTrainer:
    def __init__(self, data_dir: str, model_family: str = 'xgboost',
                 ou_family: str = 'xgboost', draw_family: str = 'xgboost'):
        self.data_dir = data_dir
        # Per-head model families, each resolved through the registry seam.
        # 'xgboost' is production and byte-identical to the pre-seam path;
        # the other families (see model_registry) are swappable challengers.
        self.model_family = model_family   # 1X2 head
        self.ou_family = ou_family         # Over/Under 2.5 head
        self.draw_family = draw_family     # binary draw head
        self.common_features = [
            'H_form_pts', 'H_form_gf', 'H_form_ga',
            'A_form_pts', 'A_form_gf', 'A_form_ga',
            # ELO Ratings
            'H_elo', 'A_elo', 
            # 'elo_diff', # Removed
            
            # League Encoding
            'league_cat', # Re-added
            
            # H/A Specific Basic
            'H_home_pts', 'H_home_gf', 'H_home_ga', 'H_home_sf', 'H_home_sa',
            'A_away_pts', 'A_away_gf', 'A_away_ga', 'A_away_sf', 'A_away_sa',
            
            # Shots and Corners (Overall Form)
            'H_form_sf', 'H_form_sa', 'H_form_cf', 'H_form_ca',
            'A_form_sf', 'A_form_sa', 'A_form_cf', 'A_form_ca',

            # Longer-horizon form (2026-09-13). The L5 columns above are noisy
            # over a 5-match sample; L10/L15 give the model a slower-moving
            # read of the same quantities so it can separate a genuine level
            # shift from a short streak. Built by FeatureEngineer from
            # FORM_WINDOWS and mirrored at serve time by
            # predict_matches.get_team_stats off the same corpus.
            # Shot/corner columns are omitted here: they are absent for 23 of
            # 44 leagues (league-clustered, ~50% of rows) and a longer window
            # buys no extra coverage, so they would add width without signal.
            'H_form_pts_l10', 'H_form_gf_l10', 'H_form_ga_l10',
            'A_form_pts_l10', 'A_form_gf_l10', 'A_form_ga_l10',
            'H_form_pts_l15', 'H_form_gf_l15', 'H_form_ga_l15',
            'A_form_pts_l15', 'A_form_gf_l15', 'A_form_ga_l15',
            # Momentum: short-window form minus long-window form. The level is
            # already in the columns above; these give the model the delta
            # directly rather than making it learn a subtraction.
            'H_form_trend', 'A_form_trend',

            # NOTE: the nine h2h_* columns are deliberately NOT here. They are
            # still built by FeatureEngineer._add_h2h_features and mirrored at
            # serve time by predict_matches.get_h2h_stats, but feeding them to
            # the model was measured on 2026-09-18 and produced no edge: +0.00000
            # nats over the market, 0 of 12 picks changed on a matched retrain,
            # and a row-permuted placebo of the same nine columns scored HIGHER
            # (+0.00003). Re-adding them is one line; re-test with
            # scripts/experiment_h2h.py, which reads them off the frame whether
            # or not they are trained on. See FEATURE_ENGINEERING_IDEAS.md 1.6.

            # Implied Probabilities
            'IP_H', 'IP_D', 'IP_A',

            # NEW: Advanced Features (Re-enabled/Added for Draw Detection)
            'abs_elo_diff', 'abs_ppg_diff', 'abs_form_pts_diff',
            'elo_diff', # Re-added for context

            # Season-to-date PPG and league-relative attack/defense strength
            # (built by FeatureEngineer._add_ppg_strength_features at training,
            # mirrored from current standings by HeuristicAdjuster.get_team_strength
            # at prediction).
            'H_ppg', 'A_ppg', 'ppg_diff',
            'H_att', 'A_att', 'H_def', 'A_def',
            'att_def_diff',
        ]

    def prepare_data(self):
        print("Loading data...")
        loader = DataLoader(self.data_dir)
        df = loader.load_historical_data()

        # NOTE: We need full history for correct ELO calculation, but treating the huge DF triggers OOM.
        # OPTIMIZATION:
        # 1. Create a lightweight DF for ELO calculation (2010-Present)
        # 2. Filter the main DF early for Training (2020-Present) to save memory
        # 3. Merge ELOs back into the Training DF

        # 1. Lightweight History for ELO
        cols_for_elo = ['date', 'home_team', 'away_team', 'FTHG', 'FTAG']
        existing_cols = [c for c in cols_for_elo if c in df.columns]
        df_elo_history = df[existing_cols].copy().sort_values('date')

        print("Calculating ELO ratings on full history...")
        from elo_engine import EloTracker
        elo_tracker = EloTracker()
        df_elo_history = elo_tracker.process_history(df_elo_history)

        # Save current ELO ratings for prediction usage
        with open("data_sets/elo_ratings.json", "w") as f:
            json.dump(elo_tracker.ratings, f)
        print("Saved final ELO ratings to data_sets/elo_ratings.json")

        # 2. Filter Main DF for Feature Engineering (Buffer)
        # We need prior data to calculate rolling features for the start of the training period.
        buffer_date = pd.Timestamp("2019-01-01")
        training_start = pd.Timestamp("2020-01-01")
        print(f"Filtering data for FE since {buffer_date}...")
        df_fe = df[df['date'] >= buffer_date].copy()
        
        # Free up original huge DF
        del df
        gc.collect()
        
        # 3. Merge ELOs into df_fe
        home_elos = df_elo_history[['date', 'home_team', 'H_elo']].copy()
        df_fe = pd.merge(df_fe, home_elos, on=['date', 'home_team'], how='left')
        
        away_elos = df_elo_history[['date', 'away_team', 'A_elo']].copy()
        df_fe = pd.merge(df_fe, away_elos, on=['date', 'away_team'], how='left')
        
        # Clean up history df
        del df_elo_history
        gc.collect()
        
        # 4. Feature Engineering on the buffered DF
        print("Engineering rolling features...")
        fe = FeatureEngineer()
        df_fe = fe.add_rolling_features(df_fe)
        
        # 5. Filter for Final Training Set (2020-Present)
        print(f"Filtering final training set since {training_start}...")
        df_train = df_fe[df_fe['date'] >= training_start].copy()
        
        # Targets
        df_train['target_1x2'] = df_train['FTR'].map({'H': 0, 'D': 1, 'A': 2})
        df_train['target_draw'] = (df_train['FTR'] == 'D').astype(int)
        df_train['total_goals'] = df_train['FTHG'] + df_train['FTAG']
        df_train['target_ou'] = df_train.apply(lambda x: 1 if (x['FTHG'] + x['FTAG']) > 2.5 else 0, axis=1) # Keep for legacy check if needed
        
        return df_train

    def train_draw(self, df):
        print(f"\n--- Training Binary Draw Model (Stage A, family={self.draw_family}) ---")
        # Resolved through the seam. NOTE: the draw head is trained but NOT
        # served at inference (predict_matches leaves model_draw=None) — see
        # FOOTBALL_NEXT_STEPS D0. Seam-routing keeps it swappable for future
        # reactivation / experiments.
        spec = get_spec('draw', self.draw_family)

        features = ['B365D', 'abs_elo_diff', 'abs_ppg_diff', 'abs_form_pts_diff'] + self.common_features
        features = list(dict.fromkeys(features)) # Remove duplicates while preserving order
        if not spec.uses_categorical and 'league_cat' in features:
            features = [f for f in features if f != 'league_cat']

        # We focus on features relevant to balance/uncertainty
        df_train = self._training_rows(df, features, 'target_draw')
        df_train = df_train.sort_values('date')

        if spec.uses_categorical and 'league_cat' in df_train.columns:
            df_train['league_cat'] = df_train['league_cat'].astype('category')

        # Class balance note: draws are ~25%. Default (no scale_pos_weight,
        # i.e. weight 1.0) produces well-calibrated probabilities — mean
        # predicted P(draw) ≈ 0.26, matching the actual rate of 0.25
        # almost exactly. The "Recall: 0.0000" line in earlier logs was
        # misleading: it was the metric at threshold 0.5, while the model
        # typically outputs probabilities in the 0.20-0.30 range. Production
        # uses the raw probability via predict_proba (not the threshold),
        # so the model contributes reasonable values regardless.
        #
        # Setting scale_pos_weight > 1.0 increases recall at the cost of
        # calibration: spw=3.0 gives recall@0.5 ≈ 0.63 but overcalibrates
        # mean P(draw) to ~0.50 (2× the actual rate), biasing every
        # prediction toward draws. Not worth the trade.
        #
        # The real limitation is that draws are genuinely hard to predict
        # from these features — the binary draw model has low discriminative
        # power even at lower thresholds (rec@0.3 ≈ 0.23). It's effectively
        # a near-constant predictor that contributes little signal beyond
        # the multi-class 1X2 model. See FOOTBALL_NEXT_STEPS D0 for the proper fix
        # (richer features or different architecture). The XGBoost params
        # (binary:logistic, n_est 100, lr 0.05, max_depth 4) now live in the
        # draw spec in model_registry.

        # Quick Train (Validation split)
        split_idx = int(len(df_train) * 0.90)
        train_data = df_train.iloc[:split_idx]
        valid_data = df_train.iloc[split_idx:]

        # Plain fit (no eval_set) — the draw params carry no early stopping,
        # so the prior eval_set never altered the fitted model.
        model = spec.build()
        model.fit(train_data[features], train_data['target_draw'])

        # Metrics — report at multiple thresholds since the default 0.5
        # is the wrong cut for a positive-weighted binary classifier.
        # Production uses the raw probability (no threshold), so the
        # threshold metrics are diagnostic only.
        probs = model.predict_proba(valid_data[features])[:, 1]
        actual_draw_rate = valid_data['target_draw'].mean()
        mean_p = probs.mean()
        max_p = probs.max()

        from sklearn.metrics import recall_score, precision_score
        for thr in (0.30, 0.40, 0.50):
            preds = (probs > thr).astype(int)
            acc = accuracy_score(valid_data['target_draw'], preds)
            rec = recall_score(valid_data['target_draw'], preds, zero_division=0)
            prec = precision_score(valid_data['target_draw'], preds, zero_division=0)
            print(f"Draw Model thr={thr:.2f} | Acc: {acc:.4f} | Recall: {rec:.4f} | Precision: {prec:.4f}")
        print(f"Draw Model calibration | actual draw rate: {actual_draw_rate:.4f} | "
              f"mean predicted P(draw): {mean_p:.4f} | max P(draw): {max_p:.4f}")
        if mean_p > 1.5 * actual_draw_rate:
            print(f"  ⚠ WARN: model is overcalibrating draws — mean P(draw) is "
                  f"{mean_p/actual_draw_rate:.2f}x the actual rate. Consider lowering scale_pos_weight.")
        elif mean_p < 0.7 * actual_draw_rate:
            print(f"  ⚠ WARN: model is undercalibrating draws — mean P(draw) is "
                  f"{mean_p/actual_draw_rate:.2f}x the actual rate. Consider raising scale_pos_weight.")
        
        if self.draw_family == 'xgboost':
            artifact = "xgb_model_draw.json"
            model.save_model(f"models/{artifact}")
        else:
            import joblib
            artifact = f"sk_model_draw_{self.draw_family}.joblib"
            joblib.dump(model, f"models/{artifact}")
        save_meta('draw', self.draw_family, artifact)
        with open("models/features_draw.json", "w") as f:
            json.dump(features, f)
        print(f"Saved Draw model ({self.draw_family}).")

    def _training_rows(self, df, features, target):
        """Rows usable for fitting `features` against `target`.

        Requires every feature EXCEPT the h2h block, whose averages are
        legitimately NaN for a first-ever meeting between two teams.

        Currently inert, since the h2h columns are not in `common_features`
        (2026-09-18 — no measured edge). It is kept because it is the guard
        that makes re-adding them safe: when they were in the list, dropping
        on them cost 2,071 rows, 14.7% of the training set, and biased what
        remained toward long-established pairings. XGBoost learns a split
        direction for NaN natively, so the rows stay. The same exemption
        applies to any future feature that is legitimately missing for some
        rows rather than absent by accident.

        Mirrored by `dropna_on` in scripts/experiment_odds_free.oof_predictions,
        so the experiment scores the row set production would actually train on.
        """
        required = [f for f in features if f not in H2H_FEATURES]
        return df.dropna(subset=required + [target]).copy()

    def train_1x2(self, df):
        print(f"\n--- Training 1X2 Model (family={self.model_family}) ---")
        # The model family is resolved through the shared seam (model_registry).
        # The estimator carries its own params (XGBoost honours the tuned
        # best_params_1x2.json; linear/tree baselines carry their own pipeline),
        # so this method no longer assembles XGBoost params inline.
        spec = get_spec('1x2', self.model_family)

        # Ensure unique features
        features = ['B365H', 'B365D', 'B365A'] + self.common_features
        features = list(dict.fromkeys(features))
        # league_cat is an XGBoost-categorical column; families that can't
        # consume a category dtype (linear/tree baselines) drop it.
        if not spec.uses_categorical and 'league_cat' in features:
            features = [f for f in features if f != 'league_cat']

        df_train = self._training_rows(df, features, 'target_1x2')
        df_train = df_train.sort_values('date')

        # Enable Categorical for XGBoost (only when the family consumes it)
        if spec.uses_categorical and 'league_cat' in df_train.columns:
            df_train['league_cat'] = df_train['league_cat'].astype('category')

        # 1. Cross-Validation
        print("Running Time-Series Cross-Validation (5 Splits)...")
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = []
        recalls = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(df_train)):
            cv_train = df_train.iloc[train_idx]
            cv_test = df_train.iloc[test_idx]

            # Plain fit (no eval_set): the tuned params carry no
            # early_stopping_rounds, so the prior eval_set only logged a
            # metric — it never altered tree building. Dropping it keeps the
            # fitted model identical while unifying the fit contract across
            # families.
            model = spec.build()
            model.fit(cv_train[features], cv_train['target_1x2'])
            # Predictions
            preds_prob = model.predict_proba(cv_test[features])
            preds_class = model.predict(cv_test[features])
            
            # 1. Accuracy
            acc = accuracy_score(cv_test['target_1x2'], preds_class)
            accuracies.append(acc)
            
            # 2. Log Loss (Multiclass)
            from sklearn.metrics import log_loss
            ll = log_loss(cv_test['target_1x2'], preds_prob, labels=[0, 1, 2])
            
            # 3. Brier Score (Multiclass definition: mean squared error of probability vector)
            # sklearn brier_score_loss is binary only. We implement simple multiclass MSE.
            # Convert target to one-hot for MSE calculation
            y_true_onehot = np.eye(3)[cv_test['target_1x2'].astype(int)]
            brier_score = np.mean(np.sum((preds_prob - y_true_onehot)**2, axis=1))

            # 4. Calibration (Avg Confidence vs Accuracy)
            # We take max prob as confidence
            confidences = np.max(preds_prob, axis=1)
            is_correct = (preds_class == cv_test['target_1x2'])
            avg_conf = np.mean(confidences)
            avg_acc_conf = np.mean(is_correct)
            calibration_error = avg_conf - avg_acc_conf
            
            # 5. ROI Simulation (Flat betting on prediction)
            # Odds columns: B365H (index 0), B365D (index 1), B365A (index 2)
            odds_cols = ['B365H', 'B365D', 'B365A']
            # Get odds for the predicted class via advanced indexing.
            # preds_class must be integer here: XGBoost.predict returns int,
            # but sklearn classifiers return labels in the dtype of y, which is
            # float64 after the target's dropna — cast so the index is valid
            # for any family. (XGBoost path unchanged: astype(int) is a no-op.)
            rows = np.arange(len(preds_class))
            pred_odds = cv_test[odds_cols].values[rows, np.asarray(preds_class).astype(int)]
            
            # Profit logic: if correct, profit = odds - 1. If wrong, profit = -1.
            profits = np.where(is_correct, pred_odds - 1.0, -1.0)
            roi = (np.sum(profits) / len(profits)) * 100.0 if len(profits) > 0 else 0.0
            
            # Recall
            rec = recall_score(cv_test['target_1x2'], preds_class, average='weighted', zero_division=0)
            recalls.append(rec)

            print(f"Fold {fold+1} | Acc: {acc:.4f} | Recall: {rec:.4f} | LogLoss: {ll:.4f} | Brier: {brier_score:.4f} | ROI: {roi:.2f}% | CalibErr: {calibration_error:.4f}")
        
        print(f"Average CV Accuracy: {np.mean(accuracies):.4f} | Avg Recall: {np.mean(recalls):.4f}")
        
        # 2. Final Training on the same 95% head the pre-seam path used.
        # (The trailing 5% was only ever an eval_set for an early-stopping
        # rule that the tuned params don't enable, so the saved model was
        # always fit on this 95% slice. Fitting it without the inert eval_set
        # reproduces the identical model.)
        print("Retraining Final Model on Full Dataset (2020-Present)...")
        split_idx = int(len(df_train) * 0.95)
        final_train = df_train.iloc[:split_idx]

        final_model = spec.build()
        final_model.fit(final_train[features], final_train['target_1x2'])

        # Persistence. XGBoost keeps its native JSON path (what predict_matches
        # loads today). Non-XGBoost families dump alongside via joblib; wiring
        # the predictor to load them is step 3 of the estimator-seam plan.
        if self.model_family == 'xgboost':
            artifact = "xgb_model_1x2.json"
            final_model.save_model(f"models/{artifact}")
        else:
            import joblib
            artifact = f"sk_model_1x2_{self.model_family}.joblib"
            joblib.dump(final_model, f"models/{artifact}")
        # Record which family/artifact serves this head so predict_matches
        # loads the right one (legacy XGBoost JSON still works without it).
        save_meta('1x2', self.model_family, artifact)
        with open("models/features_1x2.json", "w") as f:
            json.dump(features, f)
        print(f"Saved final 1X2 model ({self.model_family}).")

    def train_ou(self, df):
        print(f"\n--- Training O/U 2.5 Model (Poisson, family={self.ou_family}) ---")
        # Resolved through the seam — the spec carries the Poisson objective
        # and tuned params (XGBoost) or its own pipeline (GLM baseline).
        spec = get_spec('ou', self.ou_family)

        features = ['B365H', 'B365D', 'B365A', 'H_form_ou', 'A_form_ou'] + self.common_features
        features = list(dict.fromkeys(features))
        if not spec.uses_categorical and 'league_cat' in features:
            features = [f for f in features if f != 'league_cat']

        # Target: Total Goals
        df['total_goals'] = df['FTHG'] + df['FTAG']

        df_train = self._training_rows(df, features, 'total_goals')
        df_train = df_train.sort_values('date')

        # Enable Categorical for XGBoost (only when the family consumes it)
        if spec.uses_categorical and 'league_cat' in df_train.columns:
            df_train['league_cat'] = df_train['league_cat'].astype('category')

        # 1. Cross-Validation
        print("Running Time-Series Cross-Validation (5 Splits)...")
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = []
        recalls = []
        log_losses = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(df_train)):
            cv_train = df_train.iloc[train_idx]
            cv_test = df_train.iloc[test_idx]

            # Plain fit (no eval_set) — the tuned OU params carry no early
            # stopping, so the prior eval_set never altered the fitted model.
            model = spec.build()
            model.fit(cv_train[features], cv_train['total_goals'])

            preds_lambda = model.predict(cv_test[features])
            
            # --- Convert Lambda to Probability (Poisson) ---
            # P(Over 2.5) = 1 - P(X <= 2)
            # P(X=k) = e^-lam * lam^k / k!
            # P(X<=2) = e^-lam * (1 + lam + lam^2/2)
            
            prob_le_2 = np.exp(-preds_lambda) * (1 + preds_lambda + (preds_lambda**2 / 2))
            prob_over = 1.0 - prob_le_2
            
            # Clip for safety
            prob_over = np.clip(prob_over, 0.001, 0.999)
            
            # Evaluation against Binary Targets (for consistency)
            binary_target = (cv_test['total_goals'] > 2.5).astype(int)
            preds_class = (prob_over > 0.5).astype(int)
            
            # 1. Accuracy
            acc = accuracy_score(binary_target, preds_class)
            accuracies.append(acc)
            
            # 2. Log Loss
            from sklearn.metrics import log_loss
            ll = log_loss(binary_target, prob_over)
            log_losses.append(ll)
            
            # 3. Regression Error (RMSE)
            rmse = np.sqrt(np.mean((cv_test['total_goals'] - preds_lambda)**2))

            # Recall (Binary because converting to Over/Under classes)
            rec = recall_score(binary_target, preds_class, zero_division=0)
            recalls.append(rec)

            print(f"Fold {fold+1} | Acc: {acc:.4f} | Recall: {rec:.4f} | LogLoss: {ll:.4f} | RMSE: {rmse:.4f}")
            
        print(f"Average CV Accuracy: {np.mean(accuracies):.4f} | Avg Recall: {np.mean(recalls):.4f} | Avg LogLoss: {np.mean(log_losses):.4f}")
        
        # 2. Final Training (same 95% head the pre-seam path used; the inert
        # eval_set is dropped, reproducing the identical model).
        print("Retraining Final Model on Full Dataset (2020-Present)...")
        split_idx = int(len(df_train) * 0.95)
        final_train = df_train.iloc[:split_idx]

        final_model = spec.build()
        final_model.fit(final_train[features], final_train['total_goals'])

        if self.ou_family == 'xgboost':
            artifact = "xgb_model_ou.json"
            final_model.save_model(f"models/{artifact}")
        else:
            import joblib
            artifact = f"sk_model_ou_{self.ou_family}.joblib"
            joblib.dump(final_model, f"models/{artifact}")
        save_meta('ou', self.ou_family, artifact)
        with open("models/features_ou.json", "w") as f:
            json.dump(features, f)
        print(f"Saved final O/U model ({self.ou_family}).")

if __name__ == "__main__":
    # Per-head family overrides for challenger experiments
    # (default = production xgboost for every head).
    trainer = ModelTrainer(
        "data_sets/MatchHistory",
        model_family=os.environ.get("MODEL_FAMILY_1X2", "xgboost"),
        ou_family=os.environ.get("MODEL_FAMILY_OU", "xgboost"),
        draw_family=os.environ.get("MODEL_FAMILY_DRAW", "xgboost"),
    )
    data = trainer.prepare_data()
    trainer.train_1x2(data)
    trainer.train_draw(data)
    trainer.train_ou(data)
