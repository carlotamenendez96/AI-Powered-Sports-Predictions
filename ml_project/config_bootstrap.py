"""Seed gitignored per-deployment state files from their committed templates.

Two files under `data_sets/` are deployment state rather than source, so they
are gitignored and shipped as `<name>.template.json`:

  betting_config.json   lane bankroll balances, rewritten by the running web UI
                        on every settlement and cashout. Tracking it meant every
                        `git status` was dirty and every merge risked writing a
                        stale balance over live money — on 2026-09-18 a merge to
                        main would have replaced live bankrolls
                        (775.99/1075.64/801.36) with a branch snapshot
                        (698.40/912.69/625.03) while the server held the file.
  team_mappings.json    Flashscore -> football-data name mappings, appended by
                        EntityResolver whenever it resolves a new club.

Both loaders already tolerate a missing file (zero bankrolls / empty mappings),
so seeding is about not silently degrading a fresh clone rather than about
avoiding a crash.

The mappings template is a FULL snapshot, not a stub — 553 entries of
accumulated resolution, including hand-corrected mojibake spellings. A clone
gets all of it. The trade-off: mappings learned after the template was written
are no longer shared between checkouts. Refresh it deliberately with
`--refresh-mappings` when enough has accumulated to be worth committing.

Called on import by `web_ui/sports_config` and `ml_project/entity_resolver`, and
runnable directly:

    python3 -m ml_project.config_bootstrap            # seed anything missing
    python3 -m ml_project.config_bootstrap --refresh-mappings
"""
import argparse
import json
import os
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_SETS = os.path.join(PROJECT_ROOT, 'data_sets')

# live filename -> template filename
SEEDED = {
    'betting_config.json': 'betting_config.template.json',
    'team_mappings.json': 'team_mappings.template.json',
    # URL list for scrapy crawl standings (bin/update_leagues_data.sh).
    'standings_form_flashscore_direct_links.csv':
        'standings_form_flashscore_direct_links.template.csv',
}


def seed_file(live_name, template_name, data_dir=DATA_SETS, quiet=True):
    """Copy template -> live when live is absent. Returns True if it seeded.

    Never overwrites an existing file: the live copy is the source of truth
    once it exists, and clobbering it would destroy bankroll state.
    """
    live = os.path.join(data_dir, live_name)
    template = os.path.join(data_dir, template_name)
    if os.path.exists(live) or not os.path.exists(template):
        return False
    try:
        os.makedirs(data_dir, exist_ok=True)
        shutil.copyfile(template, live)
    except OSError:
        return False                      # read-only checkout: degrade, don't crash
    if not quiet:
        print(f'Seeded {live_name} from {template_name}')
    return True


def seed_all(data_dir=DATA_SETS, quiet=True):
    return {k: seed_file(k, v, data_dir, quiet) for k, v in SEEDED.items()}


def refresh_mappings(data_dir=DATA_SETS):
    """Copy the live mappings back over the template, so a commit shares them.

    Only the mappings — a bankroll balance must never be promoted to template.
    """
    live = os.path.join(data_dir, 'team_mappings.json')
    template = os.path.join(data_dir, 'team_mappings.template.json')
    if not os.path.exists(live):
        print('No live team_mappings.json to refresh from.')
        return False
    with open(live, encoding='utf-8') as f:
        data = json.load(f)
    with open(template, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write('\n')
    print(f'Refreshed team_mappings.template.json ({len(data)} entries). '
          f'Commit it to share them.')
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--refresh-mappings', action='store_true',
                    help='Promote the live team_mappings.json into the template.')
    args = ap.parse_args()
    if args.refresh_mappings:
        refresh_mappings()
        return
    seeded = seed_all(quiet=False)
    if not any(seeded.values()):
        print('Nothing to seed — all deployment files already present.')


if __name__ == '__main__':
    main()
