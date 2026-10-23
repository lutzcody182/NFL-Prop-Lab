# Phase 9 pre-registration: role changes, game scores, live market lines

Written 2026-10-23, before any of these was built or scored.

## A. Snap share and role changes

Volume (targets, carries) uses a player's recent share of team volume, which reacts slowly when a role changes.
Snap counts react faster. **Role signal** r = short-term offensive snap share (EWMA, half-life 2 games) divided by
long-term snap share (half-life 8 games), both over the player's earlier games only, shrunk toward 1 with 2 pseudo-games
at the long-term share. The volume share becomes share × r^γ, before teammates' shares are rescaled to the team total.

- γ tried: 0.5 and 1.0. Markets: WR/TE receptions and yards, RB rush attempts, rush yards, receptions, receiving yards,
  anytime TD (the volume-driven markets).
- **Development:** walk-forward 2019–2024, survives if the mean CRPS change (Brier for anytime TD) has a 95% game-bootstrap
  interval below zero.
- **Confirmation and adoption (as Phase 6):** below zero on 2025 and on 2026 weeks 1–6, and the pooled interval below zero.
- Snap counts are published the day after a game (known_at = kickoff + 1 day); only earlier games are used.

## B. Game score simulation

- **Model S:** for each game, the 200 games from the previous five seasons whose home and away implied points (from
  the closing spread and total) are nearest; their actual final scores, used as they are, are the simulated outcomes.
  This keeps real scoring patterns (3s, 7s, ties) and the link between the two teams' scores.
- **Baseline N:** home margin ~ Normal(spread, 13.5) and total ~ Normal(total, 13.5), independent.
- **Scored once on 2025** (and reported on 2026 weeks 1–6): Brier score for home win, home cover and over the total,
  and CRPS for margin and total. **S is used for game outcomes on the dashboard** if its mean Brier over the three events
  is lower than N's with a 95% game-bootstrap interval below zero; otherwise N is shown. Either way it is descriptive:
  inputs are the betting lines, so this is not an independent view of who wins.
- Live, the latest consensus spread and total are used. Player simulations are linked to the score by matching the
  difference in the two teams' simulated pass attempts to the simulated margin (trailing teams throw more), with the
  rank correlation fit on the previous three seasons.

## C. Live sportsbook lines (descriptive until a record exists)

- The weekly run takes one snapshot of player prop lines from The Odds API for games not yet started (Wednesday for
  Thursday games, Saturday for the rest), within a fixed request budget. The key is read from a GitHub secret.
- For each prop with a line, the dashboard shows the consensus line, the market's no-vig over chance (multiplicative
  method, median across books) and the model's chance at that same line.
- After games are final, a running record compares model and market Brier scores at those lines, per market. **No
  "edge" or value claim is made** until at least 8 weeks are recorded and the model's Brier is lower than the market's
  with a 95% interval below zero for that market (the Phase 2 rule, applied to live data).

## Deviations

(none yet)
