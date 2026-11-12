# Phase 12 pre-registration: tracked picks at US-licensed sportsbooks

Written 2026-10-28, before any pick was logged. **These are tracked picks, not betting recommendations.** They are logged
and graded as if real money were on them. They become recommendations only if the graduation rule below is met.

## Start

The first picks are for **2026 week 9**, the first week frozen with the WR/TE model that passed every test (Phase 11).

## Books

Only state-licensed US sportsbooks count: DraftKings, FanDuel, BetMGM, Caesars, BetRivers, ESPN BET, Fanatics, Hard Rock
Bet, Bally Bet and betPARX (the list is in `proplab/p12/books.py`). Offshore books (Bovada, BetOnline, MyBookie, LowVig,
BetUS), sweepstakes apps and pick'em/DFS apps are excluded. The list can be narrowed to the books licensed in one
state; that changes which prices count from the next logged pick onward. Prices come from the existing line snapshots,
so this costs no extra API credits. Each pick stores the best licensed price and the book that offered it.

**State: Pennsylvania** (set 2026-10-28, before any pick). Prices count only from DraftKings, FanDuel, BetMGM, Caesars,
BetRivers, theScore Bet (formerly ESPN BET), Fanatics and betPARX. bet365, PlaySugarHouse and Mohegan are licensed in PA
but not in our odds feed.

## Which props

- Only markets that passed every pre-registered test: **WR/TE receptions and WR/TE receiving yards**. Others join only
  if they later pass.
- The line is the consensus line across books, and it must be offered by at least one licensed book with prices on both
  sides.
- The model's chances come from the **frozen ledger projection** for that week (the 100-point grid), never from a later
  re-run. For a whole-number line, the push chance is counted separately.

## Selection and units

- For each prop: EV per unit for each side = P(win) × best licensed decimal price + P(push) − 1. Keep the better side.
- Picked only if **EV ≥ 5%** and the model's over chance is within **0.20** of the market's no-vig over chance. Bigger
  disagreements usually mean the model is missing news, such as an injury or a role change.
- **Units:** 1 unit for EV 5–8%, 1.5 units for EV 8–12%, 2 units for EV of 12% or more. At most one pick per player per
  game and at most 3 picks per game (highest EV first).
- **When picks are logged:** once per game, at the last scheduled weekly run before kickoff (Wednesday for Thursday
  games, Saturday for Sunday and Monday games), using that game's latest line snapshot if it is under 48 hours old.
  Picks are written once and never changed afterward.

## Grading

- Graded 3 days after kickoff, so stat corrections are in (changed on 2026-11-09: see Deviation 1). A win pays units × (price − 1), a loss costs the units and a
  push returns them.
- **Void** (units returned) if the player was not active for the game, as sportsbooks do.
- The record shows wins, losses, pushes and voids, units staked, net units, ROI and a 95% interval for ROI (bootstrap
  over games).

## Graduation rule

The picks may be called recommendations only after **at least 8 weeks with graded picks**, and only if the 95% ROI
interval is entirely above zero. Until then the dashboard labels them as a tracked record, not advice. If the rule is
never met, they stay tracked picks.

## Version 2 (test, tracked alongside version 1): blended chances

Written 2026-11-11, before any week 10 pick was logged.

- **Why.** In weeks 7–9, WR/TE over chances at the book line were scored for 642 props. When the model and market
  disagreed, the result usually landed between them but closer to the model: Brier was lowest at about 75% model and
  25% market for both markets. Version 1's expected returns (median 23%) are far above any realistic edge, so its unit
  tiers act as flat 2-unit bets.
- **Rule.** Everything is the same as version 1 (markets, books, the 20-point gap filter, EV ≥ 5%, unit tiers, at most
  3 per game, when picks are logged, how they settle). The only change: the chance used for EV is 0.75 × model + 0.25 ×
  market no-vig chance. The push chance stays as the model has it, and the rest is split by the blend.
- **Judged only on weeks 10 onward**, because the 75/25 weight was chosen using weeks 7–9. It has its own record and the
  same graduation rule (8+ graded weeks, 95% ROI interval above zero). Version 1 continues unchanged. After 8 weeks
  both records are compared, but the version that graduates is decided by its own record, not by picking the better one.

## Deviations

1. **Faster settling (2026-11-09, after the first 3 picks were graded).** Picks now settle the way sportsbooks pay
   out: as soon as the game is final and the player's box-score line is loaded, without waiting for stat corrections.
   Extra weekly runs were added at about 5:41 am ET on Friday, Monday and Tuesday to settle the Thursday, Sunday and
   Monday games. A player with no stat line still waits 3 days: he either sat out (void) or played without a catch
   (0), and game-day inactives reach the data only after games. Selection, prices and units are unchanged, and picks
   are still logged at the Wednesday and Saturday runs. The 3 picks already graded keep their results (no stat
   correction had changed them).
