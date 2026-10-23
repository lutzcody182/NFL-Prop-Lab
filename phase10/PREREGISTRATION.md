# Phase 10 pre-registration: WR/TE role signal, quarterback changes, air yards

Written 2026-10-23, before any of these was built or scored. All three are tested inside the **Phase 1 WR/TE model**
(the live model for WR/TE receptions and receiving yards), against that model as locked.

## Groups (all features use earlier games only)

1. **Role (port of Phase 9A).** Target share × role_r^γ before teammates' shares are rescaled, with role_r exactly as in
   Phase 9 (snap share, half-life 2 vs 8 games). γ = 1, as chosen in Phase 9; not re-tuned.
2. **Quarterback change.** Each QB's quality as of a game: completion rate and yards per completion over his earlier
   games (half-life 16 games), shrunk toward the league QB average of the prior seasons (150 pseudo attempts / 90 pseudo
   completions). The projected starter is the active QB with the highest recent share of team pass attempts. The team's
   recent quarterback context is the average quality of its actual starters (most attempts) over its previous games
   (half-life 4 games). Receivers' catch rate × (starter completion rate / team context)^γ and yards per catch ×
   (starter yards per completion / team context)^γ. γ tried: 0.5, 1.0. Live, the projected starter skips a QB listed
   Out or Doubtful on the final injury report when that report is available (QB only; the rest of the lineup rule is
   unchanged).
3. **Air yards.** A receiver's depth of target (air yards per target over earlier games, half-life 8, shrunk toward the
   position average with 20 pseudo targets) implies an expected yards per catch, from a league fit on the prior seasons
   (receptions-weighted, yards per catch on depth of target). Yards per catch × (implied / his shrunk actual yards per
   catch)^γ: receivers whose yards have lagged their target depth are expected to catch up, and vice versa. γ tried:
   0.25, 0.5.

## Rules (same as Phases 6 and 9)

- **Development:** walk-forward 2019–2024, both WR/TE markets; a group survives if its better γ improves mean CRPS with a
  95% game-bootstrap interval below zero. Survivors are combined; any whose removal no longer hurts is dropped.
- **Confirmation and adoption:** the combined survivors vs the locked model on 2025 and on 2026 weeks 1–6: adopted if
  both changes are below zero and the pooled interval is below zero. The WR/TE yards "pass" status is then re-checked on
  2025 with the same three rules as Phase 1, and reported.
- Adopted changes go into live WR/TE projections from the next frozen week, recorded as a new model version.

## Deviations

(none yet)
