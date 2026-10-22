# Tact against qwen2.5:7b-instruct

Run `2026-10-22 16:05`, provider `ollama`. The decider's answers and each scenario's 8-memory shortlist are replayed from saved runs (the decider is deterministic), so only these models were called. Plan: `eval/BEDROCK_PLAN.md`.

Inputs (sha256):

- `eval/scenarios.json` `ff29598588cf96dc815f7b72bf5000e5971fe3d09f524387541c4d4341880f39`
- `eval/results/run-20261021-202009.json` `20e431019961bd790133fb152be1c718c0dedfd5705fedc25c7862dad07d825c`
- `eval/fresh_scenarios.json` `30661b8d83026327441616dd44d7d078e0137a76e2883090180966fb42010fcc`
- `eval/results/run-20261022-001433.json` `8c8d67628ebc2462c7546defa4c1ee0b55589ab7539f1b3d119d36a74dc87669`

## Gate decisions vs. hand labels (original scenarios)

Scored on the prefiltered candidates; memories labelled "either" are not scored.

| gate | 3-way accuracy | sensitive facts it let be mentioned ↓ | sensitive facts the user raised, mentioned ↑ | helpful facts kept ↑ | median gate latency | gate cost |
|---|---|---|---|---|---|---|
| tags | 59/260 (23%) | 0/60 | 0/7 | 52/52 | n/a | $0.00000 |
| Qwen2.5 3B (saved run) | 157/260 (60%) | 5/60 | 2/7 | 26/52 | 6470 ms | $0.00000 |
| qwen2.5:7b-instruct | 198/260 (76%) | 2/60 | 2/7 | 20/52 | 14309 ms | $0.03828 |
| tact | 215/260 (83%) | 0/60 | 4/7 | 32/52 | 10641 ms | $0.03588 |

## Gate decisions vs. hand labels (24 held-out original scenarios)

Scored on the prefiltered candidates; memories labelled "either" are not scored.

| gate | 3-way accuracy | sensitive facts it let be mentioned ↓ | sensitive facts the user raised, mentioned ↑ | helpful facts kept ↑ | median gate latency | gate cost |
|---|---|---|---|---|---|---|
| tags | 35/181 (19%) | 0/40 | 0/4 | 34/34 | n/a | $0.00000 |
| Qwen2.5 3B (saved run) | 110/181 (61%) | 5/40 | 2/4 | 17/34 | 6657 ms | $0.00000 |
| qwen2.5:7b-instruct | 143/181 (79%) | 1/40 | 0/4 | 15/34 | 14323 ms | $0.02839 |
| tact | 153/181 (85%) | 0/40 | 2/4 | 21/34 | 10722 ms | $0.02559 |

### Escalation (original): the decider keeps its confident decisions, a model takes the rest

| gate | decisions sent to the model | 3-way accuracy | sensitive facts let through ↓ | user-raised subjects mentioned ↑ | helpful facts kept ↑ | model cost per 1,000 turns |
|---|---|---|---|---|---|---|
| tact, below 0.33 to qwen2.5:7b-instruct | 17/272 | 212/260 (82%) | 0/60 | 3/7 | 27/52 | $0.43 |
| tact, below 0.5 to qwen2.5:7b-instruct | 38/272 | 209/260 (80%) | 1/60 | 2/7 | 20/52 | $0.78 |
| tact, below 0.66 to qwen2.5:7b-instruct | 76/272 | 205/260 (79%) | 2/60 | 2/7 | 19/52 | $1.11 |

## Gate decisions vs. hand labels (fresh scenarios)

Scored on the prefiltered candidates; memories labelled "either" are not scored.

| gate | 3-way accuracy | sensitive facts it let be mentioned ↓ | sensitive facts the user raised, mentioned ↑ | helpful facts kept ↑ | median gate latency | gate cost |
|---|---|---|---|---|---|---|
| tags | 22/105 (21%) | 0/24 | 0/4 | 19/19 | n/a | $0.00000 |
| Qwen2.5 3B (saved run) | 73/105 (70%) | 3/24 | 4/4 | 13/19 | 5526 ms | $0.00000 |
| qwen2.5:7b-instruct | 83/105 (79%) | 5/24 | 3/4 | 10/19 | 13298 ms | $0.01696 |
| tact | 92/105 (88%) | 0/24 | 3/4 | 15/19 | 11209 ms | $0.01789 |

### Escalation (fresh): the decider keeps its confident decisions, a model takes the rest

| gate | decisions sent to the model | 3-way accuracy | sensitive facts let through ↓ | user-raised subjects mentioned ↑ | helpful facts kept ↑ | model cost per 1,000 turns |
|---|---|---|---|---|---|---|
| tact, below 0.33 to qwen2.5:7b-instruct | 8/128 | 93/105 (89%) | 1/24 | 3/4 | 14/19 | $0.33 |
| tact, below 0.5 to qwen2.5:7b-instruct | 22/128 | 90/105 (86%) | 1/24 | 3/4 | 11/19 | $0.90 |
| tact, below 0.66 to qwen2.5:7b-instruct | 42/128 | 86/105 (82%) | 3/24 | 3/4 | 9/19 | $0.99 |

## What a gate decision costs, per 1,000 turns

| gate | median time per turn | cost per 1,000 turns | how |
|---|---|---|---|
| tact (decider on a c7i.2xlarge CPU) | 11.0 s | $1.09 | the instance's hourly price × the measured time, as if always busy |
| qwen2.5:7b-instruct | 14.3 s | $1.10 | the same instance's price × the measured time |

The decider's published speed on a GPU is about 115 ms per question; it ran on a CPU here (no GPU quota), and that is not measured.

Raw answers: `eval/results/ollama-20261022-160406.json`.
