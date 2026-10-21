# Tact evaluation results

34 scenarios (6 where the user raises a sensitive subject themselves, 4 multi-turn), 30 memories, top-8 prefilter. Chat model: `tact-chat`. Prices assumed: $0.0/M input, $0.0/M output tokens.

## Gate decisions vs. hand labels (all scenarios)

Scored on the prefiltered candidates; memories labelled "either" are not scored.

| gate | 3-way accuracy | sensitive facts it let be mentioned ↓ | sensitive facts the user raised, mentioned ↑ | helpful facts kept ↑ | median gate latency | gate cost |
|---|---|---|---|---|---|---|
| tags | 59/260 (23%) | 0/60 | 0/7 | 52/52 | n/a | $0.00000 |
| llm_gate | 157/260 (60%) | 5/60 | 2/7 | 26/52 | 6470 ms | $0.00000 |
| tact | 215/260 (83%) | 0/60 | 4/7 | 32/52 | 10641 ms | $0.00000 |
| tact_split | 193/260 (74%) | 1/60 | 0/7 | 29/52 | 17844 ms | $0.00000 |

## Gate decisions vs. hand labels (24 held-out scenarios)

Scored on the prefiltered candidates; memories labelled "either" are not scored.

| gate | 3-way accuracy | sensitive facts it let be mentioned ↓ | sensitive facts the user raised, mentioned ↑ | helpful facts kept ↑ | median gate latency | gate cost |
|---|---|---|---|---|---|---|
| tags | 35/181 (19%) | 0/40 | 0/4 | 34/34 | n/a | $0.00000 |
| llm_gate | 110/181 (61%) | 5/40 | 2/4 | 17/34 | 6657 ms | $0.00000 |
| tact | 153/181 (85%) | 0/40 | 2/4 | 21/34 | 10722 ms | $0.00000 |
| tact_split | 139/181 (77%) | 0/40 | 0/4 | 21/34 | 17772 ms | $0.00000 |

