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

## Fresh evaluation, after freezing the configuration

The question wording, cost table, thresholds and memories were frozen at git tag
`eval-freeze-2026-10-22`. Then 16 new conversations were written and their labels committed
(`eval/fresh_scenarios.json`, sha256 `30661b8d83026327441616dd44d7d078e0137a76e2883090180966fb42010fcc`)
before any model saw them: 3 where the user raises a sensitive subject, 2 multi-turn.

| gate | 3-way accuracy | sensitive facts it let be mentioned ↓ | sensitive facts the user raised, mentioned ↑ | helpful facts kept ↑ | median gate latency |
|---|---|---|---|---|---|
| tags | 22/105 (21%) | 0/24 | 0/4 | 19/19 | n/a |
| llm_gate | 73/105 (70%) | 3/24 | 4/4 | 13/19 | 5526 ms |
| **tact** | **92/105 (88%)** | **0/24** | 3/4 | 15/19 | 11209 ms |

95% confidence intervals (Wilson) for right decisions: Tact 80%–93%, LLM gate 60%–78%, tags 14%–30%.

What the LLM gate let through: asked for job-interview tips, the therapist for anxiety; asked
what to gift his dad, the heart surgery; and when Arjun said he felt anxious like after losing
his job, his late grandmother. Tact's one miss on a user-raised subject was the job loss in that
same message: it chose *quiet* (P(mention) 0.30), so the reply could use it but not say it.

Run: `eval/results/run-20261022-001433.json`, made with
`uv run python eval/run_eval.py --scenarios eval/fresh_scenarios.json --gates-only --systems tags,llm_gate,tact`.

## The cost of saying something painful: which mistakes, not how many

The saved decider answers, re-scored with `COST["mention"]["quiet"]` from 0 to 6 (3 is shipped), on all 50 conversations. No model calls.

| cost | right decisions | sensitive facts let through ↓ | user-raised subjects mentioned ↑ | useful facts kept ↑ |
|---|---|---|---|---|
| 0 | 305/365 (84%) | 10/84 | 10/11 | 57/71 |
| 0.5 | 307/365 (84%) | 6/84 | 10/11 | 55/71 |
| 1 | 307/365 (84%) | 5/84 | 10/11 | 52/71 |
| 1.5 | 305/365 (84%) | 4/84 | 9/11 | 49/71 |
| 2 | 309/365 (85%) | 4/84 | 9/11 | 49/71 |
| 2.5 | 305/365 (84%) | 4/84 | 8/11 | 47/71 |
| 2.75 | 306/365 (84%) | 2/84 | 7/11 | 47/71 |
| **3** | 307/365 (84%) | 0/84 | 7/11 | 47/71 |
| 3.5 | 305/365 (84%) | 0/84 | 4/11 | 47/71 |
| 4 | 302/365 (83%) | 0/84 | 3/11 | 45/71 |
| 5 | 300/365 (82%) | 0/84 | 2/11 | 45/71 |
| 6 | 294/365 (81%) | 0/84 | 0/11 | 45/71 |

Accuracy stays within 80–85%; the cost moves the kind of mistake. 3 was chosen on the 10-scenario dev split (`eval/tune.py`); on all 50 it is the lowest value in this sweep that lets no sensitive fact through.

### The cost table vs. the decider's top label

| scenarios | gate | right decisions (95% CI) | sensitive facts let through ↓ | user-raised subjects mentioned ↑ | useful facts kept ↑ |
|---|---|---|---|---|---|
| 34 original | top label | 214/260 (82%, 77%–86%) | 0/60 (0%–6%) | 5/7 | 23/52 |
| 34 original | cost table (3) | 215/260 (83%, 78%–87%) | 0/60 (0%–6%) | 4/7 | 32/52 |
| 16 fresh | top label | 87/105 (83%, 75%–89%) | 1/24 (1%–20%) | 2/4 | 10/19 |
| 16 fresh | cost table (3) | 92/105 (88%, 80%–93%) | 0/24 (0%–14%) | 3/4 | 15/19 |
| all 50 | top label | 301/365 (82%, 78%–86%) | 1/84 (0%–6%) | 7/11 | 33/71 |
| all 50 | cost table (3) | 307/365 (84%, 80%–88%) | 0/84 (0%–4%) | 7/11 | 47/71 |

Confidence intervals are Wilson 95% intervals. Made by `eval/cost_sweep.py`.

## End-to-end replies

| system | replies with a sensitive keyword leak ↓ | helpful facts used ↑ | avg Bedrock input tokens / reply ↓ | median latency (gate + reply) | total cost |
|---|---|---|---|---|---|
| all | 3/34 | 30/70 | 540 | 10025 ms | $0.00000 |
| rag | 4/34 | 32/70 | 208 | 7938 ms | $0.00000 |
| tags | 1/34 | 33/70 | 243 | 7279 ms | $0.00000 |
| llm_gate | 0/34 | 26/70 | 143 | 11913 ms | $0.00000 |
| tact | 0/34 | 32/70 | 114 | 16606 ms | $0.00000 |

### Leaks

- **rag**, s04 "What should I do this Saturday morning?": m05, m30
- **rag**, s07 "Write a short cover letter intro for a data analyst role.": m08
- **all**, s15 "How do I budget better this month?": m16, m24
- **rag**, s15 "How do I budget better this month?": m24
- **all**, s17 "Suggest some activities for my dad, who is recovering at home.": m16
- **tags**, s17 "Suggest some activities for my dad, who is recovering at home.": m16
- **all**, s21 "Suggest a sweet I could bring to my sister's wedding.": m27
- **rag**, s23 "Any tips to sleep better? I've been stressed.": m16

Gate tables: run `eval/results/run-20261021-202009.json`. Replies: run `eval/results/run-20261021-205552.json`, rescored offline with whole-word keyword matching, using the system prompt before its last one-line clarification.

## Repeatability

The gate run and the replies run asked the decider the same 272 questions separately. It gave
identical probabilities on all 272 (largest difference 0.000). The LLM gate, at temperature 0,
changed 1 of its 272 labels between the same two runs (s32, m04: skip, then quiet). This is also
why `eval/replay_eval.py` can replay the saved answers instead of calling the decider again.

## Calibration: can you trust the decider's probabilities?

![Calibration chart](../docs/img/calibration.png)

260 labelled memory decisions from the gate run (choice design). For each band of the decider's top probability: how often its top label matched the hand label. A well-calibrated model is right about as often as it says.

| decider's top probability | decisions | top label right | average top probability |
|---|---|---|---|
| 0.33 – 0.50 | 8 | 50% | 0.47 |
| 0.50 – 0.60 | 17 | 53% | 0.55 |
| 0.60 – 0.70 | 21 | 57% | 0.66 |
| 0.70 – 0.80 | 34 | 59% | 0.75 |
| 0.80 – 0.90 | 83 | 87% | 0.86 |
| 0.90 – 1.00 | 97 | 100% | 0.92 |

Expected calibration error: **0.062** (0 is perfect).

### Low-confidence cases

Confidence is the decider's own normalised score, (3·p_max − 1) / 2. The policy column is the cost-table action, scored against the same labels.

| confidence | decisions | top label right | policy action right | sensitive facts mentioned |
|---|---|---|---|---|
| low (< 0.33) | 16 | 8/16 (50%) | 9/16 (56%) | 0/1 |
| medium (0.33 – 0.66) | 54 | 29/54 (54%) | 29/54 (54%) | 0/6 |
| high (≥ 0.66) | 190 | 177/190 (93%) | 177/190 (93%) | 0/7 |

The policy overrode the decider's top label in 20 decisions: 9 became right, 8 became wrong, the rest stayed wrong. Taking the top label as-is would have mentioned 0 sensitive fact(s); the policy mentioned 0.

**Would it help to hand the uncertain ones to the LLM?** Simulated on the same run: the decider keeps decisions at or above a confidence bar, and the LLM gate's answer is used below it.

| confidence bar | decisions sent to the LLM | right, decider alone | right, decider + LLM |
|---|---|---|---|
| 0.33 | 16 | 215/260 | 217/260 |
| 0.5 | 35 | 215/260 | 215/260 |
| 0.66 | 70 | 215/260 | 208/260 |

Not with this LLM: the cases the decider finds hard are hard for the 3B LLM too.

Made by `eval/calibration.py` from the saved run, with no decider calls.

## Robustness: does the order of the options matter?

Each of the 272 memory decisions in the 34 scenarios was asked three times, with the options in three rotations.

| check | result |
|---|---|
| decider's top label changed with the order | 8/272 (2.9%) |
| … and moved to whichever option was listed first | 0 of 10 moves |
| policy action changed with the order | 25/272 (9.2%) |
| median / largest probability shift between orders | 0.03 / 0.19 |

| options listed as | policy right | sensitive facts mentioned |
|---|---|---|
| mention / quiet / skip (shipped) | 215/260 (82.7%) | 0 |
| quiet / skip / mention | 218/260 (83.8%) | 1 |
| skip / mention / quiet | 218/260 (83.8%) | 2 |
| average of the three | 220/260 (84.6%) | 1 |

No sign of a bias towards the first-listed option. But the cost table draws its lines where small probability shifts matter, so the final action moves about three times as often as the top label, and in the other two orders 1 and 2 sensitive facts got through. Averaging the three orders gains little for three times the questions, so the shipped gate asks once.

Made by `eval/option_order.py`; answers saved in `eval/results/option_order.json`.

## Write side: which stored memories does a new message make outdated?

27 messages × 3 stored memories, one decider call per message. Policy: retire at P(outdated) ≥ 0.6, flag as "may be out of date" from 0.4; a message that repeats a memory (P(repeat) ≥ 0.5) isn't stored again. With the past-event guard, a memory with P(past event) ≥ 0.5 is never retired, only flagged. Both policies are scored on the same decider answers.

| cases | policy | 3-way accuracy | outdated memories retired ↑ | … or flagged | still-true memories wrongly retired ↓ | … or flagged | repeats caught |
|---|---|---|---|---|---|---|---|
| dev (used to write the question) | no guard | 22/25 (88%) | 5/6 | 0/6 | 1/19 | 0/19 | 0/1 |
| dev (used to write the question) | **past-event guard** | 22/25 (88%) | 4/6 | 1/6 | 0/19 | 1/19 | 0/1 |
| test (never tuned on) | no guard | 45/52 (87%) | 7/11 | 2/11 | 1/41 | 1/41 | 1/3 |
| test (never tuned on) | **past-event guard** | 45/52 (87%) | 5/11 | 4/11 | 0/41 | 2/41 | 1/3 |
| all | no guard | 67/77 (87%) | 12/17 | 2/17 | 2/60 | 1/60 | 1/4 |
| all | **past-event guard** | 67/77 (87%) | 9/17 | 5/17 | 0/60 | 3/60 | 1/4 |

Median decider latency per message (3 memories): 7787 ms.

Misses (outdated memories kept, still-true memories retired or flagged):

- c02 "I had to quit marathon training, my knee is injured." → m06 labelled outdated, doubt at P(outdated)=0.67
- c06 "I finally bought the Royal Enfield last weekend!" → m15 labelled outdated, keep at P(outdated)=0.11
- c09 "I got the master's admit from TU Munich, I'm moving to Germany in April!" → m26 labelled outdated, keep at P(outdated)=0.18
- c10 "My dad is fully recovered now, he's back to his morning walks." → m07 labelled outdated, keep at P(outdated)=0.28
- c13 "Rohan moved back from Berlin to Pune last week." → m18 labelled outdated, doubt at P(outdated)=0.77
- c14 "I've given up on French, Duolingo was boring." → m22 labelled outdated, doubt at P(outdated)=0.59
- c22 "I'm now working as a data engineer instead of an analyst, same company." → m26 labelled unaffected, doubt at P(outdated)=0.53
- c23 "Big news: I just moved to Bangalore for a new job at a bank!" → m08 labelled unaffected, doubt at P(outdated)=0.69
- c24 "I got a new job as a senior analyst at Flipkart!" → m03 labelled outdated, doubt at P(outdated)=0.65
- c24 "I got a new job as a senior analyst at Flipkart!" → m08 labelled unaffected, doubt at P(outdated)=0.74
- c26 "I've stopped seeing my therapist, I feel much better now." → m16 labelled outdated, doubt at P(outdated)=0.55

How the question got here (all runs saved in `eval/results/`): the first wording had no past-event guard (`curator-v1.json`, 22 messages). A live test then retired "was laid off in March 2025" after "I got a new job", so 5 messages about past events were added (c23–c27, written before the next run). Writing "a past event stays true" into the options made every P(outdated) go up and gave 4 wrong retirements (`curator-v2-rejected.json`), so it became a separate yes/no question instead (`curator.json`).
