# Evaluation Report

- **Run:** 2026-10-09 20:54 UTC
- **Generator:** `fake-extractive` · **Judge:** `fake-extractive`
- **Embedder:** `hashing` · **Retrieval:** `hybrid` · **k:** 6
- **Cases:** 34 (29 answerable, 5 unanswerable)

## Headline metrics

| Metric | Value |
|---|---|
| Retrieval recall@6 | 100.0% |
| Retrieval MRR | 0.97 |
| Answer rate (answerable questions) | 65.5% |
| Abstention accuracy (unanswerable questions) | 40.0% |
| Citation precision | 94.7% |
| Unsupported sentence rate | 0.0% |
| Judge faithfulness (1-5) | 5.00 |
| Judge correctness (1-5) | 3.58 |
| Correct (score >= 4) | 52.6% |
| Latency p50 / p95 | 0.62 ms / 7.43 ms |
| Tokens in / out (generation) | 0 / 0 |

## Retrieval ablation

Same questions, three retrieval strategies. No LLM involved.

| Mode | Recall@1 | Recall@3 | Recall@6 | MRR |
|---|---|---|---|---|
| bm25 | 94.8% | 100.0% | 100.0% | 0.98 |
| dense | 84.5% | 96.6% | 100.0% | 0.93 |
| hybrid | 91.4% | 100.0% | 100.0% | 0.97 |

## By question type

| Type | N | Recall@6 | Correctness |
|---|---|---|---|
| lookup | 18 | 100.0% | 3.60 |
| multi_hop | 2 | 100.0% | 4.00 |
| paraphrase | 9 | 100.0% | 3.00 |
| unanswerable | 5 | n/a | n/a |

## Failures (22)

**q01** (lookup): How many PTO days do full-time employees get per year?  
*Why it failed:* correctness 3/5  
*Answer:* Full-time employees accrue 20 days of paid time off (PTO) per calendar year, accrued evenly each pay period. [1]  
*Reference:* 20 days per year, rising to 25 days after five years of continuous service.

**q02** (lookup): How many unused vacation days can I roll over, and when do they expire?  
*Why it failed:* correctness 3/5  
*Answer:* Up to 5 unused PTO days may be carried over into the next calendar year. [1]  
*Reference:* Up to 5 unused PTO days carry over, and they expire on March 31 if unused.

**q03** (paraphrase): When do I need a note from a doctor for being out sick?  
*Why it failed:* abstained on an answerable question  
*Answer:* I couldn't find an answer to that in the available documents.  
*Reference:* Only for absences longer than three consecutive working days.

**q04** (lookup): How much paid parental leave do non-birthing parents get?  
*Why it failed:* correctness 3/5  
*Answer:* Birthing parents receive 16 weeks of fully paid parental leave. [1]  
*Reference:* 12 weeks of fully paid leave, including adoptive and foster parents.

**q05** (paraphrase): Does the office close over the holidays at the end of the year?  
*Why it failed:* abstained on an answerable question  
*Answer:* I couldn't find an answer to that in the available documents.  
*Reference:* Yes, there is a winter shutdown from December 24 to January 1 that does not count against PTO.

**q07** (lookup): Who has to approve a $6,000 expense?  
*Why it failed:* abstained on an answerable question  
*Answer:* I couldn't find an answer to that in the available documents.  
*Reference:* A director, because expenses between $2,500 and $10,000 require director approval.

**q08** (lookup): What is the nightly hotel limit in London?  
*Why it failed:* abstained on an answerable question  
*Answer:* I couldn't find an answer to that in the available documents.  
*Reference:* London is a high-cost city, so hotels are reimbursed up to $350 per night.

**q10** (lookup): What is the daily meal allowance when travelling?  
*Why it failed:* correctness 3/5  
*Answer:* When travelling, meals are covered up to $75 per day, including tips. [1]  
*Reference:* Up to $75 per day including tips; alcohol is not reimbursable except at pre-approved client dinners.

**q11** (paraphrase): How much money do I get to set up my workspace at home?  
*Why it failed:* abstained on an answerable question  
*Answer:* I couldn't find an answer to that in the available documents.  
*Reference:* A one-time $500 home office stipend, available during the first 90 days.

**q13** (lookup): What is the minimum password length?  
*Why it failed:* abstained on an answerable question  
*Answer:* I couldn't find an answer to that in the available documents.  
*Reference:* Passwords must be at least 14 characters long.

**q14** (paraphrase): Can I use text message codes for two-factor login?  
*Why it failed:* abstained on an answerable question  
*Answer:* I couldn't find an answer to that in the available documents.  
*Reference:* No, SMS codes are not an accepted second factor.

**q15** (paraphrase): I left my work laptop on a train. What should I do?  
*Why it failed:* abstained on an answerable question  
*Answer:* I couldn't find an answer to that in the available documents.  
*Reference:* Report it to the security team within 1 hour via #security-help or security@northwind.example; the device will be remotely wiped.

**q16** (lookup): How is customer robot telemetry classified?  
*Why it failed:* correctness 1/5  
*Answer:* Customer telemetry from robots is classified as Restricted. [1]  
*Reference:* As Restricted data, which must never be pasted into external tools such as public AI chatbots.

**q17** (lookup): How long does the Kestrel X2 battery last?  
*Why it failed:* correctness 1/5  
*Answer:* The Kestrel X2 is Northwind Robotics' autonomous warehouse inventory robot. [1]  
*Reference:* Up to 10 hours of continuous operation; a full charge takes 90 minutes.

**q18** (paraphrase): How high can the robot scan shelves?  
*Why it failed:* abstained on an answerable question  
*Answer:* I couldn't find an answer to that in the available documents.  
*Reference:* Up to 9 metres using its telescoping mast.

**q23** (paraphrase): A customer says a robot almost hit a worker. Who needs to be told and how fast?  
*Why it failed:* abstained on an answerable question  
*Answer:* I couldn't find an answer to that in the available documents.  
*Reference:* The Head of Safety must be notified within 1 hour, and the site's fleet is put in safe mode.

**q24** (lookup): When is a postmortem due?  
*Why it failed:* correctness 3/5  
*Answer:* The postmortem draft is due within 5 business days of resolution and is reviewed at the weekly reliability meeting. [1]  
*Reference:* Within 5 business days of resolution, for every SEV1 and SEV2 incident.

**q26** (paraphrase): Which days am I expected in the office?  
*Why it failed:* correctness 2/5  
*Answer:* Each employee has a one-time home office stipend of $500, available during their first 90 days, for items such as a chair, desk or monitor. [1]  
*Reference:* Tuesdays and Thursdays, for employees within 50 miles of an office.

**q28** (multi_hop): What do engineers need before they get production access?  
*Why it failed:* correctness 3/5  
*Answer:* Hardware security keys are required for engineers with production access and for anyone in Finance; everyone else may use an authenticator app. [1]  
*Reference:* They must complete security training and receive a hardware security key, which is required for engineers with production access.

**q31** (unanswerable): What is the Kestrel X2's maximum payload capacity?  
*Why it failed:* answered an unanswerable question  
*Answer:* The Kestrel X2 has a top speed of 2.0 metres per second and a maximum scanning height of 9 metres using its telescoping mast. [1]  
*Reference:* (should abstain)

**q32** (unanswerable): Who is the CEO of Northwind Robotics?  
*Why it failed:* answered an unanswerable question  
*Answer:* This policy applies to all full-time employees of Northwind Robotics. [1]  
*Reference:* (should abstain)

**q33** (unanswerable): How many sabbatical weeks do employees get after ten years?  
*Why it failed:* answered an unanswerable question  
*Answer:* All employees complete security awareness training within their first two weeks and then once a year. [1]  
*Reference:* (should abstain)

