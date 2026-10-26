# Output examples (public dev)

Correct = token F1 >= 0.5. Picked by question ID order within each bucket; F1 in parentheses.

## Closed-book vs Weighted hybrid RAG

### Weighted hybrid RAG right, Closed-book wrong (80 questions)

| id | topic | question | reference | Closed-book | Weighted hybrid RAG |
|---|---|---|---|---|---|
| 2 | Carnegie Museum of Art | What accession number does the Carnegie Museum of Art list for the painting “Odalisque Coi | 62.19.2 | 1978.12.1 (0.00) | 62.19.2 (1.00) |
| 3 | Carnegie Museum of Art | Which artist is credited for the Carnegie Museum of Art painting “Metamorphosis (Homage to | Manuel Rivera | Sol LeWitt (0.00) | Manuel Rivera (1.00) |
| 5 | CMU | What is the significance of the NavLab project at CMU? | The NavLab (Navigation Laboratory) project at CMU pioneered self-driving car research in t | Autonomous vehicle development (0.17) | The NavLab project at CMU is significant because it pioneered research in autonomous and s (0.58) |
| 7 | Carnegie Museum of Art | Which artist or school does the Carnegie Museum of Art list for “Summer in Havana” in its  | Corneille (Cornelius Guillaume Van Beverloo) | Clyfford Still (0.00) | Corneille (Cornelius Guillaume Van Beverloo) (1.00) |

### Closed-book right, Weighted hybrid RAG wrong (8 questions)

| id | topic | question | reference | Closed-book | Weighted hybrid RAG |
|---|---|---|---|---|---|
| 1 | Events/music/comedy | What open mic event occurs weekly on Mondays from 8:00-10:00 PM? | Monday Night Open Mic Comedy! | CMU Open Mic (0.50) | I don't know (0.00) |
| 19 | Pittsburgh local/history | What is Mattress Factory’s Instagram handle? | @mattressfactory | @mattressfactory (1.00) | I don't know (0.00) |
| 67 | Carnegie Museum of Art | How did the Carnegie Museum of Art acquire “Untitled (Lead Checks/Lead Chevron #3)”, accor | Gift | Gift of the artist (0.50) | I don't know (0.00) |
| 76 | Events/music/comedy | What distinction does the Farmers' Market Cooperative of East Liberty claim in western Pen | It is the oldest continuously operating market in western Pennsylvania. | First Farmers' Market in Western Pennsylvania (0.53) | Pittsburgh’s oldest continually-operating farmers’ market (0.29) |

### both wrong (65 questions)

| id | topic | question | reference | Closed-book | Weighted hybrid RAG |
|---|---|---|---|---|---|
| 4 | Events/music/comedy | What comedy show runs at Pittsburgh Improv starting October 22, 2026? | Steph Tolev runs at Pittsburgh Improv from October 22–24, 2026. | Unknown (0.00) | I don't know (0.00) |
| 6 | Restaurants/food | What type of cuisine does täkō in Pittsburgh serve? | täkō offers a fusion of Mexican and Asian-inspired tacos, blending unique flavors in a vib | Japanese (0.00) | Japanese fusion with Mexican influences (0.21) |
| 8 | Sports | When is the Pittsburgh Penguins game against the Florida Panthers scheduled in December? | December 10, 2026 | 2023-12-15 (0.00) | I don't know (0.00) |
| 11 | Pittsburgh local/history | What is the Syracuse Orange's strength of schedule ranking for 2026? | The Syracuse Orange rank 16th in the 2026 college-football strength-of-schedule ranking. | Unknown (0.00) | I don't know (0.00) |

## BM25 RAG vs Dense RAG

### Dense RAG right, BM25 RAG wrong (9 questions)

| id | topic | question | reference | BM25 RAG | Dense RAG |
|---|---|---|---|---|---|
| 5 | CMU | What is the significance of the NavLab project at CMU? | The NavLab (Navigation Laboratory) project at CMU pioneered self-driving car research in t | The NavLab project at CMU is significant because it pioneered research in autonomous and s (0.40) | The NavLab project at CMU is significant because it pioneered research in autonomous and s (0.53) |
| 26 | Carnegie Museum of Art | What accession number does the Carnegie Museum of Art list for the painting “By the River” | 1996.44 | I don't know (0.00) | 1996.44 (1.00) |
| 45 | Sports | What team will the Pittsburgh Penguins play against on November 5, 2026? | Boston Bruins | I don't know (0.00) | Boston (0.67) |
| 56 | Sports | What jersey number does the Steelers roster list for Gabriel Rubio? | #96 | 95 (0.00) | 96 (1.00) |

### BM25 RAG right, Dense RAG wrong (7 questions)

| id | topic | question | reference | BM25 RAG | Dense RAG |
|---|---|---|---|---|---|
| 33 | Events/music/comedy | What venue hosts the Pittsburgh Savoyards' Winter 2026 production of A Christmas Carol? | The Margaret Partee Performing Arts Center hosts the Pittsburgh Savoyards' Winter 2026 pro | Margaret Partee Performing Arts Center (1.00) | I don't know (0.00) |
| 51 | Carnegie Museum of Art | What credit line does the Carnegie Museum of Art record for “George Nevill, Lord Bergavenn | Bequest of Roy A. Hunt | Bequest of Roy A. Hunt (1.00) | I don't know (0.00) |
| 62 | Taxes/budget | How many uniformed police officers are budgeted in Pittsburgh? | The FY2026 Bureau of Police Position Summary budgets 800 Total Uniformed Police Officers. | 800 (1.00) | 850 (0.00) |
| 64 | Events/music/comedy | Which Pittsburgh-area arts event is scheduled at Arcade Comedy Theater at 6:00 PM on Decem | Sketch Comedy Lab | Sketch Comedy Lab (1.00) | I don't know (0.00) |

### both wrong (69 questions)

| id | topic | question | reference | BM25 RAG | Dense RAG |
|---|---|---|---|---|---|
| 1 | Events/music/comedy | What open mic event occurs weekly on Mondays from 8:00-10:00 PM? | Monday Night Open Mic Comedy! | I don't know (0.00) | I don't know (0.00) |
| 4 | Events/music/comedy | What comedy show runs at Pittsburgh Improv starting October 22, 2026? | Steph Tolev runs at Pittsburgh Improv from October 22–24, 2026. | I don't know (0.00) | I don't know (0.00) |
| 6 | Restaurants/food | What type of cuisine does täkō in Pittsburgh serve? | täkō offers a fusion of Mexican and Asian-inspired tacos, blending unique flavors in a vib | Japanese fusion with Mexican influences (0.21) | Japanese fusion with Mexican influences (0.21) |
| 8 | Sports | When is the Pittsburgh Penguins game against the Florida Panthers scheduled in December? | December 10, 2026 | I don't know (0.00) | I don't know (0.00) |

## Dense RAG vs Weighted hybrid RAG

### Weighted hybrid RAG right, Dense RAG wrong (6 questions)

| id | topic | question | reference | Dense RAG | Weighted hybrid RAG |
|---|---|---|---|---|---|
| 32 | Taxes/budget | What is Bethel Park Municipality's real estate tax rate? | Bethel Park Municipality's real estate tax rate is 4.13 mills. | 1.25 (0.00) | 4.13 (0.67) |
| 33 | Events/music/comedy | What venue hosts the Pittsburgh Savoyards' Winter 2026 production of A Christmas Carol? | The Margaret Partee Performing Arts Center hosts the Pittsburgh Savoyards' Winter 2026 pro | I don't know (0.00) | Margaret Partee Performing Arts Center (1.00) |
| 39 | Taxes/budget | What is Dormont Borough's real estate tax rate? | Dormont Borough's real estate tax rate is 12.4700 mills. | I don't know (0.00) | 12.47 (0.67) |
| 51 | Carnegie Museum of Art | What credit line does the Carnegie Museum of Art record for “George Nevill, Lord Bergavenn | Bequest of Roy A. Hunt | I don't know (0.00) | Bequest of Roy A. Hunt (1.00) |

### Dense RAG right, Weighted hybrid RAG wrong (3 questions)

| id | topic | question | reference | Dense RAG | Weighted hybrid RAG |
|---|---|---|---|---|---|
| 18 | Sports | In what year was Pittsburgh Pirates player Brandon Eisert born? | 1998 | 1998 (1.00) | 2000 (0.00) |
| 93 | Taxes/budget | When is first quarter Payroll Tax due in Pittsburgh? | The first-quarter Payroll Expense Tax report and payment are due May 31. | May 31 (1.00) | February 28 (0.00) |
| 150 | Events/music/comedy | When does Fruit Bats perform in Pittsburgh in 2026? | November 12, 2026. | November 12, 2026 (1.00) | I don't know (0.00) |

### both wrong (70 questions)

| id | topic | question | reference | Dense RAG | Weighted hybrid RAG |
|---|---|---|---|---|---|
| 1 | Events/music/comedy | What open mic event occurs weekly on Mondays from 8:00-10:00 PM? | Monday Night Open Mic Comedy! | I don't know (0.00) | I don't know (0.00) |
| 4 | Events/music/comedy | What comedy show runs at Pittsburgh Improv starting October 22, 2026? | Steph Tolev runs at Pittsburgh Improv from October 22–24, 2026. | I don't know (0.00) | I don't know (0.00) |
| 6 | Restaurants/food | What type of cuisine does täkō in Pittsburgh serve? | täkō offers a fusion of Mexican and Asian-inspired tacos, blending unique flavors in a vib | Japanese fusion with Mexican influences (0.21) | Japanese fusion with Mexican influences (0.21) |
| 8 | Sports | When is the Pittsburgh Penguins game against the Florida Panthers scheduled in December? | December 10, 2026 | I don't know (0.00) | I don't know (0.00) |
