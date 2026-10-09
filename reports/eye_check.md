# Eye check — 20 articles

Checked by hand on 2026-10-09 against `data/raw/law.pdf` and `data/processed/civil_code.json`.

For each article the number, Arabic text, and English text in the JSON match the PDF page. Repealed articles are still in the JSON and marked `is_repealed: true`.

| Article | Page | Repealed | Result | Note |
|---|---|---|---|---|
| 1 | 1 | no | pass | Opening article. Arabic and English match the PDF. |
| 6 | 1 | no | pass | Same first page as Article 1. |
| 10 | 2 | no | pass | Short article near the start. |
| 54 | 7 | yes | pass | Start of the first repealed block. Record kept. |
| 80 | 7 | yes | pass | End of articles 54–80. Record kept. |
| 81 | 7 | no | pass | First live article after that repeal. |
| 89 | 8 | no | pass | Still live. Text matches the page. |
| 147 | 16 | no | pass | Contract article. Text matches the page. |
| 200 | 24 | no | pass | Middle of the Code. |
| 389 | 53 | yes | pass | Start of the second repealed block. Record kept. |
| 417 | 53 | yes | pass | End of articles 389–417. Record kept. |
| 418 | 53 | no | pass | First live article after that repeal. |
| 452 | 59 | no | pass | Sale warranty. JSON keeps the text printed on the page. |
| 500 | 64 | no | pass | Middle of the Code. |
| 700 | 98 | no | pass | Later book. |
| 890 | 128 | no | pass | Later book. |
| 1000 | 144 | no | pass | Later book. |
| 1022 | 147 | no | pass | Article number and body are both present. |
| 1100 | 161 | no | pass | Near the end of the Code. |
| 1149 | 170 | no | pass | Last article. Text matches the last page. |

Result: 20 of 20 checked articles match. No article was deleted.
