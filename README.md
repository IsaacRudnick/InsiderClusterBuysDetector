# Insider Cluster Buys Detector

This project finds cluster buys. A cluster buy occurs when two or more insiders
of one company buy stock in that company in a short period. The project also
does research to find out if cluster buys help an investor.

> **NOTE:** This project does not give financial advice. Backtests show that
> automatic purchases of cluster-buy stocks, without research, do not give
> good returns.

## Overview

### Insiders and cluster buys

An insider is an officer of a company, a director of the company, or a person
who owns more than 10% of its stock. U.S. law has a rule for insiders. Each
insider must tell the SEC about each trade in the stock of their company. The
insider must do this in two business days. Anyone can read these reports.

Insiders sell stock for many reasons. For example, an insider sells to pay
taxes or to buy a house. Insiders usually buy stock for one reason: they think
that the price will increase. Many investors think that a cluster buy is a
good sign for the stock.

### The two parts of the project

1. **The screener.** The screener reads the latest SEC reports and finds recent
   cluster buys. It shows them on a dashboard. You open the dashboard in a web
   browser.
2. **The research.** The research uses data for eight years, from 2018 to 2026.
   This data contains approximately 11,000 cluster buys. The research measures
   how the stock price changed after each cluster buy.

### Results

- **A cluster buy is not a signal to buy the stock.** Two weeks after a
  cluster buy, the stock did approximately as well as an index fund of small
  companies. After a longer period, the result was worse.
- **The model can identify the cluster buys with the most risk.** The model
  puts each cluster buy into one of four bands. In the lowest band,
  approximately 1 in 13 stocks lost more than 30% of their value in one month.
  This rate is approximately three times the rate for the other bands. In
  each year from 2020 to 2026, the rate was between 6.6% and 9.9%.
- **A higher score does not always show a better stock.** The top 10% of
  scores gave worse results than the band below them.
- **The research did not find a portfolio that does better than an index
  fund.** Some strategies looked better at first. Their advantage went away
  when the research included the cost of each trade. Their advantage also went
  away on years that the research did not use to make the strategy.
- **One sell rule gave better results in all tests.** This rule is a trailing
  stop. In 25 of 25 tests, the trailing stop gave better results than a sale
  after one month. The trailing stop does not use the model. Refer to the
  [glossary](#glossary).
- **All of the return values in this project are too high.** Approximately
  one third of the companies in the data do not trade now. Other companies
  bought some of them. Some changed their names, and some became bankrupt. The
  data does not contain prices for these companies. When the research adds
  them back, the average profit of a trade decreases to zero or less.

Use the dashboard to select the cluster buys to examine first, and the cluster
buys to avoid. Do not use the dashboard to select stocks to buy.

## How the screener operates

```
 SEC reports  ──▶  find cluster buys  ──▶  score each cluster buy  ──▶  dashboard
```

1. The screener downloads the insider reports from the SEC for the period that
   you select. The name of these reports is **Form 4**. The default period is
   30 days.
2. The screener keeps only purchases at the market price. It ignores shares
   that a company gives to an insider as pay.
3. The screener finds each company where two or more insiders bought stock in
   a period of 14 days.
4. The model gives a score and a band to each cluster buy. The research made
   the model from eight years of data.
5. The screener writes the file `out/dashboard.html`. You can open this file
   in a web browser or send it by email. The screener also writes a
   spreadsheet with the same data.

## How to read the dashboard

Each cluster buy has a score from 0 to 100. The score compares the cluster buy
with the cluster buys of the last eight years. A score of 80 means that the
cluster buy has a higher rank than 80% of those cluster buys.

| Band | Score | Results in the past | Recommendation |
|---|---|---|---|
| **Top band** | 70 to 90 | This band had the lowest risk of a large loss. Approximately 1 in 75 stocks lost more than 30% in one month. On average, these stocks did approximately as well as the market. | Examine these first. |
| **Above band** | 90 to 100 | These stocks had higher scores than the top band, but their results were worse. Their risk of a large loss was approximately two times the risk in the top band. | Examine these second. |
| **Middle** | 30 to 70 | On average, these stocks did slightly worse than the market. | Low priority. |
| **Elevated risk** | 0 to 30 | Approximately 1 in 13 stocks lost more than 30% in one month. This is the clearest warning that the project found. | Ignore these, or examine them carefully. |

The dashboard can also show two other labels:

- **Unavailable:** The screener did not have sufficient data to score the
  company.
- **Not scored:** The screener did not find a model. Refer to
  [Make the model](#make-the-model).

## Procedures

### Requirements

- Python 3.11 or a later version.
- Your name and your email address. The SEC tells each automatic program to
  identify itself. The SEC can block a program that does not identify itself.
- Time. On the first run, the screener downloads each report in the period.
  This can take 30 minutes or more. Later runs use the reports that the
  screener downloaded before.

### Run the screener

1. Make a copy of the settings file:

   ```
   copy .env.example .env        (Windows)
   cp .env.example .env          (Mac and Linux)
   ```

2. Open the file `.env` in a text editor.
3. Set `SEC_USER_AGENT` to your name and your email address. For example:
   `SEC_USER_AGENT=Jane Smith jane@example.com`.
4. Start the screener:
   - **Windows:** Run `run.bat`. On the first run, `run.bat` installs the
     necessary Python packages.
   - **Mac and Linux:** Type these commands:

     ```
     python3 -m venv .venv
     . .venv/bin/activate
     pip install -r requirements.txt
     python insider_cluster_buys.py
     ```

5. Answer the three questions. To use the default value, push Enter.
   - The number of days to examine. The default is 30.
   - The minimum number of insiders in a cluster buy. The default is 2.
   - The maximum number of days between the purchases. The default is 14.
6. Open the file `out/dashboard.html` in a web browser.

### Make the model

A new copy of this project does not contain a model. Thus, the dashboard shows
"Not scored" for each cluster buy. To make the model, the project must
download data for eight years. This takes approximately one day for each year
of data. The data also uses many GB of disk space.

1. To download the data, type `python run_research.py --all`.
2. To make the model, type `python run_research.py --fit-screen`.

After this procedure, the screener gives a score to each cluster buy. For the
backtester, refer to [docs/backtester.md](docs/backtester.md).

## Accuracy of the results

The project tests its own results. These tests found errors in the past:

- Until September 2026, the project counted each insider report two times.
  The project corrected this error. Then it measured all of the results again.
- One strategy showed a large profit. Most of that profit came from one stock.
  The data for that stock did not include a stock split.
- The project tested 7,560 strategies. The best strategy lost money on years
  that the project did not use to select it.

The project now accepts a result only if the result passes two tests:

1. The result must stay true on years that the research did not use to select
   it.
2. The result must be better than the result of the same search on random
   data.

Most results do not pass these tests. This README gives only the positive
results that passed. For all of the results and their numbers, refer to
[docs/findings.md](docs/findings.md).

## Contents of the repository

| File or folder | Contents |
|---|---|
| `insider_cluster_buys.py` | The screener. |
| `run.bat` | Starts the screener on Windows. |
| `findings.py` | Each result that the dashboard shows, with the source of the result. When you change a value in this file, the dashboard also changes. |
| `build_html.py` | Makes the dashboard. |
| `research/` | The model. |
| `run_research.py` | Makes the dataset and the model. |
| `backtest.py`, `backtest/`, `backtest.bat` | The backtester. The backtester tests trading strategies on past data. |
| `tools/` | Research experiments. You can run each experiment separately. |
| `tests/` | Automatic tests. To run them, type `python -m pytest tests -q`. |
| `RESEARCH_NOTES.md` | A record of all of the research. The record includes the experiments that failed. |

## More information

- [docs/screener.md](docs/screener.md): How the screener and the model
  operate. All of the output files and settings. How to run the screener on a
  schedule.
- [docs/backtester.md](docs/backtester.md): The backtester and the research
  procedure.
- [docs/findings.md](docs/findings.md): All of the results, with their numbers
  and limits.
- [docs/code-layout.md](docs/code-layout.md): All of the modules, and the data
  caches.
- [RESEARCH_NOTES.md](RESEARCH_NOTES.md): The full research record, in the
  sequence that it occurred.

## Glossary

- **Backtest:** A test of a trading strategy on past data.
- **Cluster buy:** Two or more insiders of one company buy stock in that
  company at the market price, in a specified number of days. The default is
  14 days.
- **EDGAR:** The public database of company reports at the SEC.
- **Form 4:** The form that an insider uses to report a trade in the stock of
  their company. The insider must send it in two business days.
- **Holdout:** Data that the research does not use to make or adjust a
  strategy. The research keeps this data for a final test.
- **Index fund:** A fund that holds all of the stocks in one part of the
  market. SPY holds the 500 largest U.S. companies. IWM holds approximately
  2,000 small U.S. companies.
- **Insider:** An officer of a company, a director of the company, or a person
  who owns more than 10% of its stock.
- **Open-market purchase:** A purchase of stock at the market price with the
  money of the insider. Stock that a company gives to an insider as pay is not
  an open-market purchase.
- **Percentile:** A rank from 0 to 100. A value at the 80th percentile is
  higher than 80% of the values in the group.
- **Permutation test:** A test that compares a result with the results of the
  same search on random data. If random data often gives an equal result, the
  result is not real.
- **SEC:** The U.S. Securities and Exchange Commission. This government agency
  controls the stock market.
- **Sharpe ratio:** The return divided by the amount of change in the return.
  A higher value is better.
- **Small-cap:** A company with a small total stock value.
- **Survivorship bias:** An error that occurs when the data contains only the
  companies that continue to trade. The data loses the companies that became
  bankrupt or that other companies bought. Thus, the past looks better than it
  was.
- **Trading day:** A day when the stock market is open. One month has
  approximately 21 trading days. One year has approximately 252.
- **Trailing stop:** A sell rule. The rule monitors the highest price of a
  stock after the purchase. The rule sells the stock when the price decreases
  by a specified percentage from that highest price. The rule in this project
  starts when the price increases by 10%, and sells at 15% below the highest
  price.
