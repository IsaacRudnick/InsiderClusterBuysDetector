# Insider Cluster Buys Detector

Finds the moments when several people inside a company buy that company's
stock at the same time, and tests, honestly, whether that is worth acting on.

> **Not financial advice.** This is a research and screening tool, and its own
> conclusion is that following these purchases does not beat an index fund.

## The idea in one minute

The people who run a company (its executives, its board members and its
largest shareholders) are called **insiders**. U.S. law requires them to
report every trade in their own company's stock to the SEC, the government's
market regulator, within two business days. Those reports are public.

Insiders sell for all sorts of reasons: taxes, a house, spreading their money
around. They usually *buy* for only one: they think the stock will go up. When
**several** insiders at the same company buy with their own money within a
couple of weeks, that is a **cluster buy**. A common piece of investing advice
says to pay attention to them.

This project does two things with that idea:

1. **The screener** reads the latest SEC filings, finds recent cluster buys
   and sorts them into a dashboard you open in a browser.
2. **The research** replays eight years of history (2018 to 2026, about 11,000
   cluster buys) to check whether the advice actually works.

## What it found

The short answer: **following cluster buys does not beat simply owning an index
fund.** But the research did turn up something useful.

- **A cluster buy is not, on its own, a reason to buy.** On average, buying
  after a cluster buy did about as well as a small-company index fund over two
  weeks. The longer the stock was held, the worse it did.
- **The project can spot the risky ones.** Its scoring model sorts cluster buys
  into four groups. In the lowest-scoring 30%, about 1 in 13 stocks lost more
  than 30% of its value within a month. That is roughly three times the rate of
  the rest, and it held in every year tested. It is the most reliable result
  here.
- **A higher score is not always better.** The very top-scoring 10% did *worse*
  than the group just below it.
- **It does not add up to a winning portfolio.** Some versions looked as if
  they beat the market, but the advantage disappeared once realistic trading
  costs were included, or when they were tested on years they had not been
  tuned on.
- **One selling rule helped every time.** A *trailing stop* (once a stock is up
  10%, sell if it falls 15% below its best price) beat "hold for a month, then
  sell" in all 25 combinations tested. It doesn't need the scoring model.
- **Every return figure here is a little too good.** About a third of the
  companies in the data no longer trade: they were bought out, renamed or went
  bankrupt, and their prices are missing. Putting them back lowers every
  result, enough to wipe out the average trade's profit.

So the dashboard works best as a **reading list and a skip list**. It tells you
which cluster buys to research first and which to be wary of. It does not tell
you which ones to buy.

## How it works

```
 SEC filings  ──▶  find cluster buys  ──▶  score each one  ──▶  dashboard
 (public,          (2+ insiders buying      (a model trained      (four groups,
  every day)        within 14 days)          on 8 years of         open it in a
                                             history)              browser)
```

1. **Download.** Every business day the SEC publishes a list of new filings.
   The screener reads the insider-trade reports (a form called **Form 4**) for
   the period you choose, 30 days by default.
2. **Find clusters.** It keeps only ordinary purchases at the market price,
   ignoring shares handed out as pay. Then it looks for companies where at
   least 2 different insiders bought within 14 days of each other.
3. **Score.** A model trained on eight years of past cluster buys gives each
   new one a score and places it in a group.
4. **Report.** It writes `out/dashboard.html`, a single file you can open or
   email, plus a spreadsheet with the same data.

## Reading the dashboard

Each cluster buy gets a score from 0 to 100: its rank against eight years of
past cluster buys. A score of 80 means it ranked above 80% of them. Scores are
grouped into four bands:

| Band | Score | How these have gone before | Suggested use |
|---|---|---|---|
| **Top band** | 70 to 90 | The lowest chance of a big loss: about 1 in 75 lost more than 30% within a month. Typically about even with the market, so not a predicted winner. | Research first |
| **Above band** | 90 to 100 | Scored higher, but did worse than the top band, with about twice its chance of a big loss. | Research second |
| **Middle** | 30 to 70 | Typically trailed the market slightly. | Low priority |
| **Elevated risk** | 0 to 30 | About 1 in 13 lost more than 30% within a month. The clearest warning sign the project found. | Skip, or research with care |

Two other labels can appear. **Unavailable** means there was not enough data
about that company to score it. **Not scored** means no trained model was
found; see [Getting scores](#getting-scores).

## Getting started

### What you need

- **Python 3.11 or newer.**
- **Your name and email address.** The SEC asks every automated visitor to
  identify itself and may block one that doesn't. The screener also keeps
  itself under the SEC's speed limit.
- **Some patience the first time.** The screener downloads every filing in the
  period you choose, which can take half an hour or more. Later runs reuse what
  they already downloaded.

### Run the screener

1. Copy the settings template:

   ```
   copy .env.example .env        (Windows)
   cp .env.example .env          (Mac / Linux)
   ```

   Open `.env` and set `SEC_USER_AGENT` to your name and email, for example
   `SEC_USER_AGENT=Jane Smith jane@example.com`.

2. Start it.

   - **Windows:** run `run.bat`. It sets up its own Python environment the
     first time.
   - **Mac / Linux:**

     ```
     python3 -m venv .venv
     . .venv/bin/activate
     pip install -r requirements.txt
     python insider_cluster_buys.py
     ```

3. Answer three questions, or press Enter to keep each default:
   - how many days back to look (30),
   - how many different insiders make a cluster (2),
   - how close together their purchases must be (14 days).

4. Open `out/dashboard.html` in a browser.

### Getting scores

A fresh copy of this project has no trained model, so the first dashboard
lists every cluster buy as **Not scored**. The model is built from eight years
of history, which has to be downloaded once. That is slow (roughly a day per
year of history) and takes several GB of disk:

```
python run_research.py --all          # download the history and build the research data
python run_research.py --fit-screen   # train the model the screener uses
```

After that, every screener run is scored. To also replay trading strategies
over that history, see [docs/backtester.md](docs/backtester.md).

## How much to trust the numbers

The project checks itself hard, and has caught itself several times:

- It once counted every insider filing twice. That was fixed in September
  2026, and every number was measured again.
- A strategy that looked like a big winner turned out to owe most of its profit
  to one stock whose share split had not been accounted for.
- The best of 7,560 strategy variations, tested on years it had never seen,
  lost money.

A result now counts only if it holds up on years it was not chosen on, and if
it does better than the same search run on shuffled, meaningless data. Most
results don't survive that. The positive results in this README are the ones
that did. The full list, with numbers, is in
[docs/findings.md](docs/findings.md).

## What's in this repository

| File or folder | What it is |
|---|---|
| `insider_cluster_buys.py` | The screener. Start here. |
| `run.bat` | Windows launcher for the screener. |
| `findings.py` | Every measured result the dashboard quotes, each with its source. Correct a number here and the dashboard corrects itself. |
| `build_html.py` | Draws the dashboard. |
| `research/` | The scoring model. |
| `run_research.py` | Builds the history dataset and trains the model. |
| `backtest.py`, `backtest/`, `backtest.bat` | The backtester, which replays strategies over history. |
| `tools/` | Individual research experiments, each runnable on its own. |
| `tests/` | Automated tests: `python -m pytest tests -q`. |
| `RESEARCH_NOTES.md` | The lab notebook: every experiment, including the failures. |

## Learn more

- [docs/screener.md](docs/screener.md): how the screener and its model work,
  every output and setting, and running it on a schedule.
- [docs/backtester.md](docs/backtester.md): the backtester and the research
  pipeline.
- [docs/findings.md](docs/findings.md): every result, with numbers and
  caveats.
- [docs/code-layout.md](docs/code-layout.md): every module, and the data
  caches the project builds.
- [RESEARCH_NOTES.md](RESEARCH_NOTES.md): the full research record, in the
  order it happened.

## Glossary

- **Insider:** a company's officers (CEO, CFO and so on), its directors, and
  anyone who owns more than 10% of its stock.
- **SEC, EDGAR:** the U.S. Securities and Exchange Commission, and its public
  database of company filings.
- **Form 4:** the form an insider files to report a trade in their company's
  stock, due within two business days.
- **Open-market purchase:** buying shares at the going price with one's own
  money, as opposed to receiving them as pay.
- **Cluster buy:** two or more different insiders at one company making
  open-market purchases within a set number of days (14 by default).
- **Index fund, SPY, IWM:** a fund that simply owns a whole slice of the
  market. SPY follows the 500 largest U.S. companies; IWM follows about 2,000
  small ones. "Beating the market" means doing better than funds like these.
- **Small-cap:** a company whose total stock-market value is small. Most
  cluster buys happen in small-caps.
- **Percentile:** a rank out of 100. The 80th percentile is higher than 80% of
  the group.
- **Backtest:** replaying a trading strategy on past data to see how it would
  have done.
- **Holdout, out of sample:** data set aside and not used to build or tune a
  strategy, kept for a fair final test. A result that only works on the data
  it was tuned on is usually luck.
- **Permutation test:** a luck check. Run the same search many times on
  shuffled data; if the shuffled data often does as well, the original result
  is not real.
- **Trailing stop:** a selling rule that tracks a stock's highest price since
  purchase and sells once it falls a set percentage below that peak.
- **Survivorship bias:** the distortion from only seeing what survived.
  Companies that went bankrupt or were bought out drop out of price data, so
  history looks better than it really was.
- **Trading day:** a day the stock market is open; about 21 in a month and 252
  in a year.
- **Sharpe ratio:** return divided by how much it swung up and down along the
  way. Higher is better. The technical docs use it throughout.
