Predict Bitcoin's Next Hour — AlphaI × Polaris Challenge



Deadline: 7 days from today, 23:59 IST.
Time budget: 6–10 hours of work, spread across the week.
Who this is for: Polaris students who were in the AlphaI guest lecture and want to work with us.
What you walk away with: top 3-10 submissions get invited to the interview.



Your mission

Every hour, a new candle closes on Bitcoin's chart. Your job is to predict the price range where BTC will land one hour from now.

Not an exact number. A range.

You say: "I'm 95% sure BTC will be between $67,200 and $67,800 one hour from now."

An hour later, the real price is $67,650. You were right. Score.

An hour after that you predict $67,500–$68,100 and real price is $68,400. Wrong.

The best forecaster is the one who is (1) right ~95% of the time AND (2) keeps the range as narrow as possible. A range of $0–$1,000,000 is always right but useless. A super-narrow range that keeps missing means you're overconfident. We want both — accurate and tight.



What we give you

A working Python model in a Colab notebook. It already knows how to:





Look at recent Bitcoin price history



Measure how jumpy the price has been (volatility)



Simulate 10,000 possible "next hours"



Read off the 95% range from those simulations

The model is called a Geometric Brownian Motion (GBM) simulator. Don't stress the name. Think of it as a weather forecast: given current conditions, simulate many possible tomorrows, see where 95% of them end up.

The Colab notebook currently runs on currency data (daily USD/Swiss-Franc prices). Your job: make it run on Bitcoin (BTCUSDT, 1-hour bars) using Binance's free public API. No API key or account needed — the data is fully public.

Starter Colab: https://colab.research.google.com/drive/1ST_0CEk3EB17mruUUoShVutJug4bnCyV?usp=sharing 

Open it. Click "File → Save a copy in Drive". Work on your copy.



What you build — three parts

Part A — 30-day backtest (required)

Fetch the last 30 days of BTCUSDT 1-hour bars from Binance — that's about 720 bars.

For each of those 720 bars, pretend you don't know the future. At each bar:





Look only at data up to that bar (no peeking at later bars).



Use the model to predict the 95% range for the next bar.



Reveal the actual next-bar price. Check if it fell inside your predicted range.

At the end, you will have three numbers that summarize how good your forecaster is:





Coverage. What fraction of the 720 predictions actually contained reality inside your 95% range? Target: close to 0.95. Higher than 0.95 means your ranges are too wide (you're playing it too safe). Lower means you're overconfident and missing.



Average width. How wide was your typical range? Narrower is better — but only if coverage stays near 0.95.



Winkler score. A single formula that combines both accuracy and tightness. A range that contained reality gets a score equal to its width. A range that missed gets a big penalty proportional to how far off it was. Lower Winkler = better forecaster.

The Colab has a helper function evaluate(predictions) that computes all three for you. You don't implement them yourself.

Save your 720 predictions to a file named backtest_results.jsonl (one prediction per line).

Part B — Live dashboard (required)

Build a small live dashboard. When anyone visits it, it should:





Fetch the very latest closed BTCUSDT 1-hour bar from Binance (in real time, right now).



Run your model on the last 500 bars.



Display:





The current BTC price



Your predicted 95% range for the next hour



A chart of the last 50 bars showing the price line with your predicted range as a shaded ribbon



Your Part-A backtest metrics (coverage, avg width, Winkler) as headline numbers at the top

Use whatever tool you want to build and host it. Streamlit on Streamlit Community Cloud is the easiest route and we recommend it if you've never done this before — ~100 lines of Python and a free public URL. But Gradio on HuggingFace Spaces, a plain Flask/FastAPI app with Plotly, a Dash app, a Next.js page calling a Python backend, or anything else with a public URL is equally fine. Pick what you'll ship fastest.

You paste the public URL into the submission form. We open the URL to verify it works.

Don't overthink the design. We grade "does it load and show the right numbers", not "is it beautiful".

Part C — Persistence (optional bonus)

Every time someone visits your dashboard, save the prediction it just made. Next time the dashboard is opened, show the full history of predictions — not just the current one. After a few days, we see a growing timeline of predictions with actuals filled in as bars close.

No points deducted for skipping Part C. But candidates who do this signal they understand how real trading dashboards work. It's a tiebreaker.



The three concepts you need to actually understand

Forget everything else in the starter notebook for a moment. If you grasp these three things, you can do well on this assignment.

1. No peeking. When predicting the price at bar N, you can only use data up to bar N−1. If you peek at bar N's actual price to "help" your prediction, your backtest will look amazing in testing and your model will fail live. This is the #1 bug in beginner forecasters. Your code must be structured so that there is no way for bar-N data to leak into the bar-N prediction.

2. Volatility clustering. Bitcoin isn't always equally jumpy. Calm hours tend to cluster together; violent hours tend to cluster together. A good forecaster uses recent volatility as the main input for how wide the next-hour range should be. If the last 10 bars were calm, predict a narrow range. If they were violent, predict a wider one.

3. Fat tails. Compared to stocks, Bitcoin has surprisingly frequent huge moves. A model that assumes returns follow a normal bell curve will systematically under-predict how wide the range needs to be. The starter notebook already uses a Student-t distribution (a bell curve with heavier tails) to handle this. Don't replace it with a normal distribution — you'll get worse coverage.

Everything else in the starter is plumbing. Spend your time on these three concepts and on building a clean dashboard.



How we grade





Coverage accuracy — how close your observed coverage is to 0.95.



Range tightness — your Winkler score vs our reference solution's.



Dashboard alive — your dashboard URL produces a fresh prediction when we open it.



How to submit

Fill the submission form (link below). You will need to provide:





Your name, email, Polaris year/batch



Link to your work (GitHub repo URL or Colab notebook share link — either is fine, as long as "Anyone with the link can view" is enabled for Colab)



Your deployed dashboard URL (any framework, any host, as long as it's publicly accessible)



Your observed coverage_95 and mean_winkler_95 from Part A

Submission form: https://docs.google.com/forms/d/e/1FAIpQLSe93W4OX2z08uA_pkGddMRSJk9mb4F7d2mXozNez8kT4hyXwQ/viewform?usp=publish-editor

One submission per person. You can re-submit to fix bugs — we take the latest.



FAQ

I've never built a web dashboard. Is that a blocker?
No. Streamlit is the shortest path — read the 30-minute Streamlit tutorial, and hosting is free on Streamlit Community Cloud in ~100 lines of Python. Gradio on HuggingFace Spaces is similarly easy. If you already know Flask, FastAPI, Dash, Next.js, or anything else — use what you already know. We only care that the URL loads and shows the right numbers.

Is Binance blocked in India?
api.binance.com can be geo-blocked for some networks. Use https://data-api.binance.vision/api/v3/klines instead — same endpoint, no geo block, fully public. The starter Colab uses this by default.

Can I use a different model than GBM?
Stick with GBM for the main path. If you beat it with something else, document what you changed and why — you'll get tiebreaker points. But the first goal is making the starter GBM work on Bitcoin.

Can I team up?
No. Individual submissions only. Discuss ideas with peers if you want; the code must be yours.

I don't have a GPU. Does that matter?
Not at all. This runs on a laptop CPU in minutes.

I found a bug in the starter notebook. Can I fix it in my submission?
Yes. And tell us about it in the "bugs you spotted" field of the form. That's exactly the signal we're looking for.

How do I keep my dashboard live for grading?
Use any host that keeps your app reachable for at least 7 days. Streamlit Community Cloud and HuggingFace Spaces both do this for free; they "sleep" after zero traffic and wake in ~30 seconds when we visit, which is fine. Don't use a setup that auto-deletes after a few hours.

Where do I ask questions?
Reply to the email that sent you the form. We'll consolidate answers into this FAQ as they come in.