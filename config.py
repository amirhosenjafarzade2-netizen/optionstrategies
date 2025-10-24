# config.py
from dataclasses import dataclass
from typing import List, Literal

Position = Literal["long", "short"]
LegType = Literal["call", "put", "stock"]


@dataclass
class Leg:
    type: LegType
    position: Position
    strike: float
    premium: float = 0.0  # 0 for stock


@dataclass
class Strategy:
    name: str
    description: str
    legs: List[Leg]
    max_profit: str
    max_loss: str
    good: str
    bad: str
    is_custom: bool = False


# ================================================================== #
# ALL 17 PREDEFINED STRATEGIES (exact match to original JS)
# ================================================================== #
PREDEFINED_STRATEGIES: List[Strategy] = [
    Strategy(
        name="Long Call",
        description="Buying a call option gives you the right to buy the underlying stock at the strike price. Benefits from rising prices.",
        legs=[Leg(type="call", position="long", strike=100, premium=5)],
        max_profit="Unlimited",
        max_loss="Premium paid",
        good="Bullish market outlook; expect significant price increase; low capital outlay for high potential returns.",
        bad="Bearish or flat market; high volatility increases premium costs; time decay erodes value."
    ),
    Strategy(
        name="Long Put",
        description="Buying a put option gives you the right to sell the underlying stock at the strike price. Benefits from falling prices.",
        legs=[Leg(type="put", position="long", strike=100, premium=5)],
        max_profit="Strike price - Premium",
        max_loss="Premium paid",
        good="Bearish market outlook; expect sharp price decline; hedge against stock declines.",
        bad="Bullish or flat market; high volatility raises premium; time decay hurts value."
    ),
    Strategy(
        name="Short Call",
        description="Selling a call option obligates you to sell the stock at the strike price if exercised. Profits from premium if stock stays flat or falls.",
        legs=[Leg(type="call", position="short", strike=100, premium=5)],
        max_profit="Premium received",
        max_loss="Unlimited",
        good="Neutral to bearish outlook; low volatility; expect stock to stay below strike.",
        bad="Bullish market; unexpected price surge; high volatility increases risk."
    ),
    Strategy(
        name="Short Put",
        description="Selling a put option obligates you to buy the stock at the strike price if exercised. Profits from premium if stock stays flat or rises.",
        legs=[Leg(type="put", position="short", strike=100, premium=5)],
        max_profit="Premium received",
        max_loss="Strike price - Premium",
        good="Neutral to bullish outlook; stock likely above strike; low volatility.",
        bad="Bearish market; sharp price drop; high volatility increases risk."
    ),
    Strategy(
        name="Covered Call",
        description="Own the stock and sell a call option. Generates income but caps upside potential.",
        legs=[
            Leg(type="stock", position="long", strike=100, premium=0),
            Leg(type="call", position="short", strike=105, premium=3)
        ],
        max_profit="Strike - Stock Price + Premium",
        max_loss="Stock Purchase Price - Premium",
        good="Neutral to slightly bullish; generate income; stock unlikely to surge past strike.",
        bad="Strong bullish market; missed upside potential; significant stock price drop."
    ),
    Strategy(
        name="Protective Put",
        description="Own the stock and buy a put option. Insurance against downside risk.",
        legs=[
            Leg(type="stock", position="long", strike=100, premium=0),
            Leg(type="put", position="long", strike=95, premium=3)
        ],
        max_profit="Unlimited",
        max_loss="Stock Price - Strike + Premium",
        good="Holding stock with uncertain downside; bearish concerns; portfolio protection.",
        bad="Strong bullish market; premium cost reduces returns; low volatility."
    ),
    Strategy(
        name="Bull Call Spread",
        description="Buy a lower strike call and sell a higher strike call. Limited profit and loss.",
        legs=[
            Leg(type="call", position="long", strike=100, premium=5),
            Leg(type="call", position="short", strike=110, premium=2)
        ],
        max_profit="Difference in strikes - Net Premium",
        max_loss="Net premium paid",
        good="Moderately bullish outlook; want lower cost than long call; defined risk.",
        bad="Sharp price drop or flat market; large price surge beyond higher strike."
    ),
    Strategy(
        name="Bear Put Spread",
        description="Buy a higher strike put and sell a lower strike put. Limited profit and loss.",
        legs=[
            Leg(type="put", position="long", strike=100, premium=5),
            Leg(type="put", position="short", strike=90, premium=2)
        ],
        max_profit="Difference in strikes - Net Premium",
        max_loss="Net premium paid",
        good="Moderately bearish outlook; lower cost than long put; defined risk.",
        bad="Sharp price rise or flat market; limited profit potential."
    ),
    Strategy(
        name="Long Straddle",
        description="Buy both a call and put at the same strike. Profits from large moves in either direction.",
        legs=[
            Leg(type="call", position="long", strike=100, premium=5),
            Leg(type="put", position="long", strike=100, premium=5)
        ],
        max_profit="Unlimited",
        max_loss="Total premiums paid",
        good="Expect large price movement (up or down); high volatility expected.",
        bad="Flat market; low volatility; high premium costs."
    ),
    Strategy(
        name="Short Straddle",
        description="Sell both a call and put at the same strike. Profits from low volatility.",
        legs=[
            Leg(type="call", position="short", strike=100, premium=5),
            Leg(type="put", position="short", strike=100, premium=5)
        ],
        max_profit="Total premiums received",
        max_loss="Unlimited",
        good="Neutral market; expect low volatility; stock stays near strike.",
        bad="Large price movement; high volatility; unlimited risk."
    ),
    Strategy(
        name="Long Strangle",
        description="Buy a call and put at different strikes. Cheaper than straddle, needs larger move to profit.",
        legs=[
            Leg(type="call", position="long", strike=110, premium=3),
            Leg(type="put", position="long", strike=90, premium=3)
        ],
        max_profit="Unlimited",
        max_loss="Total premiums paid",
        good="Expect very large price movement; lower cost than straddle.",
        bad="Flat or small price movement; low volatility; premium costs."
    ),
    Strategy(
        name="Short Strangle",
        description="Sell a call and put at different strikes. Higher profit range than short straddle.",
        legs=[
            Leg(type="call", position="short", strike=110, premium=3),
            Leg(type="put", position="short", strike=90, premium=3)
        ],
        max_profit="Total premiums received",
        max_loss="Unlimited",
        good="Neutral market; wider range for profit than short straddle; low volatility.",
        bad="Large price movement; high volatility; unlimited risk."
    ),
    Strategy(
        name="Iron Condor",
        description="Sell a strangle and buy a wider strangle for protection. Profits from low volatility with defined risk.",
        legs=[
            Leg(type="put", position="long", strike=85, premium=1),
            Leg(type="put", position="short", strike=90, premium=3),
            Leg(type="call", position="short", strike=110, premium=3),
            Leg(type="call", position="long", strike=115, premium=1)
        ],
        max_profit="Net premiums received",
        max_loss="Difference in strikes - Net Premium",
        good="Neutral market; expect stock to stay within a range; defined risk.",
        bad="Large price movement; high volatility; breakout beyond strikes."
    ),
    Strategy(
        name="Butterfly Spread (Call)",
        description="Buy 1 lower strike call, sell 2 middle strike calls, buy 1 higher strike call. Profits from low volatility.",
        legs=[
            Leg(type="call", position="long", strike=90, premium=12),
            Leg(type="call", position="short", strike=100, premium=6),
            Leg(type="call", position="short", strike=100, premium=6),
            Leg(type="call", position="long", strike=110, premium=3)
        ],
        max_profit="Middle strike - Lower strike - Net Premium",
        max_loss="Net premium paid",
        good="Neutral market; expect stock to stay near middle strike; low cost.",
        bad="Large price movement; high volatility; limited profit potential."
    ),
    Strategy(
        name="Iron Butterfly",
        description="Sell a straddle and buy a strangle for protection. Profits from very low volatility with defined risk.",
        legs=[
            Leg(type="put", position="long", strike=90, premium=2),
            Leg(type="put", position="short", strike=100, premium=5),
            Leg(type="call", position="short", strike=100, premium=5),
            Leg(type="call", position="long", strike=110, premium=2)
        ],
        max_profit="Net premiums received",
        max_loss="Difference in strikes - Net Premium",
        good="Neutral market; expect minimal price movement; defined risk.",
        bad="Large price movement; high volatility; breakout beyond strikes."
    ),
    Strategy(
        name="Calendar Spread (Call)",
        description="Sell a near-term call and buy a longer-term call at the same strike. Note: This visualization assumes same expiration for simplicity.",
        legs=[
            Leg(type="call", position="short", strike=100, premium=3),
            Leg(type="call", position="long", strike=100, premium=5)
        ],
        max_profit="Varies with time decay",
        max_loss="Net premium paid",
        good="Neutral to slightly bullish; benefit from time decay; expect stable price.",
        bad="Large price movement; high volatility; expiration mismatch complexity."
    ),
    Strategy(
        name="Collar",
        description="Own stock, buy protective put, sell covered call. Limits both upside and downside.",
        legs=[
            Leg(type="stock", position="long", strike=100, premium=0),
            Leg(type="put", position="long", strike=95, premium=3),
            Leg(type="call", position="short", strike=105, premium=3)
        ],
        max_profit="Call strike - Stock price + Net Premium",
        max_loss="Stock price - Put strike - Net Premium",
        good="Protect stock holdings; neutral market; generate income with downside protection.",
        bad="Strong bullish market; caps upside potential; premium costs."
    ),
]
