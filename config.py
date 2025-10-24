# config.py
from dataclasses import dataclass, field
from typing import List, Literal

Position = Literal["long", "short"]
LegType = Literal["call", "put", "stock"]

@dataclass
class Leg:
    type: LegType
    position: Position
    strike: float
    premium: float = 0.0
    
    def __post_init__(self):
        """Validate leg parameters"""
        if self.strike < 0:
            raise ValueError("Strike price cannot be negative")
        if self.premium < 0:
            raise ValueError("Premium cannot be negative")
        if self.type == "stock" and self.premium != 0:
            self.premium = 0.0  # Force stock premium to 0
    
    def net_cost(self) -> float:
        """Calculate net cost of the leg (negative for credits)"""
        if self.position == "long":
            return self.premium  # Cost
        else:
            return -self.premium  # Credit

@dataclass
class Strategy:
    name: str
    short_desc: str
    description: str
    legs: List[Leg]
    max_profit: str
    max_loss: str
    good: str
    bad: str
    is_custom: bool = False
    
    def __post_init__(self):
        """Validate strategy"""
        if not self.name:
            raise ValueError("Strategy name cannot be empty")
        if not self.legs:
            raise ValueError("Strategy must have at least one leg")
    
    def net_debit_credit(self) -> float:
        """Calculate net debit (positive) or credit (negative) of strategy"""
        return sum(leg.net_cost() for leg in self.legs)
    
    def is_debit_strategy(self) -> bool:
        """Check if strategy requires upfront payment"""
        return self.net_debit_credit() > 0
    
    def is_credit_strategy(self) -> bool:
        """Check if strategy receives upfront payment"""
        return self.net_debit_credit() < 0

# Predefined strategies with consistent formatting
PREDEFINED_STRATEGIES: List[Strategy] = [
    Strategy(
        name="Long Call",
        short_desc="Bullish: Unlimited upside, limited risk",
        description="Buying a call option gives you the right to buy the underlying stock at the strike price. Benefits from rising prices.",
        legs=[Leg(type="call", position="long", strike=100, premium=5)],
        max_profit="Unlimited",
        max_loss="Premium paid ($5)",
        good="Bullish market outlook; expect significant price increase; low capital outlay for high potential returns.",
        bad="Bearish or flat market; high volatility increases premium costs; time decay erodes value."
    ),
    Strategy(
        name="Long Put",
        short_desc="Bearish: Downside protection, limited risk",
        description="Buying a put option gives you the right to sell the underlying stock at the strike price. Benefits from falling prices.",
        legs=[Leg(type="put", position="long", strike=100, premium=5)],
        max_profit="Strike - Premium ($95)",
        max_loss="Premium paid ($5)",
        good="Bearish market outlook; expect sharp price decline; hedge against stock declines.",
        bad="Bullish or flat market; high volatility raises premium; time decay hurts value."
    ),
    Strategy(
        name="Short Call",
        short_desc="Neutral/Bearish: Income from premium",
        description="Selling a call option obligates you to sell the stock at the strike price if exercised. Profits from premium if stock stays flat or falls.",
        legs=[Leg(type="call", position="short", strike=100, premium=5)],
        max_profit="Premium received ($5)",
        max_loss="Unlimited",
        good="Neutral to bearish outlook; low volatility; expect stock to stay below strike.",
        bad="Bullish market; unexpected price surge; high volatility increases risk."
    ),
    Strategy(
        name="Short Put",
        short_desc="Neutral/Bullish: Income from premium",
        description="Selling a put option obligates you to buy the stock at the strike price if exercised. Profits from premium if stock stays flat or rises.",
        legs=[Leg(type="put", position="short", strike=100, premium=5)],
        max_profit="Premium received ($5)",
        max_loss="Strike - Premium ($95)",
        good="Neutral to bullish outlook; stock likely above strike; low volatility.",
        bad="Bearish market; sharp price drop; high volatility increases risk."
    ),
    Strategy(
        name="Covered Call",
        short_desc="Income + limited upside",
        description="Own the stock and sell a call option. Generates income but caps upside potential.",
        legs=[
            Leg(type="stock", position="long", strike=100, premium=0),
            Leg(type="call", position="short", strike=105, premium=3)
        ],
        max_profit="(Strike - Stock Price) + Premium ($8)",
        max_loss="Stock Price - Premium ($97)",
        good="Neutral to slightly bullish; generate income; stock unlikely to surge past strike.",
        bad="Strong bullish market; missed upside potential; significant stock price drop."
    ),
    Strategy(
        name="Protective Put",
        short_desc="Downside insurance",
        description="Own the stock and buy a put option. Insurance against downside risk.",
        legs=[
            Leg(type="stock", position="long", strike=100, premium=0),
            Leg(type="put", position="long", strike=95, premium=3)
        ],
        max_profit="Unlimited",
        max_loss="(Stock - Strike) + Premium ($8)",
        good="Holding stock with uncertain downside; bearish concerns; portfolio protection.",
        bad="Strong bullish market; premium cost reduces returns; low volatility."
    ),
    Strategy(
        name="Bull Call Spread",
        short_desc="Moderately bullish, defined risk",
        description="Buy a lower strike call and sell a higher strike call. Limited profit and loss.",
        legs=[
            Leg(type="call", position="long", strike=100, premium=5),
            Leg(type="call", position="short", strike=110, premium=2)
        ],
        max_profit="Strike Difference - Net Premium ($7)",
        max_loss="Net premium paid ($3)",
        good="Moderately bullish outlook; want lower cost than long call; defined risk.",
        bad="Sharp price drop or flat market; large price surge beyond higher strike."
    ),
    Strategy(
        name="Bear Put Spread",
        short_desc="Moderately bearish, defined risk",
        description="Buy a higher strike put and sell a lower strike put. Limited profit and loss.",
        legs=[
            Leg(type="put", position="long", strike=100, premium=5),
            Leg(type="put", position="short", strike=90, premium=2)
        ],
        max_profit="Strike Difference - Net Premium ($7)",
        max_loss="Net premium paid ($3)",
        good="Moderately bearish outlook; lower cost than long put; defined risk.",
        bad="Sharp price rise or flat market; limited profit potential."
    ),
    Strategy(
        name="Long Straddle",
        short_desc="High volatility play",
        description="Buy both a call and put at the same strike. Profits from large moves in either direction.",
        legs=[
            Leg(type="call", position="long", strike=100, premium=5),
            Leg(type="put", position="long", strike=100, premium=5)
        ],
        max_profit="Unlimited",
        max_loss="Total premiums paid ($10)",
        good="Expect large price movement (up or down); high volatility expected.",
        bad="Flat market; low volatility; high premium costs."
    ),
    Strategy(
        name="Short Straddle",
        short_desc="Low volatility income",
        description="Sell both a call and put at the same strike. Profits from low volatility.",
        legs=[
            Leg(type="call", position="short", strike=100, premium=5),
            Leg(type="put", position="short", strike=100, premium=5)
        ],
        max_profit="Total premiums received ($10)",
        max_loss="Unlimited",
        good="Neutral market; expect low volatility; stock stays near strike.",
        bad="Large price movement; high volatility; unlimited risk."
    ),
    Strategy(
        name="Long Strangle",
        short_desc="Cheaper volatility play",
        description="Buy a call and put at different strikes. Cheaper than straddle, needs larger move to profit.",
        legs=[
            Leg(type="call", position="long", strike=110, premium=3),
            Leg(type="put", position="long", strike=90, premium=3)
        ],
        max_profit="Unlimited",
        max_loss="Total premiums paid ($6)",
        good="Expect very large price movement; lower cost than straddle.",
        bad="Flat or small price movement; low volatility; premium costs."
    ),
    Strategy(
        name="Short Strangle",
        short_desc="Wider range income",
        description="Sell a call and put at different strikes. Higher profit range than short straddle.",
        legs=[
            Leg(type="call", position="short", strike=110, premium=3),
            Leg(type="put", position="short", strike=90, premium=3)
        ],
        max_profit="Total premiums received ($6)",
        max_loss="Unlimited",
        good="Neutral market; wider range for profit than short straddle; low volatility.",
        bad="Large price movement; high volatility; unlimited risk."
    ),
    Strategy(
        name="Iron Condor",
        short_desc="Range-bound income",
        description="Sell a strangle and buy a wider strangle for protection. Profits from low volatility with defined risk.",
        legs=[
            Leg(type="put", position="long", strike=85, premium=1),
            Leg(type="put", position="short", strike=90, premium=3),
            Leg(type="call", position="short", strike=110, premium=3),
            Leg(type="call", position="long", strike=115, premium=1)
        ],
        max_profit="Net premiums received ($4)",
        max_loss="Strike Width - Net Premium ($1)",
        good="Neutral market; expect stock to stay within a range; defined risk.",
        bad="Large price movement; high volatility; breakout beyond strikes."
    ),
    Strategy(
        name="Butterfly Spread (Call)",
        short_desc="Pinpoint neutral",
        description="Buy 1 lower strike call, sell 2 middle strike calls, buy 1 higher strike call. Profits from low volatility.",
        legs=[
            Leg(type="call", position="long", strike=90, premium=12),
            Leg(type="call", position="short", strike=100, premium=6),
            Leg(type="call", position="short", strike=100, premium=6),
            Leg(type="call", position="long", strike=110, premium=3)
        ],
        max_profit="Middle Strike - Lower Strike - Net Premium ($7)",
        max_loss="Net premium paid ($3)",
        good="Neutral market; expect stock to stay near middle strike; low cost.",
        bad="Large price movement; high volatility; limited profit potential."
    ),
    Strategy(
        name="Iron Butterfly",
        short_desc="Ultra-low volatility",
        description="Sell a straddle and buy a strangle for protection. Profits from very low volatility with defined risk.",
        legs=[
            Leg(type="put", position="long", strike=90, premium=2),
            Leg(type="put", position="short", strike=100, premium=5),
            Leg(type="call", position="short", strike=100, premium=5),
            Leg(type="call", position="long", strike=110, premium=2)
        ],
        max_profit="Net premiums received ($6)",
        max_loss="Strike Width - Net Premium ($4)",
        good="Neutral market; expect minimal price movement; defined risk.",
        bad="Large price movement; high volatility; breakout beyond strikes."
    ),
    Strategy(
        name="Calendar Spread (Call)",
        short_desc="Time decay play",
        description="Sell a near-term call and buy a longer-term call at the same strike. Note: This visualization assumes same expiration for simplicity.",
        legs=[
            Leg(type="call", position="short", strike=100, premium=3),
            Leg(type="call", position="long", strike=100, premium=5)
        ],
        max_profit="Varies with time decay",
        max_loss="Net premium paid ($2)",
        good="Neutral to slightly bullish; benefit from time decay; expect stable price.",
        bad="Large price movement; high volatility; expiration mismatch complexity."
    ),
    Strategy(
        name="Collar",
        short_desc="Costless hedge",
        description="Own stock, buy protective put, sell covered call. Limits both upside and downside.",
        legs=[
            Leg(type="stock", position="long", strike=100, premium=0),
            Leg(type="put", position="long", strike=95, premium=3),
            Leg(type="call", position="short", strike=105, premium=3)
        ],
        max_profit="Call Strike - Stock Price + Net Premium ($5)",
        max_loss="Stock Price - Put Strike - Net Premium ($5)",
        good="Protect stock holdings; neutral market; generate income with downside protection.",
        bad="Strong bullish market; caps upside potential; premium costs."
    ),
]
