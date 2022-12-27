from alpaca.trading.models import Position, Order
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class StrategyPosition(Position):
    strategy_id: str

    @classmethod
    def from_position(cls, position: Position, strategy_id: str):
        return cls(
            strategy_id,
            **asdict(position)
        )

@dataclass
class StrategyTrade(Order):
    strategy_id: str

    @classmethod
    def from_order(cls, order: Order, strategy_id: str):
        return cls(
            strategy_id,
            **asdict(order)
        )

@dataclass
class StrategyTradeHistory:
    trade_history = List[StrategyTrade]

@dataclass
class StrategyPositionHistory:
    position_history = List[StrategyPosition]













