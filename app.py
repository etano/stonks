import logging
import pandas as pd
import robin_stocks as r
import sys
from dynaconf import settings


log = logging.getLogger(__name__)
format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=format, stream=sys.stdout)


def get_orders() -> pd.DataFrame:
    orders = r.get_all_stock_orders()
    df = pd.DataFrame([{
        "shares": float(order["cumulative_quantity"]),
        "amount": float(order["average_price"]) * float(order["cumulative_quantity"]),
        "symbol": r.get_instrument_by_url(order["instrument"])["symbol"],
        "direction": 1 if order["side"] == "buy" else -1,
        "timestamp": pd.to_datetime(order["last_transaction_at"])
    } for order in orders if order["state"] == "filled"])
    df.fillna(0, inplace=True)
    df["shares"] *= df["direction"]
    df["amount"] *= df["direction"]
    log.debug(f"order history:\n{df}")
    return df[["shares", "amount", "symbol", "timestamp"]]


def get_transfers() -> pd.DataFrame:
    transfers = r.get_bank_transfers()
    df = pd.DataFrame([{
        "amount": float(transfer["amount"]),
        "direction": 1 if transfer["direction"] == "deposit" else -1,
        "timestamp": pd.to_datetime(transfer["updated_at"])
    } for transfer in transfers if transfer["state"] == "completed"])
    df.fillna(0, inplace=True)
    df["amount"] *= df["direction"]
    log.debug(f"bank transfers:\n{df}")
    return df[["amount", "timestamp"]]


def get_price_by_symbol(symbol: str) -> float:
    prices = r.get_latest_price([symbol])
    return float(prices[0])


def compute_performance(orders: pd.DataFrame) -> pd.DataFrame:
    df = orders.groupby("symbol").sum()[["shares", "amount"]].reset_index()
    df["current_price"] = df["symbol"].apply(lambda x: get_price_by_symbol(x))
    df["current_value"] = df["shares"] * df["current_price"]
    df["earnings"] = df["current_value"] - df["amount"]
    return df


if __name__ == "__main__":
    r.login(username=settings.USERNAME, password=settings.PASSWORD)

    orders = get_orders()
    transfers = get_transfers()
    performance = compute_performance(orders)
    log.info(performance)

    holdings = performance.current_value.sum()
    earnings = performance.earnings.sum()
    deposits = transfers.amount.sum()
    cash = deposits - performance.amount.sum()
    assert((cash + holdings - deposits) - earnings < 1e-2)
    log.info(f"holdings: {holdings}, earnings: {earnings}, cash: {cash}")
