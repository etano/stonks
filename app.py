import dash
import dash_auth
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import logging
import pandas as pd
import plotly.graph_objects as go
import robin_stocks as r
import sys
from dash.dependencies import Input, Output
from datetime import datetime, timezone
from dynaconf import settings
from typing import Dict, List


CASH_SYMBOL = settings.CASH_SYMBOL
TRANSFER_SYMBOL = settings.TRANSFER_SYMBOL
TOTAL_SYMBOL = settings.TOTAL_SYMBOL
STARTING_DATE_RANGE = [datetime(2020, 4, 15), datetime.now()]

log = logging.getLogger(__name__)
format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=format, stream=sys.stdout)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
auth = dash_auth.BasicAuth(
    app,
    {settings.DASH_USERNAME: settings.DASH_PASSWORD}
)
app.layout = dbc.Container([
    html.H1(children='Brown Stonks'),
    dcc.Loading(
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    html.H3(id="total_value"),
                    html.P("total value")
                ], body=True)
            ),
            dbc.Col(
                dbc.Card([
                    html.H3(id="total_return"),
                    html.P("total earnings")
                ], body=True)
            ),
            dbc.Col(
                dbc.Card([
                    html.H3(id="total_return_percentage"),
                    html.P("return on investments")
                ], body=True)
            ),
            dbc.Col(
                dbc.Card([
                    html.H3(id="total_return_on_original_investment_percentage"),
                    html.P("return on original investment")
                ], body=True)
            )
        ]),
    ),
    dcc.Loading(
        id="loading_cumulative_value",
        children=[dcc.Graph(id='cumulative_value')]
    ),
    dcc.Loading(
        id="loading_cumulative_return_percentage",
        children=[dcc.Graph(id='cumulative_return')]
    ),
    dcc.Loading(
        id="loading_cumulative_return",
        children=[dcc.Graph(id='cumulative_return_percentage')]
    ),
    dcc.Interval(
        id='interval-component',
        interval=settings.REFRESH_INTERVAL_SECONDS * 1000,
        n_intervals=0
    )
])
r.login(username=settings.ROBINHOOD_USERNAME, password=settings.ROBINHOOD_PASSWORD)


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


def get_transactions(orders: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    cash_transactions = []
    for _, order in orders.iterrows():
        cash_transactions.append({
            'shares': 0,
            'amount': -order.amount,
            'symbol': CASH_SYMBOL,
            'timestamp': order.timestamp
        })
    for _, transfer in transfers.iterrows():
        cash_transactions.append({
            'shares': 0,
            'amount': transfer.amount,
            'symbol': TRANSFER_SYMBOL,
            'timestamp': transfer.timestamp
        })
    df = pd.concat([orders, pd.DataFrame(cash_transactions)]).sort_values(by=["timestamp", "symbol"]).reset_index()
    return df


def get_price_by_symbol(symbol: str) -> float:
    if symbol in [CASH_SYMBOL, TRANSFER_SYMBOL, TOTAL_SYMBOL]:
        return 0
    prices = r.get_latest_price([symbol])
    return float(prices[0])


def compute_performance(orders: pd.DataFrame) -> pd.DataFrame:
    df = orders.groupby("symbol").sum()[["shares", "amount"]].reset_index()
    df["current_price"] = df["symbol"].apply(lambda x: get_price_by_symbol(x))
    df["current_value"] = df["shares"] * df["current_price"]
    df["earnings"] = df["current_value"] - df["amount"]
    return df


def get_historicals(symbol: str) -> pd.DataFrame:
    # TODO: cache results
    df = pd.DataFrame(r.get_historicals(symbol, span='year'))
    df['timestamp'] = pd.to_datetime(df['begins_at']).dt.date
    df['price'] = df['close_price'].astype(float)
    return df[['timestamp', 'price']]


def get_cumulatives(transactions: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    dfs = dict(tuple(transactions.groupby('symbol')))
    now = datetime.now(tz=timezone.utc)
    for symbol in dfs.keys():
        log.info(f"Computing cumulatives for {symbol}...")

        # compute cumulative shares
        dfs[symbol]['timestamp'] = pd.to_datetime(dfs[symbol]['timestamp'], utc=True).dt.date
        dfs[symbol] = dfs[symbol].groupby(['timestamp']).sum()
        dfs[symbol]["cumulative_shares"] = dfs[symbol]["shares"].cumsum()
        dfs[symbol]["cumulative_value"] = dfs[symbol]["amount"].cumsum()
        dfs[symbol]["cumulative_invested"] = dfs[symbol][dfs[symbol]['shares'] >= 0]["amount"].cumsum()
        dfs[symbol]["cumulative_invested"].fillna(inplace=True, method='ffill')
        dfs[symbol]["cumulative_sold"] = -dfs[symbol][dfs[symbol]['shares'] < 0]["amount"].cumsum()

        # add today
        dfs[symbol] = pd.concat([dfs[symbol].reset_index(), pd.DataFrame([{
            'timestamp': now,
            'symbol': symbol
        }])])
        dfs[symbol]['timestamp'] = pd.to_datetime(dfs[symbol]['timestamp'], utc=True).dt.date
        dfs[symbol].fillna(inplace=True, method='ffill')
        dfs[symbol].fillna(value=0, inplace=True)

        # add historical prices
        dfs[symbol] = dfs[symbol].set_index('timestamp').asfreq('D', method='ffill').reset_index()
        if symbol not in [CASH_SYMBOL, TRANSFER_SYMBOL]:
            df_historicals = get_historicals(symbol)
            dfs[symbol]['timestamp'] = pd.to_datetime(dfs[symbol]['timestamp'], utc=True).dt.date
            dfs[symbol] = pd.merge(dfs[symbol], df_historicals, how='left', on='timestamp')
        dfs[symbol].fillna(inplace=True, method='ffill')

        # add now price
        dfs[symbol] = pd.concat([dfs[symbol], pd.DataFrame([{
            'timestamp': now,
            'symbol': symbol,
            'price': get_price_by_symbol(symbol)
        }])])
        dfs[symbol].fillna(inplace=True, method='ffill')
        dfs[symbol]['timestamp'] = pd.to_datetime(dfs[symbol]['timestamp'], utc=True)

        # compute cumulative returns
        if symbol not in [CASH_SYMBOL, TRANSFER_SYMBOL]:
            dfs[symbol]['cumulative_value'] = dfs[symbol]['cumulative_shares'] * dfs[symbol]['price']
            dfs[symbol]['cumulative_return'] = dfs[symbol]['cumulative_sold'] + dfs[symbol]['cumulative_value'] - dfs[symbol]['cumulative_invested']  # noqa
            dfs[symbol]['cumulative_return_percentage'] = 100 * dfs[symbol]['cumulative_return'] / dfs[symbol]['cumulative_invested']  # noqa
        else:
            dfs[symbol]['cumulative_return'] = 0
            dfs[symbol]['cumulative_return_percentage'] = 0

        # set index
        dfs[symbol]['timestamp'] = pd.to_datetime(dfs[symbol]['timestamp'], utc=True)
        dfs[symbol] = dfs[symbol].set_index('timestamp')

    # compute totals
    df = pd.concat([value for symbol, value in dfs.items()])
    df = df.groupby(df.index).sum()
    df['cumulative_return_percentage'] = 100 * df['cumulative_return'] / df['cumulative_invested']
    df['cumulative_return_on_original_investment_percentage'] = 100 * df['cumulative_return'] / dfs[TRANSFER_SYMBOL].cumulative_value  # noqa
    dfs[TOTAL_SYMBOL] = df

    # compute liquidity
    dfs[CASH_SYMBOL].cumulative_value += dfs[TRANSFER_SYMBOL].cumulative_value

    return dfs


def generate_figure(
        dfs: Dict[str, pd.DataFrame],
        y: str, title: str,
        x: str = None, ignored_symbols: List[str] = []
) -> go.Figure:
    fig = go.Figure()
    for symbol, df in dfs.items():
        if symbol not in ignored_symbols:
            fig.add_trace(go.Scatter(x=x or df.index, y=df[y], name=symbol))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        xaxis_range=STARTING_DATE_RANGE
    )
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=5, label="5d", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ))
    return fig


@app.callback([
    Output('total_value', 'children'),
    Output('total_return', 'children'),
    Output('total_return_percentage', 'children'),
    Output('total_return_on_original_investment_percentage', 'children'),
    Output('cumulative_value', 'figure'),
    Output('cumulative_return', 'figure'),
    Output('cumulative_return_percentage', 'figure')
], [
    Input('interval-component', 'n_intervals')
])
def update_graphs(n):
    orders = get_orders()
    transfers = get_transfers()
    transactions = get_transactions(orders, transfers)
    dfs = get_cumulatives(transactions)
    current_totals = dfs[TOTAL_SYMBOL].iloc[-1]
    total_value = f"{current_totals.cumulative_value:.2f} USD"
    total_return = f"{current_totals.cumulative_return:.2f} USD"
    total_return_percentage = f"{current_totals.cumulative_return_percentage:.2f} %"
    total_return_on_original_investment_percentage = f"{current_totals.cumulative_return_on_original_investment_percentage:.2f} %"  # noqa
    fig_cumulative_value = generate_figure(dfs, y="cumulative_value", title="Value in USD")
    fig_cumulative_return = generate_figure(dfs, y="cumulative_return", title="Return in USD", ignored_symbols=[CASH_SYMBOL, TRANSFER_SYMBOL])  # noqa
    fig_cumulative_return_percentage = generate_figure(dfs, y="cumulative_return_percentage", title="Return percentage", ignored_symbols=[CASH_SYMBOL, TRANSFER_SYMBOL])  # noqa
    return [
        total_value,
        total_return,
        total_return_percentage,
        total_return_on_original_investment_percentage,
        fig_cumulative_value,
        fig_cumulative_return,
        fig_cumulative_return_percentage
    ]


if __name__ == '__main__':
    app.run_server(debug=True)
