from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.requests import Request
from starlette.routing import Route
import uvicorn

# Additional imports
from collections import namedtuple
import pandas as pd
from time import perf_counter

    # TODO:
    # 1) Return open, high, low & close prices for the requested symbol as json records
    # 2) Allow calling app to filter the data by year using an optional query parameter

# Note - There are three versions of price_data implemented here
# accessible through the URLs:
# /nifty/stocks/{symbol}
# /nifty1/stocks/{symbol}
# /nifty2/stocks/{symbol}

# Simple formatting func
def rec_format(val):
    try:
        return float(val)
    except:
        try:
            # Date format as per example
            return '/'.join(val.split('-')[::-1])
        except:
            return val

# Simple timer decorator to compare speeds of each price_data function
def timer_fact(n):
    def simple_timer(fn):
        async def timed(*args, **kwargs):
            time_elapsed = 0
            for _ in range(n):
                start = perf_counter()
                result = await fn(*args, **kwargs)
                end = perf_counter()
                time_elapsed += end - start
            print(f'Func time: {time_elapsed}')
            return await fn(*args, **kwargs)
        return timed
    return simple_timer


# Version 0 - Basic - with no additional imports
# @timer_fact(10)
async def price_data(request: Request) -> JSONResponse:
    """
    Return price data for the requested symbol
    """
    symbol = request.path_params['symbol']
    year_query = request.query_params.get('year')
    # TODO - should do some validation on the year query param

    # Flag for symbol presence in data set
    symbol_present = False

    with open('data/nifty50_all.csv') as data:
        output = []
        header = next(data).strip('\n').split(',')
        index_keys = [(key.lower(), header.index(key)) for key in ['Date', 'Open', 'High', 'Low', 'Close']]
        for line in data:
            if symbol.upper() in line:
                symbol_present = True
                insert = line.strip('\n').split(',')
                output.append({
                key_name : rec_format(insert[idx]) for key_name, idx in index_keys
                })

    if not symbol_present:
        return JSONResponse({"Error": f"Symbol {symbol} not present in data set"},
                                status_code=400)

    # Crude sort option by date
    output = sorted(output, key=lambda entry: [int(x) for x in entry['date'].split('/')[::-1]], reverse=True)

    # year querystring
    if year_query:
        output = list(filter(lambda x: x['date'].split('/')[-1] == year_query, output))
        if not output:
            return JSONResponse(output) # Return empty list

    return JSONResponse(output)



# Version 1 - namedtuple
# @timer_fact(10)
async def price_data_1(request: Request) -> JSONResponse:
    """
    Return price data for the requested symbol
    """
    symbol = request.path_params['symbol']
    year_query = request.query_params.get('year')

    # Flag for symbol presence in data set
    symbol_present = False

    with open('data/nifty50_all.csv') as data:
        output = []
        # header row formatted for namedtuple construction
        header = next(data).strip('\n').lower().replace(' ','_').replace(',',' ').replace('%','')
        Record = namedtuple('Record', header)
        for line in data:
            if symbol.upper() in line:
                symbol_present = True
                line = line.strip('\n').split(',')
                record = Record(*line)
                if (year_query and year_query in record.date) or not year_query:
                    output.append({
                        # "symbol": record.symbol,
                        "date": rec_format(record.date),
                        "open": rec_format(record.open),
                        "high": rec_format(record.high),
                        "low": rec_format(record.low),
                        "close": rec_format(record.close),
                    })

    if not symbol_present:
        return JSONResponse({"Error": f"Symbol {symbol} not present in data set"},
                                status_code=400)
    if year_query and not output:
        return JSONResponse(output) # Return empty list

    # Crude sort option by date
    output = sorted(output, key=lambda entry: [int(x) for x in entry['date'].split('/')[::-1]], reverse=True)

    return JSONResponse(output)



# Version 2 - Pandas (for an extremely slow option)
# @timer_fact(10)
async def price_data_2(request: Request) -> JSONResponse:
    """
    Return price data for the requested symbol
    """
    symbol = request.path_params['symbol']
    year_query = request.query_params.get('year')
    # Additonal year_query validation to avoid error from date based slicing below
    if year_query and (not 1800 < int(year_query) < 2100):
        return JSONResponse({"Error": f"Year query {year_query} out of range"},
                                status_code=400)

    df = pd.read_csv('data/nifty50_all.csv', index_col='Date')
    df.index = pd.to_datetime(df.index)
    if symbol and year_query:
        df = df[df['Symbol'] == symbol.upper()].sort_index(ascending=False).loc[f'{year_query}-01-01':f'{year_query}-12-31']
    elif symbol:
        df = df[df['Symbol'] == symbol.upper()].sort_index(ascending=False)
        if df.empty:
            return JSONResponse({"Error": f"Symbol {symbol} not present in data set"},
                                    status_code=400)

    output = []
    for idx, row in df.iterrows():
        output.append({
            # "symbol": record.symbol,
            "date": rec_format(str(idx.date())),
            "open": rec_format(row.Open),
            "high": rec_format(row.High),
            "low": rec_format(row.Low),
            "close": rec_format(row.Close),
        })

    return JSONResponse(output)


# URL routes
app = Starlette(debug=True, routes=[
    Route('/nifty/stocks/{symbol}', price_data), # basic
    Route('/nifty1/stocks/{symbol}', price_data_1), # namedtuple
    Route('/nifty2/stocks/{symbol}', price_data_2), # pandas (slow)
])


def main() -> None:
    """
    start the server
    """
    uvicorn.run(app, host='0.0.0.0', port=8008) # 8888


# Entry point
main()
