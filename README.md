# ipushpull development test project

This project is a partially implemented web API that returns historical stock market data (open, close, high and low prices) for stocks in the [Indian Nifty50](https://www.nseindia.com/) stock index.

The project is implemented using python 3.6 and the [starlette](https://www.starlette.io/) ASGI web framework.

## Getting started
* Clone or fork this repository
* Install requirements using `pip install -r requirements.txt`
* Run the server using `python -m nifty`
* Access the endpoint at `localhost:8888/nifty/stocks/{symbol}`


## Summary of requirements


### 1) Return historical price data
Implement the `price_data` function in `__main__.py` to return **open**, **close**, **high** and **low** prices for the requested symbol as json records.

For example:

    GET /nifty/stocks/tatamotors/

should  return

```json
[
    {
        "date": "26/12/2003",
        "open": 435.8,
        "high": 440.5,
        "low": 431.65,
        "close": 438.6
    },
    {
        ...
    }
]
```

* The data is stored in the file `data/nifty50_all.csv`
* The endpoint should return one record for each row of data in the file
* Returned data should be sorted by date, most recent data first
* If an invalid symbol is requested, the endpoint should return 400 with an appropriate error message


### 2) Allow the price data to be filtered by year
Add a query parameter to the endpoint so that calling apps can request data for a single year.

For example:

    GET /nifty/stocks/tatamotors/?year=2017

* This should only return rows for the specified year (and symbol)
* If there is no data for the specified year, an empty list should be returned
* If the year is invalid, the endpoint should return 400 and an appropriate error message


## Additional information
* You should use python 3.6 or above
* You may use any appropriate open source libs as part of your solution
* Please upload your project to github.com or similar and provide a link
* If you have questions please email support@ipushpull.com

Thanks for taking our test!
