# My Personal Robo Advisor

## Setup
This project is a single repository with multiple sub modules. Some code is to be shared and is found in the ```shared/``` directory.

### virtualenv
Use virtualenv to create a specific environment with Python3.x:

`> virtualenv -p /usr/bin/python3 venv   # Creates a venv subdirectory`
`> source venv/bin/activate              # Activate this virual environment`


## Modules
----

### Analysis

This is the data collection module which retrieves the data for a given universe (a set of ISIN for a trading place). It determines weights for different strategies, e.g. the GMV portfolio.

----

### Rebalancing

This module determines the actual and target allocation for assets in a portfolio. It returns a set of orders which can be immediately executed.

----

### Momentum

This module determines the momentum for the assets in a portfolio. It returns a set of orders which can be executed immediately.

----

### Crash Predictor

This module predicts the potential of a crash. In case of a cash scenario this can halt the robo advisor from further execution and inform the user.

----

### Trading

This module takes a set of orders and executes them with a bank.

----

### Config

Configuration and data for the application(s).

----

### Shared

Common code to be used across all modules.