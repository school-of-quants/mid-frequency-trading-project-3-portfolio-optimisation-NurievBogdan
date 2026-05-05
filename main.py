
from equity_project.src.get_data import get_data
from equity_project.src.run_backtest import run_backtest
from equity_project.src.train import train


def main() -> None:
    get_data()
    train()
    run_backtest()


if __name__ == "__main__":
    main()
