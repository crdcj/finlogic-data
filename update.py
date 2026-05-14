import time

from src import p1_cvm as p1
from src import p2_financials as p2
from src import p3_b3 as p3


def main() -> None:
    print("Starting FinLogic data update")

    t0 = time.time()
    print("p1_cvm starting")
    p1.run()
    t1 = time.time()
    print(f"p1_cvm took {t1 - t0:.2f} seconds")

    print("p2_financials starting")
    p2.run()
    t2 = time.time()
    print(f"p2_financials took {t2 - t1:.2f} seconds")

    print("p3_b3 starting")
    p3.run()
    t3 = time.time()
    print(f"p3_b3 took {t3 - t2:.2f} seconds")

    print(f"FinLogic data update complete in {t3 - t0:.2f} seconds")


if __name__ == "__main__":
    main()
