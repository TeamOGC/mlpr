from ogc import utilities
from time import sleep, time

def callback(first, caz):
    sleep(caz)
    print(f"{first=} {caz=}")

def main():
    args = [("First Argument", 1), ("Second Argument", 2), ("Third Argument", 3)]
    caz = [("First Caz", 1), ("Second Caz", 2), ("Third Caz", 3)]
    utilities.grid_search(callback, args, caz)


if __name__ == "__main__":
    start = time()
    main()
    print(f"Time elapsed: {time() - start} seconds")