import subprocess
import sys
import Company_data


def write_poem():
    print("\nRoses are red,")
    print("Violets are blue,")
    print("Pipelines are running,")
    print("And so are you.\n")


def run_script(script_name):
    print(f"\nRunning {script_name}...\n")
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.stderr:
        print("ERROR:")
        print(result.stderr)


def main():

    
    # Step 2: Feature engineering
    run_script("Feature_engineering.py")

    # Step 3: Generate sentiment
    run_script("sen.py")

    # Step 7: Final aggregation
    run_script("main2.py")

    print("\n✅ Pipeline execution completed")


if __name__ == "__main__":
    Company_data.fetch_and_save_to_csv('TCS', '2y')
    Company_data.fetch_and_save_market_data('2y')
    main()
