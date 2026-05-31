"""
SMOKE TEST UTILITY ONLY.
This script generates a small mock dataset (10,000 rows, 150 fraud cases) designed specifically for local smoke testing
and pipeline verification in CI/CD or sandboxed environments.
WARNING: This mock dataset is NOT used to generate the final reported metrics, charts, or slides. The final reported metrics
were computed on a stratified 200,000-row sample of the actual, real PaySim synthetic transaction dataset.
"""

import csv
import random
from pathlib import Path

def generate_mock():
    print("------------------------------------------------------------------------")
    print("NOTICE: Generating a MOCK smoke-test dataset for validation purposes.")
    print("This is NOT the real PaySim dataset used in the final analysis report.")
    print("------------------------------------------------------------------------")
    
    # Setup seed
    random.seed(42)
    n_rows = 10000
    n_fraud = 150

    # We want to distribute the 150 fraud cases randomly across 10,000 rows
    fraud_indices = set(random.sample(range(n_rows), n_fraud))

    # Column header: step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,isFraud
    rows = []
    
    types_pool = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

    for i in range(n_rows):
        step = random.randint(1, 100)
        
        if i in fraud_indices:
            # Fraud transaction
            is_fraud = 1
            transaction_type = random.choice(["TRANSFER", "CASH_OUT"])
            amount = round(random.uniform(100000.0, 500000.0), 2)
            old_balance_org = amount
            new_balance_orig = 0.0
            old_balance_dest = round(random.uniform(0.0, 100000.0), 2)
            new_balance_dest = round(old_balance_dest + amount, 2)
        else:
            # Normal transaction
            is_fraud = 0
            transaction_type = random.choice(types_pool)
            # Exponential-like distribution for amount
            amount = round(random.gammavariate(1, 20000) + 5.0, 2)
            old_balance_org = round(random.gammavariate(1, 100000), 2)
            new_balance_orig = round(max(0.0, old_balance_org - amount), 2)
            old_balance_dest = round(random.gammavariate(1, 100000), 2)
            new_balance_dest = round(old_balance_dest + amount, 2)

        rows.append([
            step,
            transaction_type,
            amount,
            old_balance_org,
            new_balance_orig,
            old_balance_dest,
            new_balance_dest,
            is_fraud
        ])

    # Make sure path exists
    workspace_root = Path(__file__).resolve().parents[1]
    raw_dir = workspace_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = raw_dir / "paysim.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"])
        writer.writerows(rows)

    print(f"SUCCESS: Generated smoke-test dataset at {csv_path}")
    print("10,000 rows generated (150 fraud transactions).")

if __name__ == "__main__":
    generate_mock()
