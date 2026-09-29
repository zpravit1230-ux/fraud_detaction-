"""Generate deterministic synthetic corporate procurement data."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
REFERENCE_DATE = pd.Timestamp("2026-09-28")


def generate_demo_data(seed: int = 42, transactions_count: int = 6000) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    vendor_count, employee_count = 120, 60
    industries = ["Logistics", "Technology", "Facilities", "Consulting", "Manufacturing", "Healthcare", "Energy"]
    countries = ["US", "CA", "GB", "DE", "AU"]
    vendors = []
    for index in range(1, vendor_count + 1):
        registration = REFERENCE_DATE - pd.Timedelta(days=int(rng.integers(90, 2600)))
        vendors.append({
            "vendor_id": f"VENDOR_{index:03d}", "vendor_name": f"{rng.choice(['Northstar', 'Meridian', 'Cobalt', 'Pioneer', 'Summit', 'Cedar', 'Atlas', 'Harbor'])} {rng.choice(['Services', 'Systems', 'Partners', 'Supply', 'Group'])} {index:03d}",
            "registration_date": registration.strftime("%Y-%m-%d"), "address_id": f"ADDR_{index:03d}",
            "bank_account_id": f"BANK_{index:03d}", "industry": str(rng.choice(industries)), "country": str(rng.choice(countries)), "status": "Active",
        })
    vendors = pd.DataFrame(vendors)
    employees = pd.DataFrame([{
        "employee_id": f"EMP_{index:03d}", "employee_name": f"Employee {index:03d}", "department": str(rng.choice(["Finance", "Operations", "IT", "Procurement", "Legal"])),
        "role": str(rng.choice(["Analyst", "Manager", "Director", "Specialist"])), "email": f"employee{index:03d}@example.test", "address_id": f"EADDR_{index:03d}",
    } for index in range(1, employee_count + 1)])

    # Concentrated identity overlaps create inspectable relationship signals.
    vendor_map = vendors.set_index("vendor_id")
    vendors.loc[vendors.vendor_id.isin([f"VENDOR_{i:03d}" for i in (17, 21, 34)]), "bank_account_id"] = "BANK_SHARED_104"
    vendors.loc[vendors.vendor_id.isin([f"VENDOR_{i:03d}" for i in (17, 21, 34, 46)]), "address_id"] = "ADDR_SHARED_009"
    vendors.loc[vendors.vendor_id.isin([f"VENDOR_{i:03d}" for i in (17, 21, 34)]), "registration_date"] = (REFERENCE_DATE - pd.Timedelta(days=100)).strftime("%Y-%m-%d")
    vendor_ids = vendors.vendor_id.to_numpy()
    employee_ids = employees.employee_id.to_numpy()
    base_dates = pd.Timestamp("2025-01-01")
    transactions = []
    for index in range(1, transactions_count + 1):
        vendor_id = str(rng.choice(vendor_ids))
        vendor_row = vendors.loc[vendors.vendor_id.eq(vendor_id)].iloc[0]
        employee_id = str(rng.choice(employee_ids))
        day_offset = int(rng.integers(0, (REFERENCE_DATE - base_dates).days))
        amount = float(np.clip(rng.lognormal(mean=7.35, sigma=0.75), 250, 85000).round(2))
        # Vendor 017 receives a recent burst of high-value payments and repeats one employee.
        if vendor_id == "VENDOR_017" and day_offset > (REFERENCE_DATE - base_dates).days - 75:
            amount = round(float(rng.uniform(42000, 118000)), 2)
            employee_id = "EMP_004"
        transactions.append({
            "transaction_id": f"TXN_{index:06d}", "vendor_id": vendor_id, "employee_id": employee_id,
            "transaction_date": (base_dates + pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d"), "amount": amount,
            "payment_method": str(rng.choice(["ACH", "Wire", "Corporate card", "Check"], p=[0.62, 0.2, 0.12, 0.06])),
            "bank_account_id": vendor_row.bank_account_id, "invoice_id": f"INV_{index:06d}",
            "description": str(rng.choice(["Professional services", "Equipment purchase", "Monthly service", "Project delivery", "Maintenance"])),
        })
    transactions = pd.DataFrame(transactions)
    # Inject a few explicit but non-conclusive anomaly patterns into actual records.
    last_date = REFERENCE_DATE - pd.Timedelta(days=5)
    scenario_rows = []
    for offset in range(12):
        scenario_rows.append({
            "transaction_id": f"TXN_SCENARIO_{offset + 1:02d}", "vendor_id": "VENDOR_017", "employee_id": "EMP_004",
            "transaction_date": (last_date - pd.Timedelta(days=offset * 2)).strftime("%Y-%m-%d"),
            "amount": float(76000 + offset * 2800), "payment_method": "Wire", "bank_account_id": "BANK_SHARED_104",
            "invoice_id": f"INV_SCENARIO_{offset // 2 + 1:02d}", "description": "Project services - review requested",
        })
    transactions = pd.concat([transactions, pd.DataFrame(scenario_rows)], ignore_index=True)
    # Additional suspicious vendors share the same employee and account cluster.
    transactions.loc[transactions.vendor_id.isin(["VENDOR_021", "VENDOR_034"]), "employee_id"] = "EMP_004"

    invoice_rows = []
    transaction_view = transactions.copy()
    for row in transaction_view.itertuples(index=False):
        invoice_rows.append({
            "invoice_id": row.invoice_id, "vendor_id": row.vendor_id, "invoice_date": row.transaction_date,
            "invoice_amount": row.amount, "invoice_status": str(rng.choice(["Paid", "Approved", "Pending"], p=[0.76, 0.17, 0.07])),
            "purchase_order_id": f"PO_{int(rng.integers(10000, 99999))}",
        })
    invoices = pd.DataFrame(invoice_rows)
    # Duplicate invoice references are represented as distinct invoice records.
    duplicate_source = invoices.loc[invoices.vendor_id.eq("VENDOR_017")].head(8).copy()
    if not duplicate_source.empty:
        duplicate_source["invoice_status"] = "Pending review"
        invoices = pd.concat([invoices, duplicate_source], ignore_index=True)
    return {"vendors": vendors, "employees": employees, "transactions": transactions, "invoices": invoices}


def write_demo_data(output_dir: Path | str = DATA_DIR, seed: int = 42) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in generate_demo_data(seed=seed).items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths


if __name__ == "__main__":
    generated = write_demo_data()
    print("Generated synthetic demo data:")
    for name, path in generated.items():
        print(f"  {path.name}: {len(pd.read_csv(path)):,} rows")