"""
Creates synthetic messy legal-style documents for testing.
Simulates: scanned noise, inconsistent formatting, partial illegibility.
"""

import os

SAMPLE_DIR = os.path.dirname(__file__)

# --- Sample 1: Lease Agreement (messy, inconsistent formatting) ---
LEASE_AGREEMENT = """
LEASE  AGREEM ENT
(Residentiai Tenancy)

Dat e: March 15,  2O24          [scan noise: date partially smudged]

PART IES:
Landlord:  Rajesh  Mehta,  S/O Suresh Mehta
Address : 42-B, Koregaon Park, Pune - 411OO1

Tenant :  Ananya  Sharma
Address:  Flat 3C,  Sunrise Apts,  Baner Rd,  Pune

PROPERT Y DETAILS:
Premises:   Flat No. 3C, 2nd Floor, Sunrise Apartments
Area:   850  sq.ft  (approx)
Monthl y Rent:  Rs. 22,000/-   (Rupees Twenty Two Thousand Only)
Security  Deposit:  Rs. 66,000/-  (Three months rent)

TENANCY  PERIOD:
Commencement Date:  April 1, 2O24
Termination Date:   March 31, 2O25
(12  months,  renewable  by  mutual  consent)

TERMS  AND  CONDITIONS:

1.  RENT PAYMENT:  Tenant  shall  pay  rent  by  5th  of  each  month.
    Late payment  attracts  penalty  of Rs. 500/- per day.

2.  MAINTENANCE:  Minor  repairs  (below  Rs. 500/-)  by  tenant.
    Major  structural  repairs  by  landlord.

3.  SUBLETTING:  Tenant  shall  NOT  sublet  or  assign  premises
    without  written  consent  of  Landlord.

4.  UTILITIES:  Electricity,  water  charges  payable  by  Tenant  directly.

5.  NOTICE  PERIOD:  Either  party  to  give  2  months  written  notice
    for  termination  before  lease  expiry.

6. LOCK-IN PERIOD: 6 months from commencement. Early vacating forfeits
   1 month security deposit.

7. PETS: No pets allowed on premises without written permission.

WITNESSES:
1. _______________ (Signature illegible)
2. _______________ (Signature illegible)

Landlord Signature: Rajesh Mehta         Tenant Signature: Ananya Sharma
Date: 15/03/2O24                          Date:  15/03/2O24

[NOTE: Page 2 of original document - water damaged, partially unreadable]
Stamp Duty Paid: Rs. 5OO/-    Registration No.: ___[illegible]___
"""

# --- Sample 2: Court Notice (noisy, handwritten annotations) ---
COURT_NOTICE = """
IN THE COURT OF CIVIL  JUDGE  (SENIOR  DIVISION)
PUNE

Case No.:  CIV/2024/1847

BETWEEN:
  Plaintiff:  M/s  TechBuild  Constructions  Pvt.  Ltd.
              Reg. Office:  Plot 12, Hinjewadi Phase 2, Pune  - 411057

      AND

  Defendant:  Mr. Vikram Nair
              Residing at:  B-204, Green Valley Society, Wakad, Pune

NOTICE  OF  HEARING

You are hereby directed to appear before this Court on:
Date:   22nd  April,  2024
Time:   10:30  A.M.
Courtroom:  No.  7,  District  Court  Complex,  Shivajinagar,  Pune

SUBJECT:  Recovery  of  dues  amounting  to  Rs.  14,75,000/-
          (Rupees Fourteen Lakhs Seventy Five Thousand Only)
          in  respect  of  construction  contract  dated  June 10, 2023.

[Handwritten annotation: "Adj. to 29 Apr - judge on leave"]

BACKGROUND:
Plaintiff  alleges  non-payment  of  final  installment  of  Rs. 14,75,000/-
due  on  completion  of  construction  at  Wakad  property.
Defendant  contends  work  was  incomplete/defective.

DOCUMENTS  TO  BE  PRODUCED:
- Original  contract  dated  June  10,  2023
- Completion  certificate  (if  any)
- Payment  receipts  / bank  statements
- Photographs  of  alleged  defects  (Defendant)

Failure  to  appear  will  result  in  ex-parte  proceedings.

Issued by:
Registrar,  Civil  Court  Pune
Date of Issue:  March  20,  2024
Seal: [SMUDGED]

[Handwritten note at bottom]: "Client informed - call  98XXXXXXXX"
"""

# --- Sample 3: Internal Memo (inconsistent formatting, partial info) ---
INTERNAL_MEMO = """
INTERNAL  MEMORANDUM
CONFIDENTIAL

TO:      Legal  Review  Team
FROM:    Priya  Kulkarni,  Head  -  Contracts
DATE:    April  3,  2024
RE:      Contract  Review  -  Vendor  Agreement  #VA-2024-089

This  memo  summarizes  key  findings  from  review  of  Vendor Agreement
VA-2024-089  between  Ambitio  Education  Pvt.  Ltd.  and  DataSoft
Solutions  Ltd.

CONTRACT  VALUE:  Rs. 48,00,000/-  per  annum
CONTRACT  PERIOD:  April  1,  2024  to  March  31,  2026  (2  years)
RENEWAL:  Auto-renewal  unless  terminated  90  days  prior

KEY  FINDINGS:

1.  PAYMENT  TERMS  [FLAG - UNFAVORABLE]:
    Clause  4.2  requires  advance  payment  of  25%  (Rs.  12,00,000/-)
    before  service  commencement.  Recommend  negotiating  to  10%.

2.  LIABILITY  CAP  [FLAG - REVIEW  NEEDED]:
    Clause  8.1  caps  vendor  liability  at  3  months  contract  value
    (Rs.  12,00,000/-).  Industry  standard  is  typically  6-12  months.

3.  DATA  PRIVACY  [OK]:
    Clause  11  includes  adequate  DPDP  Act  2023  compliance  provisions.
    Data  processing  agreement  attached  as  Annexure  C.

4.  TERMINATION  [FLAG]:
    Clause  13.4  allows  vendor  to  terminate  with  only  30  days  notice
    if  client  fails  2  consecutive  payment  cycles.  Risk:  operational
    disruption.  Recommend  extending  to  60  days.

5.  IP  OWNERSHIP  [OK]:
    Clause  15  clearly  assigns  all  custom  development  IP  to  Ambitio.

RECOMMENDED  ACTIONS:
- Negotiate  Clause  4.2  (advance  payment)
- Negotiate  Clause  8.1  (liability  cap)
- Negotiate  Clause  13.4  (termination  notice)
- Obtain  legal  sign-off  before  execution

DEADLINE:  Contract  execution  targeted  for  April  15,  2024.
           Negotiation  window:  April  3-10,  2024.

[Post-it note scanned]: "Shivam - pls  review  cls 8  urgently - PK"
"""

def save_samples():
    samples = {
        "lease_agreement.txt": LEASE_AGREEMENT,
        "court_notice.txt": COURT_NOTICE,
        "internal_memo.txt": INTERNAL_MEMO,
    }
    for filename, content in samples.items():
        path = os.path.join(SAMPLE_DIR, filename)
        with open(path, "w") as f:
            f.write(content)
    print(f"Saved {len(samples)} sample documents to {SAMPLE_DIR}/")

if __name__ == "__main__":
    save_samples()
