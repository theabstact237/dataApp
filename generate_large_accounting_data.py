import pandas as pd
import numpy as np
import random
import datetime
import sqlite3
from faker import Faker
import argparse

# Initialize Faker for generating realistic company and person names
fake = Faker()

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Create a class to generate accounting data
class AccountingDataGenerator:
    def __init__(self, num_records=10000): # Increased default number of records
        self.num_records = num_records
        self.start_date = datetime.date(2020, 1, 1) # Extended date range
        self.end_date = datetime.date(2024, 5, 31)
        
        # Generate company departments
        self.departments = [fake.job().split(",")[0] for _ in range(15)] # More diverse departments
        
        # Generate expense categories
        self.expense_categories = [fake.bs() for _ in range(20)] # More diverse expense categories
        
        # Generate revenue streams
        self.revenue_streams = [fake.catch_phrase() for _ in range(15)] # More diverse revenue streams
        
        # Generate contractor types
        self.contractor_types = [fake.job() for _ in range(20)] # More diverse contractor types
        
        # Generate payment terms
        self.payment_terms = [f"Net {random.choice([7, 15, 30, 45, 60, 90])}", "Due on Receipt", "2/10 Net 30"]
        
        # Generate payment status
        self.payment_status = ["Paid", "Pending", "Overdue", "Partially Paid", "Disputed", "Cancelled"]
        
        # Generate customer and vendor names
        self.customers = [fake.company() for _ in range(100)] # More customers
        self.vendors = [fake.company() for _ in range(50)] # More vendors
        self.contractors = [fake.company() for _ in range(30)] # More contractors
        
        # Generate employee names
        self.employees = [fake.name() for _ in range(200)] # More employees
    
    def random_date(self):
        """Generate a random date between start_date and end_date"""
        days_between = (self.end_date - self.start_date).days
        random_days = random.randint(0, days_between)
        return self.start_date + datetime.timedelta(days=random_days)
    
    def generate_accounts_receivable(self):
        """Generate accounts receivable data"""
        data = []
        
        for i in range(self.num_records):
            invoice_date = self.random_date()
            due_date = invoice_date + datetime.timedelta(days=random.choice([7, 15, 30, 45, 60, 90]))
            
            # Some invoices might be overdue
            if due_date < datetime.date.today() and random.random() < 0.3:
                payment_status = "Overdue"
            else:
                payment_status = random.choice(self.payment_status)
            
            # Calculate amount paid based on status
            amount = round(random.uniform(500, 100000), 2) # Wider range
            if payment_status == "Paid":
                amount_paid = amount
            elif payment_status == "Partially Paid":
                amount_paid = round(amount * random.uniform(0.1, 0.9), 2)
            elif payment_status == "Overdue" and random.random() < 0.4:
                amount_paid = round(amount * random.uniform(0, 0.5), 2)
            else:
                amount_paid = 0
            
            data.append({
                "InvoiceID": f"INV-{20200000 + i}",
                "CustomerID": f"CUST-{random.randint(10000, 99999)}",
                "CustomerName": random.choice(self.customers),
                "InvoiceDate": invoice_date.strftime("%Y-%m-%d"),
                "DueDate": due_date.strftime("%Y-%m-%d"),
                "Amount": amount,
                "AmountPaid": amount_paid,
                "Balance": round(amount - amount_paid, 2),
                "PaymentTerms": random.choice(self.payment_terms),
                "PaymentStatus": payment_status,
                "Department": random.choice(self.departments),
                "Region": fake.state(), # Added column
                "SalespersonID": f"EMP-{random.randint(1000, 1000 + len(self.employees) - 1)}", # Added column
                "DiscountApplied": round(random.uniform(0, 0.15), 2) if random.random() < 0.2 else 0, # Added column
                "Notes": "" if random.random() < 0.7 else fake.sentence()
            })
        
        return pd.DataFrame(data)
    
    def generate_fixed_costs(self):
        """Generate fixed costs data"""
        data = []
        
        # Generate monthly fixed costs for the date range
        current_date = self.start_date
        
        while current_date <= self.end_date:
            month_year = current_date.strftime("%Y-%m")
            
            # Generate fixed costs for each category
            for category in self.expense_categories:
                # Base amount for the category
                base_amount = random.uniform(1000, 75000) # Wider range
                
                # Add some variation month to month (±15%)
                variation = random.uniform(0.85, 1.15)
                
                data.append({
                    "ExpenseID": f"EXP-{len(data) + 10000}",
                    "Month": current_date.strftime("%Y-%m"),
                    "Category": category,
                    "Amount": round(base_amount * variation, 2),
                    "Department": random.choice(self.departments),
                    "IsRecurring": random.choice([True, True, True, False]),
                    "PaymentDate": (current_date + datetime.timedelta(days=random.randint(1, 28))).strftime("%Y-%m-%d"),
                    "VendorID": f"VEN-{random.randint(10000, 99999)}",
                    "VendorName": random.choice(self.vendors),
                    "Location": fake.city(), # Added column
                    "ApprovalStatus": random.choice(["Approved", "Pending", "Rejected"]), # Added column
                    "BudgetCode": f"BC-{random.randint(100, 999)}-{random.choice(['OPEX', 'CAPEX'])}", # Added column
                    "Notes": "" if random.random() < 0.8 else fake.sentence()
                })
            
            # Move to next month
            if current_date.month == 12:
                current_date = datetime.date(current_date.year + 1, 1, 1)
            else:
                current_date = datetime.date(current_date.year, current_date.month + 1, 1)
        
        return pd.DataFrame(data)
    
    def generate_revenue(self):
        """Generate revenue data"""
        data = []
        
        # Generate monthly revenue for each stream
        current_date = self.start_date
        
        while current_date <= self.end_date:
            # For each revenue stream
            for stream in self.revenue_streams:
                # Base amount for the stream
                base_amount = random.uniform(10000, 500000) # Wider range
                
                # Add seasonal variation
                month = current_date.month
                if month in [11, 12]: seasonal_factor = random.uniform(1.1, 1.6)
                elif month in [1, 2]: seasonal_factor = random.uniform(0.6, 0.9)
                elif month in [6, 7, 8]: seasonal_factor = random.uniform(0.7, 1.3)
                else: seasonal_factor = random.uniform(0.8, 1.2)
                
                # Add some random transactions for this stream in this month
                num_transactions = random.randint(5, 25)
                for _ in range(num_transactions):
                    transaction_date = datetime.date(
                        current_date.year, 
                        current_date.month, 
                        random.randint(1, 28)
                    )
                    
                    # Transaction amount with some variation
                    amount = round(base_amount * seasonal_factor / num_transactions * random.uniform(0.6, 1.4), 2)
                    
                    data.append({
                        "TransactionID": f"REV-{len(data) + 50000}",
                        "Date": transaction_date.strftime("%Y-%m-%d"),
                        "Month": current_date.strftime("%Y-%m"),
                        "RevenueStream": stream,
                        "Amount": amount,
                        "CustomerID": f"CUST-{random.randint(10000, 99999)}",
                        "CustomerName": random.choice(self.customers),
                        "Department": random.choice([d for d in self.departments if "Sales" in d or "Marketing" in d or "Service" in d] or self.departments),
                        "ProductCategory": fake.word().capitalize(), # Added column
                        "Channel": random.choice(["Online", "Retail", "Direct", "Partner"]), # Added column
                        "Region": fake.state(), # Added column
                        "Notes": "" if random.random() < 0.9 else fake.sentence()
                    })
            
            # Move to next month
            if current_date.month == 12:
                current_date = datetime.date(current_date.year + 1, 1, 1)
            else:
                current_date = datetime.date(current_date.year, current_date.month + 1, 1)
        
        return pd.DataFrame(data)
    
    def generate_contractors(self):
        """Generate contractor payment data"""
        data = []
        
        # Generate contractor payments
        for i in range(self.num_records):
            payment_date = self.random_date()
            contract_start = payment_date - datetime.timedelta(days=random.randint(30, 365))
            contract_end = payment_date + datetime.timedelta(days=random.randint(30, 730))
            
            hourly_rate = round(random.uniform(40, 300), 2) # Wider range
            hours_worked = random.randint(5, 200)
            
            data.append({
                "ContractID": f"CON-{30000 + i}",
                "ContractorID": f"CONT-{random.randint(10000, 99999)}",
                "ContractorName": random.choice(self.contractors),
                "ContractorType": random.choice(self.contractor_types),
                "ContractStartDate": contract_start.strftime("%Y-%m-%d"),
                "ContractEndDate": contract_end.strftime("%Y-%m-%d"),
                "PaymentDate": payment_date.strftime("%Y-%m-%d"),
                "HourlyRate": hourly_rate,
                "HoursWorked": hours_worked,
                "TotalPayment": round(hourly_rate * hours_worked, 2),
                "Department": random.choice(self.departments),
                "ProjectID": f"PROJ-{random.randint(1000, 9999)}",
                "PaymentStatus": random.choice(self.payment_status),
                "InvoiceNumber": f"CINV-{random.randint(100000, 999999)}", # Added column
                "ServiceDescription": fake.bs(), # Added column
                "ApprovedBy": random.choice(self.employees), # Added column
                "Notes": "" if random.random() < 0.7 else fake.sentence()
            })
        
        return pd.DataFrame(data)
    
    def generate_employees(self):
        """Generate employee data"""
        data = []
        
        positions = [fake.job() for _ in range(30)] # More diverse positions
        
        for i, name in enumerate(self.employees):
            hire_date = self.random_date() - datetime.timedelta(days=random.randint(0, 365*10)) # Up to 10 years ago
            
            data.append({
                "EmployeeID": f"EMP-{1000 + i}",
                "Name": name,
                "Department": random.choice(self.departments),
                "Position": random.choice(positions),
                "HireDate": hire_date.strftime("%Y-%m-%d"),
                "Salary": round(random.uniform(35000, 250000), 2), # Wider range
                "BonusPercentage": round(random.uniform(0, 0.25), 2),
                "IsActive": random.random() < 0.95, # 95% are active
                "ManagerID": f"EMP-{random.randint(1000, 1000 + len(self.employees) - 1)}" if random.random() < 0.9 else None,
                "Email": f"{name.replace(' ', '.').lower()}@{fake.domain_name()}", # More realistic domain
                "Phone": fake.phone_number(), # Realistic phone number
                "Location": fake.city(), # Added column
                "PerformanceRating": round(random.uniform(1, 5), 1) if random.random() < 0.8 else None, # Added column
                "LastReviewDate": (hire_date + datetime.timedelta(days=random.randint(180, 365*2))).strftime("%Y-%m-%d") if random.random() < 0.7 else None # Added column
            })
        
        return pd.DataFrame(data)
    
    def generate_all_data(self):
        """Generate all accounting datasets"""
        print(f"Generating {self.num_records} records for Accounts Receivable...")
        accounts_receivable = self.generate_accounts_receivable()
        print("Generating Fixed Costs data...")
        fixed_costs = self.generate_fixed_costs()
        print("Generating Revenue data...")
        revenue = self.generate_revenue()
        print(f"Generating {self.num_records} records for Contractors...")
        contractors = self.generate_contractors()
        print(f"Generating {len(self.employees)} records for Employees...")
        employees = self.generate_employees()
        
        return {
            "accounts_receivable": accounts_receivable,
            "fixed_costs": fixed_costs,
            "revenue": revenue,
            "contractors": contractors,
            "employees": employees
        }
    
    def export_to_excel(self, filename="accounting_data_large.xlsx"):
        """Export all data to Excel file with multiple sheets"""
        data_dict = self.generate_all_data()
        
        print(f"\nExporting to Excel: {filename}...")
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            for sheet_name, df in data_dict.items():
                print(f"  Writing sheet: {sheet_name} ({len(df)} rows)")
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"Data exported to Excel: {filename}")
        return filename
    
    def export_to_csv(self, base_filename="accounting_data_large"):
        """Export all data to separate CSV files"""
        data_dict = self.generate_all_data()
        filenames = {}
        
        print(f"\nExporting to CSV files with base name: {base_filename}...")
        for name, df in data_dict.items():
            filename = f"{base_filename}_{name}.csv"
            print(f"  Writing CSV: {filename} ({len(df)} rows)")
            df.to_csv(filename, index=False)
            filenames[name] = filename
            print(f"Data exported to CSV: {filename}")
        
        return filenames
    
    def export_to_sqlite(self, filename="accounting_data_large.db"):
        """Export all data to SQLite database"""
        data_dict = self.generate_all_data()
        
        print(f"\nExporting to SQLite: {filename}...")
        conn = sqlite3.connect(filename)
        
        for table_name, df in data_dict.items():
            print(f"  Writing table: {table_name} ({len(df)} rows)")
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        
        conn.close()
        print(f"Data exported to SQLite: {filename}")
        return filename

# Generate the data
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate large fake accounting data.")
    parser.add_argument("--records", type=int, default=10000, help="Number of records for AR and Contractors tables.")
    parser.add_argument("--output_dir", type=str, default="/home/ubuntu/data_analytics_app", help="Directory to save output files.")
    args = parser.parse_args()

    print(f"Starting data generation with {args.records} records...")
    generator = AccountingDataGenerator(num_records=args.records)
    
    # Define output file paths
    excel_file = f"{args.output_dir}/accounting_data_large.xlsx"
    csv_base = f"{args.output_dir}/accounting_data_large"
    sqlite_file = f"{args.output_dir}/accounting_data_large.db"
    
    # Export to all formats
    # Note: Generating all data multiple times for export, consider optimizing if performance is critical
    generator.export_to_excel(excel_file)
    generator.export_to_csv(csv_base)
    generator.export_to_sqlite(sqlite_file)
    
    print("\nAll large data files generated successfully!")
