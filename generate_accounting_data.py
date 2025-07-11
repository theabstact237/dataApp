import pandas as pd
import numpy as np
import random
import datetime
import sqlite3
from faker import Faker

# Initialize Faker for generating realistic company and person names
fake = Faker()

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Create a class to generate accounting data
class AccountingDataGenerator:
    def __init__(self, num_records=100):
        self.num_records = num_records
        self.start_date = datetime.date(2023, 1, 1)
        self.end_date = datetime.date(2024, 5, 31)
        
        # Generate company departments
        self.departments = ['Sales', 'Marketing', 'IT', 'HR', 'Finance', 'Operations', 'R&D', 'Customer Support']
        
        # Generate expense categories
        self.expense_categories = ['Rent', 'Utilities', 'Salaries', 'Equipment', 'Software', 'Travel', 
                                  'Office Supplies', 'Insurance', 'Professional Services', 'Maintenance']
        
        # Generate revenue streams
        self.revenue_streams = ['Product Sales', 'Service Contracts', 'Consulting', 'Licensing', 
                               'Subscription Fees', 'Maintenance Contracts']
        
        # Generate contractor types
        self.contractor_types = ['IT Consultant', 'Marketing Agency', 'Legal Services', 'Accounting Services',
                                'Cleaning Services', 'Security Services', 'Temporary Staff']
        
        # Generate payment terms
        self.payment_terms = ['Net 30', 'Net 60', 'Net 90', 'Due on Receipt', 'Net 15', 'Net 45']
        
        # Generate payment status
        self.payment_status = ['Paid', 'Pending', 'Overdue', 'Partially Paid', 'Disputed']
        
        # Generate customer and vendor names
        self.customers = [fake.company() for _ in range(30)]
        self.vendors = [fake.company() for _ in range(20)]
        self.contractors = [fake.company() for _ in range(15)]
        
        # Generate employee names
        self.employees = [fake.name() for _ in range(50)]
    
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
            due_date = invoice_date + datetime.timedelta(days=random.choice([15, 30, 45, 60]))
            
            # Some invoices might be overdue
            if due_date < datetime.date.today() and random.random() < 0.3:
                payment_status = 'Overdue'
            else:
                payment_status = random.choice(self.payment_status)
            
            # Calculate amount paid based on status
            amount = round(random.uniform(1000, 50000), 2)
            if payment_status == 'Paid':
                amount_paid = amount
            elif payment_status == 'Partially Paid':
                amount_paid = round(amount * random.uniform(0.1, 0.9), 2)
            elif payment_status == 'Overdue' and random.random() < 0.4:
                amount_paid = round(amount * random.uniform(0, 0.5), 2)
            else:
                amount_paid = 0
            
            data.append({
                'InvoiceID': f'INV-{2023000 + i}',
                'CustomerID': f'CUST-{random.randint(1000, 9999)}',
                'CustomerName': random.choice(self.customers),
                'InvoiceDate': invoice_date.strftime('%Y-%m-%d'),
                'DueDate': due_date.strftime('%Y-%m-%d'),
                'Amount': amount,
                'AmountPaid': amount_paid,
                'Balance': round(amount - amount_paid, 2),
                'PaymentTerms': random.choice(self.payment_terms),
                'PaymentStatus': payment_status,
                'Department': random.choice(self.departments),
                'Notes': '' if random.random() < 0.7 else fake.sentence()
            })
        
        return pd.DataFrame(data)
    
    def generate_fixed_costs(self):
        """Generate fixed costs data"""
        data = []
        
        # Generate monthly fixed costs for the past year
        current_date = self.start_date
        
        while current_date <= self.end_date:
            month_year = current_date.strftime('%Y-%m')
            
            # Generate fixed costs for each category
            for category in self.expense_categories:
                # Base amount for the category
                base_amount = random.uniform(5000, 50000)
                
                # Add some variation month to month (±10%)
                variation = random.uniform(0.9, 1.1)
                
                data.append({
                    'ExpenseID': f'EXP-{len(data) + 1000}',
                    'Month': current_date.strftime('%Y-%m'),
                    'Category': category,
                    'Amount': round(base_amount * variation, 2),
                    'Department': random.choice(self.departments),
                    'IsRecurring': random.choice([True, True, True, False]),  # Most fixed costs are recurring
                    'PaymentDate': (current_date + datetime.timedelta(days=random.randint(1, 28))).strftime('%Y-%m-%d'),
                    'VendorID': f'VEN-{random.randint(1000, 9999)}',
                    'VendorName': random.choice(self.vendors),
                    'Notes': '' if random.random() < 0.8 else fake.sentence()
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
                base_amount = random.uniform(20000, 200000)
                
                # Add seasonal variation
                month = current_date.month
                if month in [11, 12]:  # Holiday season boost
                    seasonal_factor = random.uniform(1.2, 1.5)
                elif month in [1, 2]:  # Post-holiday slump
                    seasonal_factor = random.uniform(0.7, 0.9)
                elif month in [6, 7, 8]:  # Summer variation
                    seasonal_factor = random.uniform(0.8, 1.2)
                else:
                    seasonal_factor = random.uniform(0.9, 1.1)
                
                # Add some random transactions for this stream in this month
                num_transactions = random.randint(3, 15)
                for _ in range(num_transactions):
                    transaction_date = datetime.date(
                        current_date.year, 
                        current_date.month, 
                        random.randint(1, 28)
                    )
                    
                    # Transaction amount with some variation
                    amount = round(base_amount * seasonal_factor / num_transactions * random.uniform(0.7, 1.3), 2)
                    
                    data.append({
                        'TransactionID': f'REV-{len(data) + 5000}',
                        'Date': transaction_date.strftime('%Y-%m-%d'),
                        'Month': current_date.strftime('%Y-%m'),
                        'RevenueStream': stream,
                        'Amount': amount,
                        'CustomerID': f'CUST-{random.randint(1000, 9999)}',
                        'CustomerName': random.choice(self.customers),
                        'Department': random.choice(['Sales', 'Marketing']),
                        'Notes': '' if random.random() < 0.9 else fake.sentence()
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
            contract_start = payment_date - datetime.timedelta(days=random.randint(30, 180))
            contract_end = payment_date + datetime.timedelta(days=random.randint(30, 365))
            
            hourly_rate = round(random.uniform(50, 200), 2)
            hours_worked = random.randint(10, 160)
            
            data.append({
                'ContractID': f'CON-{3000 + i}',
                'ContractorID': f'CONT-{random.randint(1000, 9999)}',
                'ContractorName': random.choice(self.contractors),
                'ContractorType': random.choice(self.contractor_types),
                'ContractStartDate': contract_start.strftime('%Y-%m-%d'),
                'ContractEndDate': contract_end.strftime('%Y-%m-%d'),
                'PaymentDate': payment_date.strftime('%Y-%m-%d'),
                'HourlyRate': hourly_rate,
                'HoursWorked': hours_worked,
                'TotalPayment': round(hourly_rate * hours_worked, 2),
                'Department': random.choice(self.departments),
                'ProjectID': f'PROJ-{random.randint(100, 999)}',
                'PaymentStatus': random.choice(self.payment_status),
                'Notes': '' if random.random() < 0.7 else fake.sentence()
            })
        
        return pd.DataFrame(data)
    
    def generate_employees(self):
        """Generate employee data"""
        data = []
        
        for i, name in enumerate(self.employees):
            hire_date = self.random_date() - datetime.timedelta(days=random.randint(0, 1825))  # Up to 5 years ago
            
            data.append({
                'EmployeeID': f'EMP-{1000 + i}',
                'Name': name,
                'Department': random.choice(self.departments),
                'Position': random.choice(['Manager', 'Director', 'Associate', 'Specialist', 'Analyst', 'Coordinator']),
                'HireDate': hire_date.strftime('%Y-%m-%d'),
                'Salary': round(random.uniform(40000, 150000), 2),
                'BonusPercentage': round(random.uniform(0, 0.2), 2),
                'IsActive': random.random() < 0.9,  # 90% are active
                'ManagerID': f'EMP-{random.randint(1000, 1000 + len(self.employees) - 1)}' if random.random() < 0.8 else None,
                'Email': f"{name.replace(' ', '.').lower()}@acmecorp.com",
                'Phone': f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
            })
        
        return pd.DataFrame(data)
    
    def generate_all_data(self):
        """Generate all accounting datasets"""
        accounts_receivable = self.generate_accounts_receivable()
        fixed_costs = self.generate_fixed_costs()
        revenue = self.generate_revenue()
        contractors = self.generate_contractors()
        employees = self.generate_employees()
        
        return {
            'accounts_receivable': accounts_receivable,
            'fixed_costs': fixed_costs,
            'revenue': revenue,
            'contractors': contractors,
            'employees': employees
        }
    
    def export_to_excel(self, filename='accounting_data.xlsx'):
        """Export all data to Excel file with multiple sheets"""
        data_dict = self.generate_all_data()
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"Data exported to Excel: {filename}")
        return filename
    
    def export_to_csv(self, base_filename='accounting_data'):
        """Export all data to separate CSV files"""
        data_dict = self.generate_all_data()
        filenames = {}
        
        for name, df in data_dict.items():
            filename = f"{base_filename}_{name}.csv"
            df.to_csv(filename, index=False)
            filenames[name] = filename
            print(f"Data exported to CSV: {filename}")
        
        return filenames
    
    def export_to_sqlite(self, filename='accounting_data.db'):
        """Export all data to SQLite database"""
        data_dict = self.generate_all_data()
        
        conn = sqlite3.connect(filename)
        
        for table_name, df in data_dict.items():
            df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        conn.close()
        print(f"Data exported to SQLite: {filename}")
        return filename

# Generate the data
if __name__ == "__main__":
    generator = AccountingDataGenerator(num_records=150)
    
    # Export to all formats
    excel_file = generator.export_to_excel('/home/ubuntu/data_analytics_app/accounting_data.xlsx')
    csv_files = generator.export_to_csv('/home/ubuntu/data_analytics_app/accounting_data')
    sqlite_file = generator.export_to_sqlite('/home/ubuntu/data_analytics_app/accounting_data.db')
    
    print("\nAll data files generated successfully!")
