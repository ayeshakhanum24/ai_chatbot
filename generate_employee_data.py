import pandas as pd
import random
from faker import Faker

# Initialize Faker for generating random data
fake = Faker()

# Seed for reproducibility
Faker.seed(0)
random.seed(0)

# Number of records you want to generate
NUM_EMPLOYEES = 100

# Departments, positions, and genders
departments = ['IT', 'HR', 'Sales', 'Marketing', 'Finance']
positions = ['Manager', 'Analyst', 'Developer', 'Consultant', 'Executive']
genders = ['Male', 'Female']

# Generate fake employee data
employees = []
for _ in range(NUM_EMPLOYEES):
    employee = {
        'employee_id': fake.unique.random_number(digits=5),
        'name': fake.name(),
        'department': random.choice(departments),
        'position': random.choice(positions),
        'salary': random.randint(50000, 150000),
        'years_experience': random.randint(1, 30),
        'email': fake.email(),
        'contact_number': fake.phone_number(),
        'join_date': fake.date_between(start_date='-10y', end_date='today'),
        'manager': fake.name(),
        'performance_rating': random.randint(1, 5),
        'leaves_taken': random.randint(0, 30),
        'gender': random.choice(genders),
        'bonuses': [random.randint(1000, 5000) for _ in range(random.randint(0, 5))],
        'promotions': [fake.job() for _ in range(random.randint(0, 3))]
    }
    employees.append(employee)

# Create a DataFrame and save to CSV
employee_df = pd.DataFrame(employees)
employee_df.to_csv('employee_data.csv', index=False)

print("Employee data generated and saved as employee_data.csv")
