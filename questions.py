import pandas as pd

# Load the employee data from the CSV file
data = pd.read_csv('employee_data.csv')

# Initialize an empty list to store the questions and answers
qa_pairs = []

# General Questions and Answers
total_employees = len(data)
average_salary = data['salary'].mean()
average_experience = data['years_experience'].mean()
department_counts = data['department'].value_counts().to_dict()
gender_counts = data['gender'].value_counts().to_dict()

# Add general questions and their answers
qa_pairs.append(("What is the total number of employees?", total_employees))
qa_pairs.append(("What is the average salary of employees?", round(average_salary, 2)))
qa_pairs.append(("What is the average years of experience among employees?", round(average_experience, 2)))
qa_pairs.append(("How many employees are in each department?", department_counts))
qa_pairs.append(("What is the gender distribution of the employees?", gender_counts))

# Specific Questions about Employees
for index, row in data.iterrows():
    name = row['name']
    department = row['department']
    position = row['position']
    salary = row['salary']
    years_experience = row['years_experience']
    email = row['email']
    contact_number = row['contact_number']
    join_date = row['join_date']
    manager = row['manager']
    performance_rating = row['performance_rating']
    leaves_taken = row['leaves_taken']
    gender = row['gender']
    bonuses = row['bonuses']
    promotions = row['promotions']
    
    qa_pairs.append((f"What is the salary of {name}?", salary))
    qa_pairs.append((f"How many years of experience does {name} have?", years_experience))
    qa_pairs.append((f"What department does {name} work in?", department))
    qa_pairs.append((f"What position does {name} hold?", position))
    qa_pairs.append((f"What is {name}'s email?", email))
    qa_pairs.append((f"What is the contact number of {name}?", contact_number))
    qa_pairs.append((f"When did {name} join the company?", join_date))
    qa_pairs.append((f"Who is the manager of {name}?", manager))
    qa_pairs.append((f"What is the performance rating of {name}?", performance_rating))
    qa_pairs.append((f"How many leaves has {name} taken?", leaves_taken))
    qa_pairs.append((f"What is the gender of {name}?", gender))
    qa_pairs.append((f"What bonuses has {name} received?", bonuses))
    qa_pairs.append((f"What promotions has {name} received?", promotions))
    qa_pairs.append((f"List all details for {name}.", row.to_dict()))

# Convert the question-answer pairs into a DataFrame
qa_df = pd.DataFrame(qa_pairs, columns=['Question', 'Answer'])

# Save the question-answer dataset to a new CSV file
qa_df.to_csv('questions_answers_dataset.csv', index=False)

print("Questions and answers dataset generated successfully.")
