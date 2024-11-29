from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load pre-trained text-to-SQL model
model_name = "cssupport/t5-small-awesome-text-to-sql"  # Pre-trained text-to-SQL model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Function to generate SQL
def generate_sql(user_input: str):
    input_text = f"translate UserQuery to SQL: {user_input}"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=150)
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql_query

# Example usage
user_input = "Show me the plots available in Anna Nagar."
sql_query = generate_sql(user_input)
print("Generated SQL Query:", sql_query)
