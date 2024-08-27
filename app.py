import bcrypt
import mysql.connector
import re
from datetime import datetime
import pytz
import streamlit as st
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Connect to TiDB Cloud database
connection = mysql.connector.connect(
    host = "gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    port = 4000,
    user = "2ZCnZ2X6eMgKoxd.root",
    password = "TD67pkQaXhFl73d7",
    database = "test"
)
mycursor = connection.cursor(buffered=True)

mycursor.execute("CREATE DATABASE IF NOT EXISTS test")
mycursor.execute('USE test')

mycursor.execute('''CREATE TABLE IF NOT EXISTS users (
                user_id INT PRIMARY KEY AUTO_INCREMENT,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                registered_date TIMESTAMP,
                last_login TIMESTAMP)''')
connection.commit()

def username_exists(mycursor, username):
    mycursor.execute("SELECT COUNT(*) FROM users WHERE username = %s", (username,))
    count = mycursor.fetchone()[0]
    return count > 0
    
def email_exists(mycursor, email):
    mycursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    return mycursor.fetchone() is not None

def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def create_user(username, email, password, registered_date):
    mycursor = connection.cursor()
    if username_exists(mycursor,username):
        mycursor.close()
        connection.close()
        return 'username_exists'
    
    if email_exists(mycursor,email):
        mycursor.close()
        connection.close()
        return 'email_exists'

    hashed_password = hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    registered_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    mycursor.execute(
        "INSERT INTO users (username, email, password, registered_date) VALUES (%s, %s, %s, %s)",
        (username, email, hashed_password, registered_date)
    )
    connection.commit()
    return 'success'

def verify_users(username, password):

    mycursor.execute("SELECT password FROM users WHERE username = %s", (username,))
    record = mycursor.fetchone()
    if record:
        stored_password_hash = record[0]
        st.write(f"Stored Password Hash: {stored_password_hash}")  
        
        if bcrypt.checkpw(password.encode('utf-8'), stored_password_hash.encode('utf-8')):
            ist = pytz.timezone('Asia/Kolkata')  # Define the IST timezone
            now = datetime.now(ist)
            mycursor.execute("UPDATE users SET last_login = %s WHERE username = %s", (now, username))
            connection.commit()
            return True
        else:
            st.error("Password does not match.")  
    else:
        st.error("Username not found.")  
    
    return False

def reset_password(username, new_password):

    hash_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
    mycursor.execute("UPDATE users SET password = %s WHERE username = %s", (hash_password, username))
    connection.commit()

# Load the fine-tuned model and tokenizer
model_name_or_path = "./fine_tuned_model_New"     # Use the directory where you saved the model
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

token_name_or_path = "./fine_tuned_model_New"      # Use the directory where you saved the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

# Set the pad_token to eos_token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the text generation function
def generate_text(model, tokenizer, seed_text, max_length=100, temperature=1.0, num_return_sequences=1):
    # Tokenize the input text with padding
    inputs = tokenizer(seed_text, return_tensors='pt', padding=True, truncation=True)

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id  # Ensure padding token is set to eos_token_id
        )

    # Decode the generated text
    generated_texts = []
    for i in range(num_return_sequences):
        generated_text = tokenizer.decode(output[i], skip_special_tokens=True)
        generated_texts.append(generated_text)

    return generated_texts

# Initialize session state variables
if 'sign_up_successful' not in st.session_state:
    st.session_state.sign_up_successful = False
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'reset_password' not in st.session_state:
    st.session_state.reset_password = False
    
def home_page():
    st.title(f"Welcome, {st.session_state.username}!")
    st.write("This is home page after successful login.")
    st.warning("""
    **Disclaimer:** The information provided in this application is for educational purposes only.
    The creators of this app make no representations or warranties of any kind, express or implied,
    about the completeness, accuracy, reliability, suitability, or availability with respect to the app
    or the information contained within it.
    """)

        # Text generation section
    st.subheader("Generate Text")
    seed_text = st.text_input("Enter text:")
    max_length = st.slider("Max length", min_value=10, max_value=500, value=100)
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0)
    #num_return_sequences = st.slider("Number of sequences", min_value=1, max_value=5, value=1)

    if st.button("Generate"):
        with st.spinner("Generating..."):
            generated_texts = generate_text(model, tokenizer, seed_text, max_length, temperature=0.000001, num_return_sequences=1)
            for i, text in enumerate(generated_texts):
                st.write(f"Generated Text {i + 1}:\n{text}\n")

def login_page():

    st.subheader(':red[**Login**]')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')

    if st.button('Login'):
        if verify_users(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.page = 'home'
            st.rerun() # Rerun the app to update the UI
        elif not username or not password:
            st.error("Please fill out all fields.")
        else:
            st.error(f'username password {username} {password}Invalid username or password')
    # Login form
    if not st.session_state.logged_in:
        c1, c2 = st.columns(2)
        with c1:
            st.write(":blue[New User]")
            if st.button('Click on Sign Up'):
                st.session_state.page = 'sign_up'
                st.rerun()
        with c2:
            st.write(":blue[Forgot Password]")
            if st.button('Reset Password'):
                st.session_state.page = 'reset_password'
                st.rerun()

def signup_page():

    st.subheader(':red[**Sign Up**]')
    username = st.text_input('Username')
    email = st.text_input('Email')
    password = st.text_input('Password', type='password')
    confirm_password = st.text_input('Confirm Password', type='password')
    registered_date = datetime.now(pytz.timezone('Asia/Kolkata'))

    if st.button('Sign Up'):
        if password != confirm_password:
            st.error('Passwords do not match.')
        elif not username or not email or not password or not confirm_password:
            st.error('Please fill out all required fields.')
        elif not is_valid_email(email):
            st.error("Please enter a valid email address.")
        elif len(password) <= 5:
            st.error("Password too short")
        else:
            result = create_user(username, email, password, registered_date)
            if result == "success":
                st.success(f"Username {username} created successfully! Please Login.")
                st.session_state.sign_up_successful = True
            elif result =="username_exists":
                st.error("This username is already registered.Please use different username.")
            elif result =="email_exists":
                st.error("This Email is already registered.Please use different email.")
            else:
                st.error("Failed to create user.Please try after sometime.")

            
    if st.button('Back to Login'):
        st.session_state.page = 'login'
        st.rerun()

def reset_password_page():

    st.header(':violet[Reset Password]')
    username = st.text_input('Username')
    new_password = st.text_input('New Password', type='password')
    confirm_password = st.text_input('Confirm New Password', type='password')

    if st.button('Reset Password'):
        if not username:
                st.error("Please enter your username.")
        elif new_password != confirm_password:
            st.error('Passwords do not match.')
        elif not username_exists(mycursor,username):
            st.error("Username not found. Please enter a valid username.")
        elif len(new_password) <= 5:
            st.error("Password too short")
        elif not new_password or not confirm_password:
            st.error('Please fill out all required fields.')
        else:
            reset_password(username, new_password)
            st.success("Password reset successfully. Please login with your new password.")
            st.session_state.page = 'login'

    
    if st.button('Back to Login'):
        st.session_state.page = 'login'
        st.rerun()

# Display the appropriate page based on session state
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'login':
    login_page()
elif st.session_state.page == 'sign_up':
    signup_page()
elif st.session_state.page == 'reset_password':
    reset_password_page()