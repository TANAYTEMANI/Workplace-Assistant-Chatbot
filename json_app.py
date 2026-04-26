import json
import argparse
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Get GROQ configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")



with open('leavesEmployee.json', 'r') as file:
    leavesEmployee_data = json.load(file)
    
# def flatten_json_data(user):
#     flattened_data = {
#         'id': user['id'],
#         'firstName': user['firstName'],
#         'lastName': user['lastName'],
#         'maidenName': user.get('maidenName', ''),
#         'age': user['age'],
#         'gender': user['gender'],
#         'email': user['email'],
#         'phone': user['phone'],
#         'username': user['username'],
#         'password': user['password'],
#         'birthDate': user['birthDate'],
#         'image': user['image'],
#         'bloodGroup': user['bloodGroup'],
#         'height': user['height'],
#         'weight': user['weight'],
#         'eyeColor': user['eyeColor'],
#         'hairColor': user['hair']['color'],
#         'hairType': user['hair']['type'],
#         'ip': user['ip'],
#         'address': user['address']['address'],
#         'addCity': user['address']['city'],
#         'addState': user['address']['state'],
#         'addStatecode': user['address']['stateCode'],
#         'addPostalcode': user['address']['postalCode'],
#         'addCoordLat': user['address']['coordinates']['lat'],
#         'addCoordLng': user['address']['coordinates']['lng'],
#         'addCountry': user['address']['country'],
#         'macAddress': user['macAddress'],
#         'university': user['university'],
#         'cardExpire': user['bank']['cardExpire'],
#         'cardNumber': user['bank']['cardNumber'],
#         'cardType': user['bank']['cardType'],
#         'currency': user['bank']['currency'],
#         'iban': user['bank']['iban'],
#         'department': user['company']['department'],
#         'compName': user['company']['name'],
#         'compTitle': user['company']['title'],
#         'compAdd': user['company']['address']['address'],
#         'compCity': user['company']['address']['city'],
#         'compState': user['company']['address']['state'],
#         'compStateCode': user['company']['address']['stateCode'],
#         'compPostalCode': user['company']['address']['postalCode'],
#         'compCoordLat': user['company']['address']['coordinates']['lat'],
#         'compCoordLng': user['company']['address']['coordinates']['lng'],
#         'compCountry': user['company']['address']['country'],
#         'ein': user['ein'],
#         'ssn': user['ssn'],
#         'userAgent': user['userAgent'],
#         'coin': user['crypto']['coin'],
#         'wallet': user['crypto']['wallet'],
#         'network': user['crypto']['network'],
#         'role': user['role'],
#     }
#     return flattened_data

# flat_data = [flatten_json_data(user) for user in json_data['users']]

# Convert JSON to a string
flat_json_string = json.dumps(leavesEmployee_data, indent=2)

#Taking prompt from user
parser = argparse.ArgumentParser(description="Provide a prompt to query the user data.")
parser.add_argument("prompt", type=str, help="The prompt to use for querying the user data")
args = parser.parse_args()


prompt = f"Here is the data: {flat_json_string}. Respond to the user's query in the second person, addressing the user as 'you'. Do not use phrases like 'according to the json data provided'. The employee cannot take a leave if the number of the remaining leave is 0. Answer the question in 1 sentence. Now, answer the following query: {args.prompt}."

# Initialize the GROQ LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL,
    temperature=0.7,
    max_tokens=128
)

# Generate the response
response = llm.invoke(prompt)

# Extract the generated text
generated_text = response.content if hasattr(response, 'content') else str(response)
print(generated_text)