import os
import re
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from uuid import uuid4
from flask import Flask, request, redirect, url_for, render_template, flash, send_file, session, jsonify
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle
import boto3

app = Flask(__name__)


UPLOAD_FOLDER = '/home/ubuntu/SalesAnalysisTool/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# AWS S3 configuration
S3_BUCKET = 'my-sales-analysis-bucket'
S3_REGION = 'ap-south-1'

# Initialize S3 client
s3 = boto3.client('s3')
# Initialize S3 client with explicit credentials


def add_table_to_story(dataframe, title, story, styles):
    story.append(Paragraph(title, styles['Heading2']))
    data = [dataframe.columns.tolist()] + dataframe.values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

def save_pie_chart(data, filename, title):
    labels = data.keys()
    sizes = data.values()
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def find_most_similar_product(new_description, new_category, df, description_column, category_column):
    # Create a new column for combined text from description and category
    df['combined_description'] = df[category_column].astype(str) + " " + df[description_column].astype(str)
    # Initialize TF-IDF Vectorizer  df[category_column].astype(str)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_description'])   
    # Combine the new description and category for the query
    new_combined_description = str(new_description) + " " + str(new_category)
    new_description_vector = tfidf.transform([new_combined_description])   
    # Calculate cosine similarities
    cosine_similarities = cosine_similarity(new_description_vector, tfidf_matrix)  
    # Find the most similar product index and details
    most_similar_index = cosine_similarities.argmax()
    most_similar_product = df.iloc[most_similar_index].copy()  
    # Select specific columns if needed
    most_similar_product = most_similar_product[['CATEGORY', 'PRODUCT', 'BRAND', 'CATEGORY TYPE']]  
    # Extract the highest similarity score
    highest_similarity_score = cosine_similarities.max()   
    return most_similar_product.to_frame().T

def send_interakt_message100(user_number, message_name, language_code):
    url = 'https://api.interakt.ai/v1/public/message/'
    headers = {
        'Authorization': 'cnpha0o4ZlRzUzBNeEwwUEd2NENpajRCeE05VC1MWDk0T3pkcVhrQmZzTTo=',  # Update this
        'Content-Type': 'application/json'
    }
    data = {
        "countryCode": "+91",
        "phoneNumber": user_number,
        "callbackData": "Processed your file",
        "type": "Template",
        "template": {
            "name": message_name,
            "languageCode": language_code
        }
    }

    response = requests.post(url, headers=headers, json=data)
    return response

def calculate_volume_growth(row):
    first = row['Quantity Sold First Excel']
    second = row['Quantity Sold Second Excel']
    if second == 0:
        if first == 0:
            return 0  # No growth if both are zero
        return 100  # 100% growth if the second value is zero
    if first == 0:
        return -100  # -100% growth if the first value is zero but the second is not
    return (first / second) * 100

def calculate_volume_growthNew(row):
    if row['Quantity Sold Second Excel'] > 0:
        return f"{int((row['Quantity Sold First Excel'] / row['Quantity Sold Second Excel'] - 1) * 100)}%"
    else:
        return "0%"


def send_interakt_message(user_number, pdflink, language_code):
    url = 'https://api.interakt.ai/v1/public/message/'
    headers = {
        'Authorization': 'Basic {{cnpha0o4ZlRzUzBNeEwwUEd2NENpajRCeE05VC1MWDk0T3pkcVhrQmZzTTo=}}',  # Update this
        'Content-Type': 'application/json'
    }
    data = {
        "countryCode": "+91",
        "phoneNumber": user_number,
        "callbackData": "some text here",
        "type": "Template",
        "template": {
            "name": "sales_analysis_second_message",
            "languageCode": "en",
            "bodyValues": [
                "Vinayak"
                ],
            "buttonValues": {
                "1": [
                    pdflink
                ]
        }
    }}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        print(f"Failed to send message: {response.text}")
    return response

def parse_percentage(item):
    if item:
        percent = re.search(r'\((\d+\.\d+)%\)', item)
        return float(percent.group(1)) if percent else 0
    return 0


# Function to extract the percentage value from the category string
def extract_percentage(category):
    result = re.search(r'\((\d+\.\d+)%\)', category)
    return float(result.group(1)) if result else 0

def df_to_table_data(df):
    # Convert DataFrame to a list of lists after checking if it's empty
    if not df.empty:
        return [df.columns.tolist()] + df.values.tolist()
    else:
        return []

def save_pie_chart(data, filename, title):
    labels = data.keys()
    sizes = data.values()
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def add_table_to_story(dataframe, title, story, styles):
    story.append(Paragraph(title, styles['Heading2']))
    data = [dataframe.columns.tolist()] + dataframe.values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

def dataframe_to_list(dataframe):
    # Include headers and data rows
    return [dataframe.columns.tolist()] + dataframe.values.tolist()

def df_to_table_data1(df, columns):
    """Converts DataFrame to list format including headers for report generation."""
    return [columns] + df[columns].values.tolist()

def safe_divide(numerator, denominator):
    """Safely divide two numbers, returning NaN if the denominator is zero."""
    if denominator == 0:
        return np.nan
    else:
        return (numerator - denominator) / denominator * 100

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    session.clear()
    if request.method == 'POST':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mandatory_file = request.files['mandatoryExcel']
        optional_file = request.files.get('optionalExcel')  # This is optional

        # Initialize filenames to None or empty string before the conditions
        mandatory_filename = None
        optional_filename = None
        mandatory_columns = []
        optional_columns = []
        
        if mandatory_file and mandatory_file.filename:
            mandatory_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{mandatory_file.filename}")
            # print('mandatory_filename: ',mandatory_filename)
            mandatory_file.save(mandatory_filename)
            session['mandatory_filename'] = mandatory_filename
            df_mandatory = pd.read_excel(mandatory_filename)
            mandatory_columns = df_mandatory.columns.tolist()
            
        if optional_file and optional_file.filename:
            optional_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{optional_file.filename}")
            optional_file.save(optional_filename)
            session['optional_filename'] = optional_filename
            df_optional = pd.read_excel(optional_filename)
            optional_columns = df_optional.columns.tolist()
        return render_template('upload_success.html', 
                               mandatory_filename=mandatory_file.filename if mandatory_file and mandatory_file.filename else None, 
                               optional_filename=optional_file.filename if optional_file and optional_file.filename else None,
                               mandatory_columns=mandatory_columns,
                               optional_columns=optional_columns)

    return render_template('index.html')


@app.route('/process-one', methods=['POST'])
def process_one():
    for key in session:
        print(f"{key}: {session[key]}")
    mandatory_filename = request.files['mandatory_filename']
    optional_filename = request.files.get('optional_filename',None)
    if mandatory_filename and not optional_filename:
        print('----------------------------------1 Month-----------------------------------')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        existing_catalog_path = os.path.join(app.config['UPLOAD_FOLDER'], "Merger (1).xlsx")
        existing_df = pd.read_excel(existing_catalog_path)

        # Retrieve file names and column names
        # mandatory_filename = request.form.get('mandatory_filename')
        # if 'mandatory_filename' not in request.files:
        #     return "No file part", 400
        
        #mandatory_filename = session.get('mandatory_filename')
        # if mandatory_filename.filename == '':
        #     return "No selected file", 400
        
        if mandatory_filename:
            # filename = secure_filename(mandatory_filename)
            # print("File Name: ", filename)
            # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # print("File Path: ", file_path)
            # mandatory_filename.save(file_path)
            print('-----------------------------------------1----------------------------------') 
            # Read the uploaded file
            mandatory_df = pd.read_excel(mandatory_filename)

            mandatory_product_description_column = request.form.get('mandatory_product_description')
            mandatory_quantity_sold_column = request.form.get('mandatory_quantity_sold')
            mandatory_price_column = request.form.get('mandatory_price')
            mandatory_category_column = request.form.get('mandatory_category', None)
            mobile_number = request.form.get('phone_number')

            # Retrieve number of bills
            num_of_bills = request.form.get('num_of_bills', default=1, type=int)
            store_szie = request.form.get('store_size', default=1, type=int)

            # Initialize DataFrame for combined data
            combined_df = pd.DataFrame(columns=[
                'Item Name', 'User Category', 'Category Predicted',  'Product Predicted',
                'Brand Predicted', 'Category Type', 'Quantity Sold First Excel', 
                'Price First Excel', 'Quantity Sold Second Excel', 
                'Price Second Excel', 'Quantity Growth', 'Value Growth'
            ])
            # After processing, send a message
            # user_number = "+919892485682"  # Replace with the actual user's number
            # message_name = "sales_analysis"  # Template name configured in Interakt
            # language_code = "hi"  # Language of the template

            # response = send_interakt_message(user_number, message_name, language_code)

            # Process mandatory file
            # if mandatory_filename:
            #mandatory_df = pd.read_excel(mandatory_filename)
            # Group by item description and sum the quantities and prices
            mandatory_agg = mandatory_df.groupby(mandatory_product_description_column).agg({
                mandatory_quantity_sold_column: 'sum',
                mandatory_price_column: 'sum'
            }).reset_index()

            mandatory_merged = mandatory_agg.merge(mandatory_df[[mandatory_product_description_column, mandatory_category_column]].drop_duplicates(),
                                                on=mandatory_product_description_column,
                                                how='left')
            new_rows = []
            for index, row in mandatory_merged.iterrows():
                new_description = row[mandatory_product_description_column]
                userCat = row[mandatory_category_column]
                most_similar_product = find_most_similar_product(new_description,userCat, existing_df, 'Product Description', 'CATEGORY')
                
                new_row = {
                    'Item Name': row[mandatory_product_description_column],
                    'User Category': row[mandatory_category_column],
                    'Category Predicted': most_similar_product['CATEGORY'].iloc[0],
                    'Category Type': most_similar_product['CATEGORY TYPE'].iloc[0],
                    'Product Predicted': most_similar_product['PRODUCT'].iloc[0],
                    'Brand Predicted': most_similar_product['BRAND'].iloc[0],
                    'Quantity Sold First Excel': row[mandatory_quantity_sold_column],
                    'Price First Excel': row[mandatory_price_column],
                    'Quantity Sold Second Excel': None,
                    'Price Second Excel': None
                }
                new_rows.append(new_row)

            new_rows_df = pd.DataFrame(new_rows)
            combined_df = pd.concat([combined_df, new_rows_df], ignore_index=True)

            total_price_first_excel = combined_df['Price First Excel'].sum()
            total_item_sold = combined_df['Quantity Sold First Excel'].sum()
            category_price_sums_first = combined_df.groupby('Category Type')['Price First Excel'].sum()
            category_percentage_first = {}

            # Calculate the percentage for each category in first and second excel
            for category in category_price_sums_first.index:
                category_percentage_first[category] = (category_price_sums_first[category] / total_price_first_excel) * 100
            print('Last Month Analysis')
            for category, percentage in category_percentage_first.items():
                print(f"Category mandatory'{category}' makes up {percentage:.2f}% of the Last Month total price.")
            print()
            combined_df['Quantity Sold First Excel'] = combined_df['Quantity Sold First Excel'].replace(0, np.nan)
            combined_df['Price ASP First'] = combined_df['Price First Excel'] / combined_df['Quantity Sold First Excel']
            combined_df['Price ASP First'] = combined_df['Price ASP First'].fillna(0)

            # to print the category of the customer
            category_price_sums_first1 = combined_df.groupby('User Category')['Price First Excel'].sum()
            category_percentage_first1 = {}
            for category in category_price_sums_first1.index:
                category_percentage_first1[category] = (category_price_sums_first1[category] / total_price_first_excel) * 100
            print('Last Month Analysis')
            for category, percentage in category_percentage_first1.items():
                print(f"User Category '{category}' makes up {percentage:.2f}% of the Last Month total price.")
            
            # Sort the dictionary by percentage in descending order to get the top 5
            sorted_categories_top5 = sorted(category_percentage_first1.items(), key=lambda x: x[1], reverse=True)[:5]

            # Sort the dictionary by percentage in ascending order to get the bottom 5
            sorted_categories_bottom5 = sorted(category_percentage_first1.items(), key=lambda x: x[1])[:5]

            ABV = total_price_first_excel/num_of_bills
            ABV = f"{ABV:.2f}"
            print('Average Basket Value: ',ABV)
            print()
            ABS = total_item_sold/num_of_bills
            ABS = f"{ABS:.2f}"
            print("Average Basket Sales: ",ABS)
            print()
            AIV = total_price_first_excel/total_item_sold
            AIV = f"{AIV:.2f}"
            print("Average Item Value: ",AIV)
            print()
            sqft = total_price_first_excel/store_szie
            sqft = f"{sqft:.2f}"
            print("Average Item Value: ",sqft)
            print()

            # Calculate Growth
            for index, row in combined_df.iterrows():
                if pd.notna(row['Quantity Sold First Excel']) and pd.notna(row['Quantity Sold Second Excel']):
                    combined_df.at[index, 'Quantity Growth'] = "{:.0%}".format((row['Quantity Sold First Excel'] - row['Quantity Sold Second Excel']) / row['Quantity Sold Second Excel'])
                if pd.notna(row['Price First Excel']) and pd.notna(row['Price Second Excel']):
                    combined_df.at[index, 'Value Growth'] = "{:.0%}".format((row['Price First Excel'] - row['Price Second Excel']) / row['Price Second Excel'])
                
            
            # Save combined data to a new Excel file
            # Generate a unique identifier for the filename
            unique_id = uuid4()
            output_filename = f"StructuredData_{unique_id}.xlsx"
            output_file_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)    
            combined_df.to_excel(output_file_path, index=False)


            # Destination Food % for One Month
            # print("Destination Food for Last Month")
            destination_food_percentage = category_percentage_first.get('Destination Food', 0)
            destination_food_message = ""
            destination_insights = ""
            if destination_food_percentage > 45:
                destination_food_message = f""
            elif destination_food_percentage <= 45:
                destination_insights = f"Destination categories which are core to your business have less than desired contribution at {destination_food_percentage:.2f}%. "
                destination_food_message = (
            f"Destination categories which are core to your business have less than desired contribution at {destination_food_percentage:.2f}%."
            " Consider strengthening destination categories to bring them to at least 45% contribution. It can even go up to 50% depending upon shopping habits of your core customers."
            " You can do it by:\n"
            "Enhancing your product range\n"
            "Ensuring good display\n"
            "Ensuring product availability\n"
            "Planning promotions"
            )
            print(destination_food_message)
            print()
            
            if mandatory_filename and destination_food_percentage <= 45:
                destination_food_df = combined_df[combined_df['Category Type'] == 'Destination Food']
                total_price_first_excel1 = combined_df['Price First Excel'].sum()
                total_item_sold = combined_df['Quantity Sold First Excel'].sum()
                category_price_sums_first10 = destination_food_df.groupby('Category Type')['Price First Excel'].sum()
                filename = f"Destination_Food_Items_{timestamp}.xlsx"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                destination_food_df.to_excel(filepath, index=False)
                
                destination_food_df = combined_df[combined_df['Category Type'] == 'Destination Food']
                total_price_destination = combined_df['Price First Excel'].sum()
                category_price_sums_destination = destination_food_df.groupby('User Category')['Price First Excel'].sum()
                category_percentage_destinationNew = {category: (price / total_price_destination) * 100 for category, price in category_price_sums_destination.items()}
                for category, percentage in category_percentage_destinationNew.items():
                    print(f"User Category '{category}' makes up {percentage:.2f}% of the Last Month total price within Destination Food.")
                data = {
                    "User Category": list(category_percentage_destinationNew.keys()),
                    "Percentage": [f"{percentage:.2f}%" for percentage in category_percentage_destinationNew.values()]
                }
                df = pd.DataFrame(data)
            else:
                category_percentage_destinationNew = {}
            
            session['destinationInformation'] = {"destination_food_percentage": destination_food_percentage, "destination_food_message": destination_food_message, "destination_insights": destination_insights, "category_percentage_destinationNew": category_percentage_destinationNew}

            
            # Calculate the combined percentage of Destination Food and Routine Non Core Food
            print("All Food Categories for Last Month")
            combined_percentage = category_percentage_first.get('Destination Food', 0) + category_percentage_first.get('Routine Non Core Food', 0)
            substantially_below = "substantially " if combined_percentage < 50 else ""
            food_categories_message = ""
            food_categories_insights = ""

            if combined_percentage > 65:
                food_categories_message = f""
            else:
                food_categories_insights = f"Shopping of food categories is the key motivation for monthly shopping of a family and they spend about 2/3rd on food categories. Your Food categories fall the desired level and is at {combined_percentage:.2f}%. "
                food_categories_message = f"Shopping of food categories is the key motivation for monthly shopping of a family and they spend about 2/3rd on food categories. Your Food categories fall {substantially_below} the desired level and is at {combined_percentage:.2f}%. Consider strengthening of your food categories to improve sales and customer loyalty by -\nEnhancing your product range\nEnsuring good display\nEnsuring product availability\nPlanning promotions\nPremiumisation"
            # print("food categories message message for Last month")
            print(food_categories_message)
            print()

            if mandatory_filename and combined_percentage < 65:
                food_categories_df = combined_df[combined_df['Category Type'].isin(['Destination Food', 'Routine Non Core Food'])]
                # Generate a unique timestamp to append to the filename
                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"food_categories_Items_{timestamp}.xlsx"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                food_categories_df.to_excel(filepath, index=False)

                food_categories_df100 = combined_df[combined_df['Category Type'].isin(['Destination Food', 'Routine Non Core Food'])]
                total_price_destination100 = combined_df['Price First Excel'].sum()
                category_price_sums_destination100 = food_categories_df100.groupby('User Category')['Price First Excel'].sum()
                category_percentage_destination100 = {category: (price / total_price_destination100) * 100 for category, price in category_price_sums_destination100.items()}
                for category, percentage in category_percentage_destination100.items():
                    print(f"User Category '{category}' makes up {percentage:.2f}% of the Last Month total price within Destination Food.")
                data = {
                    "User Category": list(category_percentage_destination100.keys()),
                    "Percentage": [f"{percentage:.2f}%" for percentage in category_percentage_destination100.values()]
                }
                df = pd.DataFrame(data)
            else:
                category_percentage_destination100 = {}

            session['AllFoodCategories'] = {"combined_percentage": combined_percentage, "food_categories_message": food_categories_message, "food_categories_insights": food_categories_insights, "category_percentage_destination100": category_percentage_destination100}


            print("Non Food Categories for Last Month")
            routine_non_food_percentage = category_percentage_first.get('Routine Non Food', 0)
            routine_non_food_message = ""
            routine_non_food_insights = ""

            if routine_non_food_percentage < 30:
                routine_non_food_insights = f"Your non-food categories contribution is at {routine_non_food_percentage:.2f}%. There is a possibility that you are losing opportunity of selling these categories."
                routine_non_food_message = f"Your non-food categories contribution is at {routine_non_food_percentage:.2f}%. There is a possibility that you are losing the opportunity of selling these categories. Consider strengthening these categories without harming your food sales by -\nEnhancing your product range\nEnsuring good display\nEnsuring product availability\nPlanning promotions\nPremiumisation"
            else:
                routine_non_food_message = f""
            # print("routine non food message message message for Last month")
            print(routine_non_food_message)
            print()
            if mandatory_filename and routine_non_food_percentage < 30:
                Routine_Non_Food_df = combined_df[combined_df['Category Type'] == 'Routine Non Food']
                # Generate a unique timestamp to append to the filename
                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"Routine_Non_Food_Items_{timestamp}.xlsx"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                Routine_Non_Food_df.to_excel(filepath, index=False)

                Routine_Non_Food_df101 = combined_df[combined_df['Category Type'] == 'Routine Non Food']
                total_price_destination101 = combined_df['Price First Excel'].sum()
                category_price_sums_destination101 = Routine_Non_Food_df101.groupby('User Category')['Price First Excel'].sum()
                category_percentage_destination101 = {category: (price / total_price_destination101) * 100 for category, price in category_price_sums_destination101.items()}
                for category, percentage in category_percentage_destination101.items():
                    print(f"User Category '{category}' makes up {percentage:.2f}% of the Last Month total price within Destination Food.")
                data = {
                    "User Category": list(category_percentage_destination101.keys()),
                    "Percentage": [f"{percentage:.2f}%" for percentage in category_percentage_destination101.values()]
                }
                df101 = pd.DataFrame(data)
            else:
                category_percentage_destination101 = {}

            session['NonFoodCategories'] = {"routine_non_food_percentage": routine_non_food_percentage, "routine_non_food_message": routine_non_food_message, "routine_non_food_insights": routine_non_food_insights, "category_percentage_destination101": category_percentage_destination101}

            # Salt Circle Analysis
            # Step 1: Filter combined_df for Sugar/Salt category and Salt product
            salt_circle_df = combined_df[(combined_df['Category Predicted'] == 'Sugar/Salt') & (combined_df['Product Predicted'] == 'Salt')]
            # Step 2: Calculate No_of_Salt_Packets
            No_of_Salt_Packets = salt_circle_df['Quantity Sold First Excel'].sum()
            # Check to avoid division by zero
            if No_of_Salt_Packets > 0:
                # Step 3: Calculate No of Families Shopping Estimate
                Total_Sales = combined_df['Price First Excel'].sum()
                No_of_Families_Shopping_Estimate = Total_Sales / No_of_Salt_Packets
                No_of_Families_Shopping_Estimate = int(round(No_of_Families_Shopping_Estimate))
            else:
                No_of_Families_Shopping_Estimate = 0
            # Fixed value for Average Monthly Basket Estimate
            Average_Monthly_Basket_Estimate = 4000  # Rs
            # Step 4: Compare and determine if it's greater or less
            if No_of_Families_Shopping_Estimate > Average_Monthly_Basket_Estimate:
                salt_msg = f"No. of family shopping is {No_of_Salt_Packets}. Average Monthly Basket Estimate is Rs{No_of_Families_Shopping_Estimate:.2f}. Total Sales is Rs{Total_Sales}."
                comparison_message = f"The No of Families Shopping Estimate is greater than the Average Monthly Basket Estimate which is Rs{No_of_Families_Shopping_Estimate:.2f}"
            elif No_of_Families_Shopping_Estimate < Average_Monthly_Basket_Estimate:
                salt_msg = f"No. of family shopping is {No_of_Salt_Packets}. Average Monthly Basket Estimate is Rs{No_of_Families_Shopping_Estimate:.2f}. Total Sales is Rs{Total_Sales}."
                comparison_message = f"The No of Families Shopping Estimate is less than the Average Monthly Basket Estimate which is Rs{No_of_Families_Shopping_Estimate}"
            else:
                salt_msg = f"No. of family shopping is {No_of_Salt_Packets}. Average Monthly Basket Estimate is Rs{No_of_Families_Shopping_Estimate:.2f}. Total Sales is Rs{Total_Sales}."
            print(comparison_message)

            # Top SKU Contributing to 80% of Sales
            if mandatory_filename:
                total_sku = len(combined_df)
                total_sales = combined_df['Price First Excel'].sum()
                df_sorted = combined_df.sort_values(by='Price First Excel', ascending=False)
                df_sorted['cumulative_sales'] = df_sorted['Price First Excel'].cumsum()
                df_sorted['cumulative_percentage'] = df_sorted['cumulative_sales'] / total_sales * 100
                cutoff_index = df_sorted[df_sorted['cumulative_percentage'] >= 80].index[0]
                top_80_percent_itemsSKU = df_sorted.loc[:cutoff_index]
                num_top_80_percent_items = len(top_80_percent_itemsSKU)
                percentageofsku = (num_top_80_percent_items / total_sku) * 100
                SKU_msg = f"Total SKU is {total_sku}. Total Sales is Rs{total_sales:.2f}. {num_top_80_percent_items} SKU's contribute to 80% of sales which is {percentageofsku:.2f}% of total SKU's."
                # output_path = 'Top_80_Percent_High_Selling_Sku.xlsx'
                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f'SKU_Contributing_80%_sales_{timestamp}.xlsx'
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_path)
                top_80_percent_itemsSKU.to_excel(output_path, index=False)
                top_20_itemsSKU = top_80_percent_itemsSKU.head(20)[['Item Name', 'Price First Excel']]
                top_20_itemsSKU = top_20_itemsSKU.rename(columns={
                    'Price First Excel': 'Price Last Month',
                })
                data103 = [top_20_itemsSKU.columns.tolist()] + top_20_itemsSKU.values.tolist()
                # Get the bottom 20 entries
                bottom_20 = df_sorted.tail(20)[['Item Name', 'Price First Excel']]
                bottom_20_itemsSKU = bottom_20.rename(columns={
                    'Price First Excel': 'Price Last Month',
                })
                data199 = [bottom_20_itemsSKU.columns.tolist()] + bottom_20_itemsSKU.values.tolist()




            # Top Product Contributing to 80% of Sales
            if mandatory_filename:
                aggregated_data = combined_df.groupby('Product Predicted').agg({
                    'Quantity Sold First Excel': 'sum',
                    'Price First Excel': 'sum',
                }).reset_index()
                total_sales = aggregated_data['Price First Excel'].sum()
                df_sorted = aggregated_data.sort_values(by='Price First Excel', ascending=False)
                df_sorted['cumulative_sales'] = df_sorted['Price First Excel'].cumsum()
                df_sorted['cumulative_percentage'] = df_sorted['cumulative_sales'] / total_sales * 100
                cutoff_index = df_sorted[df_sorted['cumulative_percentage'] >= 80].index[0]
                top_80_percent_items = df_sorted.loc[:cutoff_index]
                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f'Product_Contributing_80%_sales_{timestamp}.xlsx'
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_path)
                top_80_percent_items.to_excel(output_path, index=False)
                top_20_itemsproducts = top_80_percent_items.head(20)[['Product Predicted', 'Price First Excel']]
                top_20_itemsproducts = top_20_itemsproducts.rename(columns={
                    'Price First Excel': 'Price Last Month',
                })
                data104 = [top_20_itemsproducts.columns.tolist()] + top_20_itemsproducts.values.tolist()


            # Pie Chart Generation
            save_pie_chart(category_percentage_first, 'last_month_pie_chart.png', 'Last Month Category Percentages')


            # Pdf Generation
            pdf_filename = f'Sales_Analysis_Report_{timestamp}.pdf'
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
            pdf = SimpleDocTemplate(pdf_path, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()

            # Section : Purpose
            styles.add(ParagraphStyle(name='CenteredBoldHeading2', parent=styles['Heading2'], alignment=TA_CENTER, fontName='Helvetica-Bold'))
            styles.add(ParagraphStyle(name='CenteredHeading1', parent=styles['Heading1'], alignment=TA_CENTER))

            # Header Section
            story.append(Spacer(1, 12))
            story.append(Paragraph("Sales Analysis Insight & Action Report", styles['CenteredBoldHeading2']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("A sales analysis tool is a software application that helps businesses track, analyze, and visualize their sales data to gain insights into sales performance. It typically offers features like sales forecasting, trend analysis, and performance metrics. These tools enable companies to make data-driven decisions, identify sales opportunities, optimize sales strategies, and enhance customer satisfaction. They are commonly used by sales teams, managers, and executives to monitor sales activities and achieve business goals.", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(PageBreak())
            # Section 1: Key Performance Indicators
            story.append(Paragraph("KEY Performance Indicators", styles['Heading2']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("Average Basket Value (ABV): Average Value of each bill", styles['BodyText']))
            story.append(Paragraph(f"(a) Average Basket Value (ABV) ={ABV}", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("Average Basket Sales (ABS): Average number of items in each bill", styles['BodyText']))
            story.append(Paragraph(f"(b) Average Basket Sales (ABS) = {ABS}", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("Average Item Value (ASP): Average price of each item sold", styles['BodyText']))
            story.append(Paragraph(f"(c) Average Item Value (ASP) = {AIV}", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"(d) Per Sq. ft.sales = {sqft}", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"(e) Number of bills: {num_of_bills}", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"(f) Current Month sale Value: {total_price_first_excel:.2f}", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"(g) Store Size: {store_szie}", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(Spacer(1, 12))

            story.append(Paragraph(" Family Basket", styles['Heading1']))
            story.append(Paragraph("Salt Circle", styles['BodyText']))
            story.append(Paragraph("This part of the report shows you the estimates of numbers of families who have shopped with you in the given month and their shopping basket assessment.", styles['BodyText']))
            story.append(Paragraph("Insight: ", styles['BodyText']))
            story.append(Paragraph(f"{salt_msg}", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("Recommendation: ", styles['BodyText']))
            story.append(Paragraph("1. Keep track of trends", styles['BodyText']))
            story.append(Paragraph("2. Target more families", styles['BodyText']))
            story.append(Paragraph("3. Target improving ABS and ASP", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(PageBreak())

            # Section 2: Business Group Performance
            story.append(Paragraph("Business Group Performance", styles['Heading1']))
            story.append(Paragraph("This part of the analysis shows you the performance of your various business groups that consist of similar categories in the way customers see them. It considers the benchmark participation of these business groups and provides you insights and actions required,", styles['BodyText']))
            last_month_image = Image('last_month_pie_chart.png')
            last_month_image._restrictSize(400, 400)
            story.append(last_month_image)
            if destination_food_message.strip():
                story.append(Paragraph("Destination Category %", styles['Heading2']))
                story.append(Paragraph("Destination Category Insight:", styles['BodyText']))
                destination_insights1 = Paragraph(destination_insights, styles['BodyText'])
                story.append(destination_insights1)
                story.append(Spacer(1, 12))
                story.append(Paragraph("Destination Category Recommendation:", styles['BodyText']))
                destination_food_paragraph = Paragraph(destination_food_message, styles['BodyText'])
                story.append(destination_food_paragraph)
                columns_to_display = [
                    "Item Name",
                    "Quantity Sold First Excel",
                    "Price First Excel",
                ]
                table_data = [['User Category', 'Percentage']] + [(cat, f"{perc:.2f}%") for cat, perc in category_percentage_destinationNew.items()]
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ]))
                story.append(table)
            if food_categories_message.strip():
                story.append(Paragraph("All Food Category %", styles['Heading2']))
                story.append(Paragraph("All Food Category Insight:", styles['BodyText']))
                line12 = Paragraph(food_categories_insights, styles['BodyText'])
                story.append(line12)
                story.append(Spacer(1, 12))
                story.append(Paragraph("All Food Category Recommendation:", styles['BodyText']))
                line = Paragraph(food_categories_message, styles['BodyText'])
                story.append(line)
                table_data = [['User Category', 'Percentage']] + [(cat, f"{perc:.2f}%") for cat, perc in category_percentage_destination100.items()]
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ]))
                story.append(table)
            if routine_non_food_message.strip():
                story.append(Paragraph("Non Food Category %", styles['Heading2']))
                story.append(Paragraph("Non Food Category Insight:", styles['BodyText']))
                line13 = Paragraph(routine_non_food_insights, styles['BodyText'])
                story.append(line13)
                story.append(Spacer(1, 12))
                story.append(Paragraph("Non Food Category Recommendation:", styles['BodyText']))
                line1 = Paragraph(routine_non_food_message, styles['BodyText'])
                story.append(line1)
                table_data = [['User Category', 'Percentage']] + [(cat, f"{perc:.2f}%") for cat, perc in category_percentage_destination101.items()]
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ]))
                story.append(table)
            story.append(Spacer(1, 12))
            story.append(PageBreak())

            story.append(Paragraph("Top 5 Contributing Category %", styles['Heading1']))
            story.append(Paragraph("Insight: Shows you categories growth over the previous month by Percentage", styles['BodyText']))
            table_data = [['User Category', 'Percentage']] + [(cat, f"{perc:.2f}%") for cat, perc in sorted_categories_top5]
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ]))
            story.append(table)
            story.append(Spacer(1, 12))
            story.append(Spacer(1, 12))
            story.append(Paragraph("Bottom 5 Contributing Category %", styles['Heading1']))
            story.append(Paragraph("Insight: Shows you categories de-growth over the previous month by Percentage", styles['BodyText']))
            table_data = [['User Category', 'Percentage']] + [(cat, f"{perc:.2f}%") for cat, perc in sorted_categories_bottom5]
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ]))
            story.append(table)
            story.append(PageBreak())
            # Section Top SKU
            story.append(Paragraph("Top 20 SKU", styles['Heading1']))
            story.append(Paragraph("Insights: This part of the report shows you top 20 on SKU level", styles['BodyText']))
            table = Table(data103)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
                ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ]))
            story.append(table)
            story.append(Spacer(1, 12))
            story.append(PageBreak())

            # Section Top Product
            story.append(Paragraph("Top 20 Products", styles['Heading1']))
            story.append(Paragraph("Insight: This part of the report shows you top 20 on Product level", styles['BodyText']))
            story.append(Spacer(1, 12))
            table = Table(data104)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
                ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ]))
            story.append(table)
            story.append(Spacer(1, 12))
            story.append(Paragraph("Bottom 20 Products", styles['Heading1']))
            story.append(Paragraph("Insight: This part of the report shows you bottom 20 on Product level", styles['BodyText']))
            table = Table(data199)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
                ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ]))
            story.append(table)
            story.append(Spacer(1, 12))
            story.append(PageBreak())

            pdf.build(story)
            print(f"PDF generated: {pdf_filename}")

            # Upload the PDF to S3
            try:
                s3.upload_file(pdf_path, S3_BUCKET, pdf_filename)
                s3_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{pdf_filename}"
                print(f"PDF uploaded to S3: {s3_url}")
            except Exception as e:
                print(f"Failed to upload PDF to S3: {e}")
                return jsonify({"error": "Failed to upload PDF to S3"}), 500

            # Save filename in session
            session['pdf_filename'] = pdf_filename

            presigned_url = s3.generate_presigned_url('get_object',
                                                    Params={'Bucket': S3_BUCKET, 'Key': pdf_filename},
                                                    ExpiresIn=3600)  # URL expires in 1 hour
            language_code = 'en'
            print()
            print()
            print("Mobile Number: ",mobile_number)
            print("presigned_url: ",presigned_url)
            print("language_code: ",language_code)
            print()
            print()
            response = send_interakt_message(mobile_number, presigned_url, language_code)
            print()
            print('Response: ',response)
            print()
            print('Processing complete. Structured data saved and uploaded to S3.')
            print(f"PDF generated: {pdf_filename}")
            return jsonify({"success": True,"data": {"presigned_url": presigned_url}})
        return jsonify({"success": True,"data": {"presigned_url": presigned_url}})
    else:
        print('-------------------------------------2 Month-----------------------------------')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        existing_catalog_path = os.path.join(app.config['UPLOAD_FOLDER'], "Merger (1).xlsx")
        existing_df = pd.read_excel(existing_catalog_path)

        # Retrieve file names and column names
        mandatory_filename = request.files['mandatory_filename']
        optional_filename = request.files.get('optional_filename', None)
        

        # After processing, send a message
        #user_number = "+919892485682"  # Replace with the actual user's number
        #message_name = "sales_analysis"  # Template name configured in Interakt
        #language_code = "hi"  # Language of the template

        #response = send_interakt_message(user_number, message_name, language_code)

        # mandatory_filename = request.files['mandatory_filename']
        # optional_filename = request.files['optional_filename']
        print("Mandatory file:", mandatory_filename)
        print("Optional file:", optional_filename)

        mandatory_product_description_column = request.form.get('mandatory_product_description')
        mandatory_quantity_sold_column = request.form.get('mandatory_quantity_sold')
        mandatory_price_column = request.form.get('mandatory_price')
        mandatory_category_column = request.form.get('mandatory_category', None)  # New
        optional_product_description_column = request.form.get('mandatory_product_description')
        optional_quantity_sold_column = request.form.get('mandatory_quantity_sold')
        optional_price_column = request.form.get('mandatory_price')
        optional_category_column = request.form.get('mandatory_category', None)  # New
        num_of_bills = request.form.get('may_bills', default=1, type=int)
        no_of_bills  = request.form.get('april_bills', default=1, type=int)
        store_szie = request.form.get('store_size', default=1, type=int)
        mobile_number = request.form.get('phone_number')

        # Initialize DataFrame for combined data
        combined_df = pd.DataFrame(columns=[
            'Item Name', 'User Category', 'Category Predicted',  'Product Predicted',
            'Brand Predicted', 'Category Type', 'Quantity Sold First Excel', 
            'Price First Excel', 'Quantity Sold Second Excel', 
            'Price Second Excel', 'Quantity Growth', 'Value Growth'
        ])

        # Process mandatory file
        if mandatory_filename:
            mandatory_df = pd.read_excel(mandatory_filename)
            # Group by item description and sum the quantities and prices
            mandatory_agg = mandatory_df.groupby(mandatory_product_description_column).agg({
                mandatory_quantity_sold_column: 'sum',
                mandatory_price_column: 'sum'
            }).reset_index()

            mandatory_merged = mandatory_agg.merge(mandatory_df[[mandatory_product_description_column, mandatory_category_column]].drop_duplicates(),
                                            on=mandatory_product_description_column,
                                            how='left')
            new_rows = []
            for index, row in mandatory_merged.iterrows():
                new_description = row[mandatory_product_description_column]
                userCat = row[mandatory_category_column]
                most_similar_product = find_most_similar_product(new_description,userCat, existing_df, 'Product Description', 'CATEGORY')
                
                new_row = {
                    'Item Name': row[mandatory_product_description_column],
                    'User Category': row[mandatory_category_column],
                    'Category Predicted': most_similar_product['CATEGORY'].iloc[0],
                    'Category Type': most_similar_product['CATEGORY TYPE'].iloc[0],
                    'Product Predicted': most_similar_product['PRODUCT'].iloc[0],
                    'Brand Predicted': most_similar_product['BRAND'].iloc[0],
                    'Quantity Sold First Excel': row[mandatory_quantity_sold_column],
                    'Price First Excel': row[mandatory_price_column],
                    'Quantity Sold Second Excel': None,
                    'Price Second Excel': None
                }
                new_rows.append(new_row)

            new_rows_df = pd.DataFrame(new_rows)
            combined_df = pd.concat([combined_df, new_rows_df], ignore_index=True)

            total_price_first_excel = combined_df['Price First Excel'].sum()
            total_item_sold = combined_df['Quantity Sold First Excel'].sum()
            category_price_sums_first = combined_df.groupby('Category Type')['Price First Excel'].sum()
            category_percentage_first = {}

            # # Create a pivot table or grouped table
            # unique_user_categories_per_category_type = combined_df.groupby('Category Type')['User Category'].unique().reset_index()

            # # Rename columns for clarity
            # unique_user_categories_per_category_type.columns = ['Category Type', 'Unique User Categories']

            # # Convert the unique categories list to a string for better readability in the table
            # unique_user_categories_per_category_type['Unique User Categories'] = unique_user_categories_per_category_type['Unique User Categories'].apply(lambda x: ', '.join(x))

            # Calculate the percentage for each category in first and second excel
            for category in category_price_sums_first.index:
                category_percentage_first[category] = (category_price_sums_first[category] / total_price_first_excel) * 100

            print('Last Month Analysis')
            for category, percentage in category_percentage_first.items():
                print(f"Category mandatory'{category}' makes up {percentage:.2f}% of the Last Month total price.")
            print()
            # Replace 0 with NaN to avoid division by zero error
            combined_df['Quantity Sold First Excel'] = combined_df['Quantity Sold First Excel'].replace(0, np.nan)

            # Now perform the division; any division by NaN will result in NaN instead of an error
            combined_df['Price ASP First'] = combined_df['Price First Excel'] / combined_df['Quantity Sold First Excel']

            # If you want to handle NaN values in 'Price ASP Second', you can fill them with a default value
            # For example, filling NaN with 0 or any other placeholder value
            combined_df['Price ASP First'] = combined_df['Price ASP First'].fillna(0)

            # to print the category of the customer
            category_price_sums_first1 = combined_df.groupby('User Category')['Price First Excel'].sum()
            category_percentage_first1 = {}

            category_types = ['Routine Non Core Food', 'Destination Food', 'Routine Non Food']
            unique_categories = {}
            total_sales_by_type = combined_df.groupby('Category Type')['Price First Excel'].sum()

            # Calculate the percentage for each category in first and second excel
            for category in category_price_sums_first1.index:
                category_percentage_first1[category] = (category_price_sums_first1[category] / total_price_first_excel) * 100
            
            for category in category_types:
                filtered_df = combined_df[combined_df['Category Type'] == category]
                unique_user_categories = filtered_df['User Category'].unique().tolist()
                category_sales = []

                for user_category in unique_user_categories:
                    # Use the precomputed percentage of sales for each User Category
                    percentage_of_sales = category_percentage_first1.get(user_category, 0)
                    
                    # Only include categories where the percentage is greater than 2%
                    if percentage_of_sales > 2:
                        category_sales.append(f"{user_category} ({percentage_of_sales:.2f}%)")

                unique_categories[category] = category_sales
            
            # Prepare the transposed table data
            max_length = max(len(unique_categories[cat]) for cat in unique_categories)
            transposed_data = [category_types]

            for i in range(max_length):
                row = []
                for category in category_types:
                    if i < len(unique_categories[category]):
                        row.append(unique_categories[category][i])
                    else:
                        row.append('')
                transposed_data.append(row)

            # Dictionary to store the highest percentage category across columns
            category_max = {}

            # Traverse through each category and its respective column
            for index, cat_list in enumerate(transposed_data[1:]):  # Skip the first row which is headers
                for col_index, category in enumerate(cat_list):
                    if category:  # Check if category string is not empty
                        name = re.sub(r'\(\d+\.\d+%\)', '', category).strip()
                        if name:
                            percent = float(re.search(r'\((\d+\.\d+)%\)', category).group(1))
                            if name not in category_max or category_max[name][1] < percent:
                                category_max[name] = (col_index, percent)  # Store column index and highest percentage
            
            # Create the final structure with unique highest percentages
            final_categories = [[] for _ in range(len(transposed_data[0]))]

            for name, details in category_max.items():
                col_index, percent = details
                final_categories[col_index].append(f'{name} ({percent:.2f}%)')

            # Print the result
            result_categories1 = [transposed_data[0]]  # Include headers
            max_length = max(len(col) for col in final_categories)
            for i in range(max_length):
                row = [col[i] if i < len(col) else '' for col in final_categories]
                result_categories1.append(row)
            
            # Sort each column based on the percentage values
            sorted_categories = []
            for col in result_categories1:
                header = col[0]
                items = sorted(col[1:], key=parse_percentage, reverse=True)
                sorted_categories.append([header] + items)

            # Determine the max length of actual data entries in the columns
            max_length = max(len([item for item in col if item]) for col in sorted_categories)

            # Adjust columns to be uniform by padding with empty strings only up to max_length
            adjusted_sorted_categories = []
            for col in sorted_categories:
                adjusted_sorted_categories.append(col[:max_length])

            # Prepare data for table, ensuring equal row count across all columns
            sorted_result_categories = [list(column) for column in zip(*adjusted_sorted_categories)]  # Transpose to convert to row-wise for the table




            print('Last Month Analysis')
            for category, percentage in category_percentage_first1.items():
                print(f"User Category '{category}' makes up {percentage:.2f}% of the Last Month total price.")
            print()

            ABV = total_price_first_excel/no_of_bills
            print('Average Basket Value: ',ABV)
            print()

            ABS = total_item_sold/no_of_bills
            print("Average Basket Sales: ",ABS)
            print()

            AIV = total_price_first_excel/total_item_sold
            print("Average Item Value: ",AIV)
            print()

            sqft = total_price_first_excel/store_szie
            print("Average Item Value: ",sqft)
            print()

            GwothAPICuurent = (ABS*AIV*no_of_bills)

            category_price_sums_first12 = combined_df.groupby('User Category')['Quantity Sold First Excel'].sum()
            category_percentage_first12 = {}

            # Calculate the percentage for each category in first and second excel
            for category in category_price_sums_first12.index:
                category_percentage_first12[category] = (category_price_sums_first12[category] / total_item_sold) * 100

            
            table_data200 = [['User Category', 'Value Growth %', 'Volume Growth %']]
            for category in category_percentage_first1:
                value_percent = category_percentage_first1.get(category, 0)  # Get value percent, default to 0 if not found
                volume_percent = category_percentage_first12.get(category, 0)  # Get volume percent, default to 0 if not found
                table_data200.append([category, f"{value_percent:.2f}%", f"{volume_percent:.2f}%"])





            # Process optional file
        if optional_filename:
            optional_df = pd.read_excel(optional_filename)
            # Group by item description and sum the quantities and prices
            optional_agg = optional_df.groupby(optional_product_description_column).agg({
                optional_quantity_sold_column: 'sum',
                optional_price_column: 'sum'
            }).reset_index()

            mandatory_merged1 = optional_agg.merge(optional_df[[optional_product_description_column, optional_category_column]].drop_duplicates(),
                                            on=optional_product_description_column,
                                            how='left')
            new_rows = []
            for index, row in mandatory_merged1.iterrows():
                new_description = row[optional_product_description_column]
                userCat = row[optional_category_column]
                # Check if this item is already in the combined_df
                if new_description in combined_df['Item Name'].values:
                    combined_df.loc[combined_df['Item Name'] == new_description, 'Quantity Sold Second Excel'] = row[optional_quantity_sold_column]
                    combined_df.loc[combined_df['Item Name'] == new_description, 'Price Second Excel'] = row[optional_price_column]
                else:
                    most_similar_product = find_most_similar_product(new_description,userCat, existing_df, 'Product Description', 'CATEGORY')
                    new_row = {
                        'Item Name': new_description,
                        'User Category': userCat,
                        'Category Predicted': most_similar_product['CATEGORY'].iloc[0] if not most_similar_product.empty else 'Unknown',
                        'Product Predicted': most_similar_product['PRODUCT'].iloc[0] if not most_similar_product.empty else 'Unknown',
                        'Brand Predicted': most_similar_product['BRAND'].iloc[0] if not most_similar_product.empty else 'Unknown',
                        'Category Type': most_similar_product['CATEGORY TYPE'].iloc[0] if not most_similar_product.empty else 'Unknown',
                        'Quantity Sold First Excel': None,
                        'Price First Excel': None,
                        'Quantity Sold Second Excel': row[optional_quantity_sold_column],
                        'Price Second Excel': row[optional_price_column]
                    }
                    new_rows.append(new_row)
            # Only update combined_df if there are new rows to add
            if new_rows:
                new_rows_df = pd.DataFrame(new_rows)  # Convert list of dicts to DataFrame
                combined_df = pd.concat([combined_df, new_rows_df], ignore_index=True)
    
            # Sum price and quantity sold in the second excel
            combined_df['Price Second Excel'] = pd.to_numeric(combined_df['Price Second Excel'], errors='coerce')
            combined_df['Quantity Sold Second Excel'] = pd.to_numeric(combined_df['Quantity Sold Second Excel'], errors='coerce')
    

            total_price_second_excel = combined_df['Price Second Excel'].sum()
            total_item_sold_second_excel = combined_df['Quantity Sold Second Excel'].sum()
            # category_price_sums_second = combined_df.groupby('User Category')['Price Second Excel'].sum()
            category_price_sums_second = combined_df.groupby('Category Type')['Price Second Excel'].sum()

            category_percentage_second = {}

            for category in category_price_sums_second.index:
                if total_price_second_excel > 0:  # Ensure there is no division by zero
                    category_percentage_second[category] = (category_price_sums_second[category] / total_price_second_excel) * 100

            print('Last to Last Month Analysis')
            for category, percentage in category_percentage_second.items():
                print(f"Category optional'{category}' makes up {percentage:.2f}% of the Last to Last Month total price.")
            print()
            # Replace 0 with NaN to avoid division by zero error
            combined_df['Quantity Sold Second Excel'] = combined_df['Quantity Sold Second Excel'].replace(0, np.nan)

            # Now perform the division; any division by NaN will result in NaN instead of an error
            combined_df['Price ASP Second'] = combined_df['Price Second Excel'] / combined_df['Quantity Sold Second Excel']

            # If you want to handle NaN values in 'Price ASP Second', you can fill them with a default value
            # For example, filling NaN with 0 or any other placeholder value
            combined_df['Price ASP Second'] = combined_df['Price ASP Second'].fillna(0)

            category_price_sums_second1 = combined_df.groupby('User Category')['Price Second Excel'].sum()

            category_percentage_second1 = {}

            for category in category_price_sums_second1.index:
                if total_price_second_excel > 0:  # Ensure there is no division by zero
                    category_percentage_second1[category] = (category_price_sums_second1[category] / total_price_second_excel) * 100

            print('Last to Last Month Analysis')
            for category, percentage in category_percentage_second1.items():
                print(f"User Category '{category}' makes up {percentage:.2f}% of the Last to Last Month total price.")
            print()

            ABV1 = total_price_second_excel/num_of_bills
            print('Average Basket Value: ',ABV1)
            print()

            ABS1 = total_item_sold_second_excel/num_of_bills
            print("Average Basket Sales: ",ABS1)
            print()

            AIV1 = total_price_second_excel/total_item_sold_second_excel
            print("Average Item Value: ",AIV1)
            print()

            sqft1 = total_price_second_excel/store_szie
            print("Average Item Value: ",sqft1)
            print()

            GwothAPIPrevious = (ABS1*AIV1*num_of_bills)
            
            category_price_sums_first121 = combined_df.groupby('User Category')['Quantity Sold Second Excel'].sum()
            category_percentage_first121 = {}

            # Calculate the percentage for each category in first and second excel
            for category in category_price_sums_first121.index:
                category_percentage_first121[category] = (category_price_sums_first121[category] / total_item_sold) * 100

            
            table_data201 = [['User Category', 'Value Growth %', 'Volume Growth %']]
            for category in category_percentage_second1:
                value_percent = category_percentage_second1.get(category, 0)  # Get value percent, default to 0 if not found
                volume_percent = category_percentage_first121.get(category, 0)  # Get volume percent, default to 0 if not found
                table_data201.append([category, f"{value_percent:.2f}%", f"{volume_percent:.2f}%"])


            GrowthAPI = ((GwothAPICuurent - GwothAPIPrevious)/GwothAPIPrevious)*100

            #@@@@@@@@@@@@@@
            # Destination Food % for One Month
            # print("Destination Food for previous Month")
            destination_food_percentage = category_percentage_second.get('Destination Food', 0)
            destination_food_message = ""
            if destination_food_percentage > 45:
                destination_food_message = f"Destination categories which are core to your business have a strong contribution at {destination_food_percentage:.2f}%. "
            elif destination_food_percentage <= 45:
                destination_food_message = f"Destination categories which are core to your business have less than desired contribution at {destination_food_percentage:.2f}%. Consider strengthening destination categories to bring them to at least 45% contribution. It can even go up to 50% depending upon shopping habits of your core customers. You can do it by -\nEnhancing your product range\nEnsuring good display\nEnsuring product availability\nPlanning promotions"
            # print("Destination food message for Previous month")
            # print(destination_food_message)
            # print()

            # if optional_filename:
            destination_food_items_previous_Month = combined_df[(combined_df['Category Type'] == 'Destination Food') & 
                                            (combined_df['Quantity Sold Second Excel'].notna()) &
                                            (combined_df['Quantity Sold Second Excel'] > 0) &
                                            (combined_df['Price Second Excel'].notna()) &
                                            (combined_df['Price Second Excel'] > 0)]

            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"destination_food_items_previous_Month_{timestamp}.xlsx"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            destination_food_items_previous_Month.to_excel(filepath, index=False)


            # Calculate the combined percentage of Destination Food and Routine Non Core Food
            # print("All Food Categories for previous Month")
            combined_percentage = category_percentage_second.get('Destination Food', 0) + category_percentage_second.get('Routine Non Core Food', 0)
            substantially_below = "substantially " if combined_percentage < 50 else ""
            food_categories_message = ""

            if combined_percentage > 65:
                food_categories_message = f"Shopping of food categories is the key motivation for monthly shopping of a family and they spend about 2/3rd on food categories. Your Food categories fall the desired level and is at {combined_percentage:.2f}%. "
            else:
                food_categories_message = f"Shopping of food categories is the key motivation for monthly shopping of a family and they spend about 2/3rd on food categories. Your Food categories fall {substantially_below}the desired level and is at {combined_percentage:.2f}%. Consider strengthening of your food categories to improve sales and customer loyalty by -\nEnhancing your product range\nEnsuring good display\nEnsuring product availability\nPlanning promotions\nPremiumisation"
            # print("food categories message message for Previous month")
            # print(food_categories_message)
            # print()
            # if optional_filename:
            food_categories_items_previous_Month = combined_df[
                    (combined_df['Category Type'].isin(['Destination Food', 'Routine Non Core Food'])) &
                    (combined_df['Quantity Sold Second Excel'].notna()) &
                    (combined_df['Quantity Sold Second Excel'] > 0) &
                    (combined_df['Price Second Excel'].notna()) &
                    (combined_df['Price Second Excel'] > 0)
                ]
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"food_categories_items_previous_Month_{timestamp}.xlsx"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            food_categories_items_previous_Month.to_excel(filepath, index=False)
            
            routine_non_food_percentage = category_percentage_second.get('Routine Non Food', 0)
            routine_non_food_message = ""

            if routine_non_food_percentage < 30:
                routine_non_food_message = f"Your non-food categories contribution is at {routine_non_food_percentage:.2f}%. There is a possibility that you are losing the opportunity of selling these categories. Consider strengthening these categories without harming your food sales by -\nEnhancing your product range\nEnsuring good display\nEnsuring product availability\nPlanning promotions\nPremiumisation"
            else:
                routine_non_food_message = f"Your non-food categories contribution is at {routine_non_food_percentage:.2f}%."
            # print("Non Food Categories for Previous month")
            # print(routine_non_food_message)
            # print()
            # if optional_filename:
            routine_non_food_items_previous_Month = combined_df[(combined_df['Category Type'] == 'Routine Non Food') & 
                                            (combined_df['Quantity Sold Second Excel'].notna()) &
                                            (combined_df['Quantity Sold Second Excel'] > 0) &
                                            (combined_df['Price Second Excel'].notna()) &
                                            (combined_df['Price Second Excel'] > 0)]

            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"routine_non_food_items_previous_Month{timestamp}.xlsx"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            routine_non_food_items_previous_Month.to_excel(filepath, index=False)

        for index, row in combined_df.iterrows():
            # Calculate Quantity Growth
            if pd.notna(row['Quantity Sold First Excel']) and pd.notna(row['Quantity Sold Second Excel']):
                if row['Quantity Sold Second Excel'] != 0:
                    combined_df.at[index, 'Quantity Growth'] = "{:.0%}".format((row['Quantity Sold First Excel'] - row['Quantity Sold Second Excel']) / row['Quantity Sold Second Excel'])
                else:
                    combined_df.at[index, 'Quantity Growth'] = "1"  # or some other placeholder value
            else:
                combined_df.at[index, 'Quantity Growth'] = "1"

            # Calculate Value Growth
            if pd.notna(row['Price First Excel']) and pd.notna(row['Price Second Excel']):
                if row['Price Second Excel'] != 0:
                    combined_df.at[index, 'Value Growth'] = "{:.0%}".format((row['Price First Excel'] - row['Price Second Excel']) / row['Price Second Excel'])
                else:
                    combined_df.at[index, 'Value Growth'] = "1"  # or some other placeholder value
            else:
                combined_df.at[index, 'Value Growth'] = "1"

        # Filter and create dict for unique User Categories by Category Type
        category_types = ['Routine Non Core Food', 'Destination Food', 'Routine Non Food']
        unique_categories = {}

        total_sales_by_type = combined_df.groupby('Category Type')['Price First Excel'].sum()

        for category in category_types:
            filtered_df = combined_df[combined_df['Category Type'] == category]
            unique_user_categories = filtered_df['User Category'].unique().tolist()
            category_sales = []

            for user_category in unique_user_categories:
                # Calculate the total sales for each User Category
                user_category_sales = filtered_df[filtered_df['User Category'] == user_category]['Price First Excel'].sum()
                # Calculate the percentage of total sales for the category type
                percentage_of_sales = (user_category_sales / total_sales_by_type[category]) * 100 if total_sales_by_type[category] > 0 else 0
                # Only include categories where the percentage is greater than 2%
                if percentage_of_sales > 2:
                    category_sales.append(f"{user_category} ({percentage_of_sales:.2f}%)")
                    # category_sales.append(f"{user_category}")# ({percentage_of_sales:.2f}%)")

            unique_categories[category] = category_sales

        # Prepare the transposed table data
        max_length = max(len(unique_categories[cat]) for cat in unique_categories)
        # Include only the category types as headers in the first row
        transposed_data = [category_types]  

        for i in range(max_length):
            row = []
            for category in category_types:
                if i < len(unique_categories[category]):
                    row.append(unique_categories[category][i])
                else:
                    row.append('')
            transposed_data.append(row)

        # Dictionary to store the highest percentage category across columns
        category_max = {}

        # Traverse through each category and its respective column
        for index, cat_list in enumerate(transposed_data[1:]):  # Skip the first row which is headers
            for col_index, category in enumerate(cat_list):
                if category:  # Check if category string is not empty
                    # Remove the percentage and store the name
                    name = re.sub(r'\(\d+\.\d+%\)', '', category).strip()
                    if name:
                        percent = float(re.search(r'\((\d+\.\d+)%\)', category).group(1))
                        if name not in category_max or category_max[name][1] < percent:
                            category_max[name] = (col_index, percent)  # Store column index and highest percentage

        # Create the final structure with unique highest percentages
        final_categories = [[] for _ in range(len(transposed_data[0]))]  # Initialize with empty lists for columns

        for name, details in category_max.items():
            col_index, percent = details
            # col_index = details
            final_categories[col_index].append(f'{name} ({percent:.2f}%)')

        # Print the result
        result_categories = [transposed_data[0]]  # Include headers
        max_length = max(len(col) for col in final_categories)  # Find the maximum column length
        for i in range(max_length):
            row = [col[i] if i < len(col) else '' for col in final_categories]  # Construct row ensuring alignment
            result_categories.append(row)

        # Sort by Value Growth and filter top 20 and bottom 20
        combined_df['Value Growth'] = combined_df['Value Growth'].str.rstrip('%').astype('float') / 100.0  # Convert to float
        top_20 = combined_df.nlargest(20, 'Value Growth')
        bottom_20 = combined_df.nsmallest(20, 'Value Growth')

        # Ensure 'Value Growth' is formatted as string with '%' for display in tables
        top_20['Value Growth'] = top_20['Value Growth'].apply(lambda x: f"{x:.2f}%")
        bottom_20['Value Growth'] = bottom_20['Value Growth'].apply(lambda x: f"{x:.2f}%")

        # Prepare data for the table
        top_20_data = [['User Category', 'Value Growth', 'Quantity Growth']] + top_20[['User Category', 'Value Growth', 'Quantity Growth']].values.tolist()
        bottom_20_data = [['User Category', 'Value Growth', 'Quantity Growth']] + bottom_20[['User Category', 'Value Growth', 'Quantity Growth']].values.tolist()



        # Save combined data to a new Excel file
        # Generate a unique identifier for the filename
        unique_id = uuid4()
        output_filename = f"StructuredData_{unique_id}.xlsx"
        output_file_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)    
        combined_df.to_excel(output_file_path, index=False)


        # Destination Food % for One Month
        # print("Destination Food for Last Month")
        destination_food_percentage = category_percentage_first.get('Destination Food', 0)
        destination_food_message = ""
        destination_insights = ""
        if destination_food_percentage > 45:
            destination_food_message = ""
        elif destination_food_percentage <= 45:
            destination_insights = f"Destination categories which are core to your business have less than desired contribution at {destination_food_percentage:.2f}%. "
            destination_food_message = (
        " Consider strengthening destination categories to bring them to at least 45% contribution. It can even go up to 50% depending upon shopping habits of your core customers."
        " You can do it by:\n"
        "Enhancing your product range\n"
        "Ensuring good display\n"
        "Ensuring product availability\n"
        "Planning promotions"
        )
        print(destination_food_message)
        print()

        if mandatory_filename and destination_food_percentage <= 45:
            destination_food_df = combined_df[combined_df['Category Type'] == 'Destination Food']
            total_price_first_excel1 = combined_df['Price First Excel'].sum()
            total_item_sold = combined_df['Quantity Sold First Excel'].sum()
            category_price_sums_first10 = destination_food_df.groupby('Category Type')['Price First Excel'].sum()
            # category_percentage_first = {}
            # Generate a unique timestamp to append to the filename
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Destination_Food_Items_{timestamp}.xlsx"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            destination_food_df.to_excel(filepath, index=False)
            
            destination_food_df = combined_df[combined_df['Category Type'] == 'Destination Food']
            total_price_destination = combined_df['Price First Excel'].sum()
            category_price_sums_destination = destination_food_df.groupby('User Category')['Price First Excel'].sum()
            category_percentage_destinationNew = {category: (price / total_price_destination) * 100 for category, price in category_price_sums_destination.items()}
            for category, percentage in category_percentage_destinationNew.items():
                print(f"User Category '{category}' makes up {percentage:.2f}% of the Last Month total price within Destination Food.")
            data = {
                "User Category": list(category_percentage_destinationNew.keys()),
                "Percentage": [f"{percentage:.2f}%" for percentage in category_percentage_destinationNew.values()]
            }
            df = pd.DataFrame(data)
        else:
            category_percentage_destinationNew = {}

        
        # Calculate the combined percentage of Destination Food and Routine Non Core Food
        print("All Food Categories for Last Month")
        combined_percentage = category_percentage_first.get('Destination Food', 0) + category_percentage_first.get('Routine Non Core Food', 0)
        substantially_below = "substantially " if combined_percentage < 50 else ""
        food_categories_message = ""
        food_categories_insights = ""

        if combined_percentage > 65:
            food_categories_message = ""
        else:
            food_categories_insights = f"Shopping of food categories is the key motivation for monthly shopping of a family and they spend about 2/3rd on food categories. Your Food categories fall the desired level and is at {combined_percentage:.2f}%. "
            food_categories_message = "Consider strengthening of your food categories to improve sales and customer loyalty by -\nEnhancing your product range\nEnsuring good display\nEnsuring product availability\nPlanning promotions\nPremiumisation"
        # print("food categories message message for Last month")
        print(food_categories_message)
        print()

        if mandatory_filename and combined_percentage < 65:
            food_categories_df = combined_df[combined_df['Category Type'].isin(['Destination Food', 'Routine Non Core Food'])]
            # Generate a unique timestamp to append to the filename
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"food_categories_Items_{timestamp}.xlsx"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            food_categories_df.to_excel(filepath, index=False)

            food_categories_df100 = combined_df[combined_df['Category Type'].isin(['Destination Food', 'Routine Non Core Food'])]
            total_price_destination100 = combined_df['Price First Excel'].sum()
            category_price_sums_destination100 = food_categories_df100.groupby('User Category')['Price First Excel'].sum()
            category_percentage_destination100 = {category: (price / total_price_destination100) * 100 for category, price in category_price_sums_destination100.items()}
            for category, percentage in category_percentage_destination100.items():
                print(f"User Category '{category}' makes up {percentage:.2f}% of the Last Month total price within Destination Food.")
            data = {
                "User Category": list(category_percentage_destination100.keys()),
                "Percentage": [f"{percentage:.2f}%" for percentage in category_percentage_destination100.values()]
            }
            df = pd.DataFrame(data)
        else:
            category_percentage_destination100 = {}


        print("Non Food Categories for Last Month")
        routine_non_food_percentage = category_percentage_first.get('Routine Non Food', 0)
        routine_non_food_message = ""
        routine_non_food_insights = ""

        if routine_non_food_percentage < 30:
            routine_non_food_insights = f"Your non-food categories contribution is at {routine_non_food_percentage:.2f}%. There is a possibility that you are losing opportunity of selling these categories."
            routine_non_food_message = f"Consider strengthening these categories without harming your food sales by -\nEnhancing your product range\nEnsuring good display\nEnsuring product availability\nPlanning promotions\nPremiumisation"
        else:
            routine_non_food_message = ""
        # print("routine non food message message message for Last month")
        print(routine_non_food_message)
        print()
        if mandatory_filename and routine_non_food_percentage < 30:
            Routine_Non_Food_df = combined_df[combined_df['Category Type'] == 'Routine Non Food']
            # Generate a unique timestamp to append to the filename
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Routine_Non_Food_Items_{timestamp}.xlsx"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            Routine_Non_Food_df.to_excel(filepath, index=False)

            Routine_Non_Food_df101 = combined_df[combined_df['Category Type'] == 'Routine Non Food']
            total_price_destination101 = combined_df['Price First Excel'].sum()
            category_price_sums_destination101 = Routine_Non_Food_df101.groupby('User Category')['Price First Excel'].sum()
            category_percentage_destination101 = {category: (price / total_price_destination101) * 100 for category, price in category_price_sums_destination101.items()}
            for category, percentage in category_percentage_destination101.items():
                print(f"User Category '{category}' makes up {percentage:.2f}% of the Last Month total price within Destination Food.")
            data = {
                "User Category": list(category_percentage_destination101.keys()),
                "Percentage": [f"{percentage:.2f}%" for percentage in category_percentage_destination101.values()]
            }
            df = pd.DataFrame(data)
        else:
            category_percentage_destination101 = {}

        if mandatory_filename and optional_filename:
            total_quantity_first_overall = combined_df['Quantity Sold First Excel'].sum()
            total_quantity_second_overall = combined_df['Quantity Sold Second Excel'].sum()
            total_price_first_overall = combined_df['Price First Excel'].sum()
            total_price_second_overall = combined_df['Price Second Excel'].sum()

            # Safeguard against division by zero
            if total_quantity_first_overall > 0 and total_price_first_overall > 0:
                quantity_percentage100_overall = ((total_quantity_first_overall-total_quantity_second_overall)/ total_quantity_second_overall ) * 100
                price_percentage100_overall = ((total_price_first_overall-total_price_second_overall) / total_price_second_overall) * 100
                print('Growing and degrowing Groups')
                print(f"Volume Growth % for Destination Food : {quantity_percentage100_overall:.2f}%")
                print(f"Price Growth % for Destination Food: {price_percentage100_overall:.2f}%")
            else:
                print("Insufficient data for calculating percentages.")

            # Determine the growth category based on quantity_percentage
            if 0 <= quantity_percentage100_overall <= 1:
                overallvolume = ("No volume growth is seen")
            elif 1 <= quantity_percentage100_overall < 5:
                overallvolume = (f"Volume growth is at {quantity_percentage100_overall:.2f}% which is positive but low")
            elif 5 <= quantity_percentage100_overall < 10:
                overallvolume = (f"Volume growth is at {quantity_percentage100_overall:.2f}% which is excellent.")
            elif quantity_percentage100_overall > 20:
                overallvolume = (f"Volume growth is at {quantity_percentage100_overall:.2f}% which is exceptional.")
            elif -5 <= quantity_percentage100_overall < 0:
                overallvolume = (f"Volume degrowth is {quantity_percentage100_overall:.2f}%.")
            elif -10 <= quantity_percentage100_overall < -5:
                overallvolume = (f"Volume degrowth is high at {quantity_percentage100_overall:.2f}%.")
            elif quantity_percentage100_overall < -10:
                overallvolume = (f"Volume degrowth is very high at {quantity_percentage100_overall:.2f}%.")
            else:
                overallvolume = (f"Volume growth is  {quantity_percentage100_overall:.2f}%.")
            
            # Determine the growth category based on quantity_percentage
            if 0 <= price_percentage100_overall < 1:
                overallprice = ("No value growth is seen")
            elif 1 <= price_percentage100_overall < 5:
                overallprice = (f"Value growth is at {price_percentage100_overall:.2f}% which is positive but small")
            elif 5 <= price_percentage100_overall < 10:
                overallprice = (f"Value growth is at {price_percentage100_overall:.2f}% which is excellent")
            elif price_percentage100_overall > 20:
                overallprice = (f"Value growth is at {price_percentage100_overall:.2f}% which is exceptional.")
            elif -5 <= price_percentage100_overall < 0:
                overallprice = (f"Value degrowth is {price_percentage100_overall:.2f}%.")
            elif -10 <= price_percentage100_overall < -5:
                overallprice = (f"Value degrowth is high at {price_percentage100_overall:.2f}%.")
            elif price_percentage100_overall < -10:
                overallprice = (f"Value degrowth is very high at {price_percentage100_overall:.2f}%.")
            else:
                overallprice = (f"Value growth is  {price_percentage100_overall:.2f}%.")
            print()




        if mandatory_filename and optional_filename:
            # Filter for 'Destination Food' where both quantities and prices are present
            destination_food_df = combined_df[(combined_df['Category Type'] == 'Destination Food') & pd.notna(combined_df['Quantity Sold First Excel']) & pd.notna(combined_df['Quantity Sold Second Excel']) & pd.notna(combined_df['Price First Excel']) & pd.notna(combined_df['Price Second Excel'])]

            # Calculate sums and percentages
            total_quantity_first = destination_food_df['Quantity Sold First Excel'].sum()
            total_quantity_second = destination_food_df['Quantity Sold Second Excel'].sum()
            total_price_first = destination_food_df['Price First Excel'].sum()
            total_price_second = destination_food_df['Price Second Excel'].sum()

            # Safeguard against division by zero
            if total_quantity_first > 0 and total_price_first > 0:
                quantity_percentage100 = ((total_quantity_first-total_quantity_second)/ total_quantity_second ) * 100
                price_percentage100 = ((total_price_first-total_price_second) / total_price_second) * 100
                print('Growing and degrowing Groups')
                print(f"Volume Growth % for Destination Food : {quantity_percentage100:.2f}%")
                print(f"Price Growth % for Destination Food: {price_percentage100:.2f}%")
            else:
                print("Insufficient data for calculating percentages.")

            # Determine the growth category based on quantity_percentage
            if 0 <= quantity_percentage100 < 1:
                destinationvolume = ("No Volume growth is seen.")
            elif 1 <= quantity_percentage100 < 5:
                destinationvolume = (f"Volume growth is at {quantity_percentage100:.2f}% which is positive but small.")
            elif 5 <= quantity_percentage100 < 10:
                destinationvolume = (f"Volume growth is at {quantity_percentage100:.2f}% which is excellent.")
            elif quantity_percentage100 > 20:
                destinationvolume = (f"Volume growth is at {quantity_percentage100:.2f}% which is exceptional.")
            elif -5 <= quantity_percentage100 < 0:
                destinationvolume = (f"Volume degrowth is {quantity_percentage100:.2f}%.")
            elif -10 <= quantity_percentage100 < -5:
                destinationvolume = (f"Volume degrowth is high at {quantity_percentage100:.2f}%.")
            elif quantity_percentage100 < -10:
                destinationvolume = (f"Volume degrowth is very high at {quantity_percentage100:.2f}%.")
            else:
                destinationvolume = (f"Volume growth is  {quantity_percentage100:.2f}%.")
            
            # Determine the growth category based on quantity_percentage
            if 0 <= price_percentage100 < 1:
                destinationprice = ("No value growth is seen.")
            elif 1 <= price_percentage100 < 5:
                destinationprice = (f"Value growth is at {price_percentage100:.2f}% which is positive but small.")
            elif 5 <= price_percentage100 < 10:
                destinationprice = (f"Value growth is at {price_percentage100:.2f}% which is excellent.")
            elif price_percentage100 > 20:
                destinationprice = (f"Value growth is at {price_percentage100:.2f}% which is exceptional.")
            elif -5 <= price_percentage100 < 0:
                destinationprice = (f"Value degrowth is {price_percentage100:.2f}%.")
            elif -10 <= price_percentage100 < -5:
                destinationprice = (f"Value degrowth is high at {price_percentage100:.2f}%.")
            elif price_percentage100 < -10:
                destinationprice = (f"Value degrowth is very high at {price_percentage100:.2f}%.")
            else:
                destinationprice = (f"Value growth is {price_percentage100:.2f}%.")
            print()


        if mandatory_filename and optional_filename:
            # Filter for both 'Destination Food' and 'Routine Non Core Food' categories
            filtered_df = combined_df[(combined_df['Category Type'].isin(['Destination Food', 'Routine Non Core Food'])) & pd.notna(combined_df['Quantity Sold First Excel']) & pd.notna(combined_df['Quantity Sold Second Excel']) & pd.notna(combined_df['Price First Excel']) & pd.notna(combined_df['Price Second Excel'])]

            # Calculate sums
            total_quantity_first = filtered_df['Quantity Sold First Excel'].sum()
            total_quantity_second = filtered_df['Quantity Sold Second Excel'].sum()
            total_price_first = filtered_df['Price First Excel'].sum()
            total_price_second = filtered_df['Price Second Excel'].sum()

            # Calculate and print percentage changes, ensuring no division by zero
            if total_quantity_first > 0 and total_price_first > 0:
                quantity_sold_percentage_change = ((total_quantity_first-total_quantity_second) / total_quantity_second) * 100
                price_percentage_change = ((total_price_first-total_price_second)/ total_price_second ) * 100

                print(f"Volume Growth % Change for Destination Food and Routine Non Core Food: {quantity_sold_percentage_change:.2f}%")
                print(f"Price Growth % Change for Destination Food and Routine Non Core Food: {price_percentage_change:.2f}%")
            else:
                print("Insufficient data for calculating percentage changes for Destination Food and Routine Non Core Food.")

            # Determine the growth category based on quantity_percentage
            if 0 <= quantity_sold_percentage_change < 1:
                foddvolume = ("No Volume growth is seen.")
            elif 1 <= quantity_sold_percentage_change < 5:
                foddvolume = (f"Volume growth is at {quantity_sold_percentage_change:.2f}% which is positive but small.")
            elif 5 <= quantity_sold_percentage_change < 10:
                foddvolume = (f"Volume growth is at {quantity_sold_percentage_change:.2f}% which is excellent.")
            elif quantity_sold_percentage_change > 20:
                foddvolume = (f"Volume growth is at {quantity_sold_percentage_change:.2f}% which is exceptional.")
            elif -5 <= quantity_sold_percentage_change < 0:
                foddvolume = (f"Volume degrowth is {quantity_sold_percentage_change:.2f}%.")
            elif -10 <= quantity_sold_percentage_change < -5:
                foddvolume = (f"Volume degrowth is high at {quantity_sold_percentage_change:.2f}%.")
            elif quantity_sold_percentage_change < -10:
                foddvolume = (f"Volume degrowth is very high at {quantity_sold_percentage_change:.2f}%.")
            else:
                foddvolume = (f"Volume growth is {quantity_sold_percentage_change:.2f}%.")

            # Determine the growth category based on quantity_percentage
            if 0 <= price_percentage_change < 1:
                foodprice = ("No value growth is seen.")
            elif 1 <= price_percentage_change < 5:
                foodprice = (f"Value growth is at {price_percentage_change:.2f}% which is positive but small.")
            elif 5 <= price_percentage_change < 10:
                foodprice = (f"Value growth is at {price_percentage_change:.2f}% which is excellent.")
            elif price_percentage_change > 20:
                foodprice = (f"Value growth is at {price_percentage_change:.2f}% which is exceptional.")
            elif -5 <= price_percentage_change < 0:
                foodprice = (f"Value degrowth is {price_percentage_change:.2f}%.")
            elif -10 <= price_percentage_change < -5:
                foodprice = (f"Value degrowth is high at {price_percentage_change:.2f}%.")
            elif price_percentage_change < -10:
                foodprice = (f"Value degrowth is very high at {price_percentage_change:.2f}%.")
            else:
                foodprice = (f"value growth is {price_percentage_change:.2f}%.")
            print()


        if mandatory_filename and optional_filename:
            # Filter for 'Routine Non Food' where both quantities and prices are present
            destination_food_df = combined_df[(combined_df['Category Type'] == 'Routine Non Food') & pd.notna(combined_df['Quantity Sold First Excel']) & pd.notna(combined_df['Quantity Sold Second Excel']) & pd.notna(combined_df['Price First Excel']) & pd.notna(combined_df['Price Second Excel'])]

            # Calculate sums and percentages
            total_quantity_first = destination_food_df['Quantity Sold First Excel'].sum()
            total_quantity_second = destination_food_df['Quantity Sold Second Excel'].sum()
            total_price_first = destination_food_df['Price First Excel'].sum()
            total_price_second = destination_food_df['Price Second Excel'].sum()

            # Safeguard against division by zero
            if total_quantity_first > 0 and total_price_first > 0:
                quantity_percentage102 = ((total_quantity_first-total_quantity_second) / total_quantity_second) * 100
                price_percentage102 = ((total_price_first-total_price_second) / total_price_second) * 100

                print(f"Volume Growth % for Routine Non Food : {quantity_percentage102:.2f}%")
                print(f"Price Growth % for Routine Non Food: {price_percentage102:.2f}%")
            else:
                print("Insufficient data for calculating percentages.")

            # Determine the growth category based on quantity_percentage
            if 0 <= quantity_percentage102 < 1:
                routinevolume = ("No Volume growth is seen.")
            elif 1 <= quantity_percentage102 < 5:
                routinevolume = (f"Volume growth is at {quantity_percentage102:.2f}% which is positive but small.")
            elif 5 <= quantity_percentage102 < 10:
                routinevolume = (f"Volume growth is at {quantity_percentage102:.2f}% which is excellent.")
            elif quantity_percentage102 > 20:
                routinevolume = (f"Volume growth is at {quantity_percentage102:.2f}% which is exceptional.")
            elif -5 <= quantity_percentage102 < 0:
                routinevolume = (f"Volume degrowth is {quantity_percentage102:.2f}%.")
            elif -10 <= quantity_percentage102 < -5:
                routinevolume = (f"Volume degrowth is high at {quantity_percentage102:.2f}%.")
            elif quantity_percentage102 < -10:
                routinevolume = (f"Volume degrowth is very high at {quantity_percentage102:.2f}%.")
            else:
                routinevolume = (f"Volume growth is {quantity_percentage102:.2f}%.")
            
            # Determine the growth category based on quantity_percentage
            if 0 <= price_percentage102 < 1:
                routineprice = ("No Value growth is seen.")
            elif 1 <= price_percentage102 < 5:
                routineprice = (f"Value growth is at {price_percentage102:.2f}% which is positive but small.")
            elif 5 <= price_percentage102 < 10:
                routineprice = (f"Value growth is at {price_percentage102:.2f}% which is excellent.")
            elif price_percentage102 > 20:
                routineprice = (f"Value growth is at {price_percentage102:.2f}% which is exceptional.")
            elif -5 <= price_percentage102 < 0:
                routineprice = (f"Value degrowth is {price_percentage102:.2f}%.")
            elif -10 <= price_percentage102 < -5:
                routineprice = (f"Value degrowth is high at {price_percentage102:.2f}%.")
            elif price_percentage102 < -10:
                routineprice = (f"Value degrowth is very high at {price_percentage102:.2f}%.")
            else:
                routineprice = (f"Value growth is  {price_percentage102:.2f}%.")
            print()


        # Salt Circle Analysis
        # Step 1: Filter combined_df for Sugar/Salt category and Salt product
        salt_circle_df = combined_df[(combined_df['Category Predicted'] == 'Sugar/Salt') & (combined_df['Product Predicted'] == 'Salt')]

        # Step 2: Calculate No_of_Salt_Packets
        No_of_Salt_Packets = salt_circle_df['Quantity Sold First Excel'].sum()

        # Check to avoid division by zero
        if No_of_Salt_Packets > 0:
            # Step 3: Calculate No of Families Shopping Estimate
            Total_Sales = combined_df['Price First Excel'].sum()
            No_of_Families_Shopping_Estimate = Total_Sales / No_of_Salt_Packets
        else:
            No_of_Families_Shopping_Estimate = 0

        # Fixed value for Average Monthly Basket Estimate
        Average_Monthly_Basket_Estimate = 4000  # Rs

        # Step 4: Compare and determine if it's greater or less
        if No_of_Families_Shopping_Estimate > Average_Monthly_Basket_Estimate:
            # salt_msg = f"No. of Families shopping is {No_of_Salt_Packets:.0f}.\n Average Monthly Basket Estimate is Rs{No_of_Families_Shopping_Estimate:.0f}.\n Total Sales is Rs{Total_Sales:.0f}."
            salt_msg = (f"No. of Families shopping is {No_of_Salt_Packets:.0f}<br />"
                f"Average Monthly Basket Estimate is Rs {No_of_Families_Shopping_Estimate:.0f}<br />"
                f"Total Sales is Rs {Total_Sales:.0f}")
            comparison_message = f"The No of Families Shopping Estimate is greater than the Average Monthly Basket Estimate which is Rs{No_of_Families_Shopping_Estimate:.0f}"
        elif No_of_Families_Shopping_Estimate < Average_Monthly_Basket_Estimate:
            # salt_msg = f"No. of Families shopping is {No_of_Salt_Packets:.0f}.\n Average Monthly Basket Estimate is Rs{No_of_Families_Shopping_Estimate:.0f}.\n Total Sales is Rs{Total_Sales:.0f}."
            salt_msg = (f"No. of Families shopping is {No_of_Salt_Packets:.0f}<br />"
                f"Average Monthly Basket Estimate is Rs {No_of_Families_Shopping_Estimate:.0f}<br />"
                f"Total Sales is Rs {Total_Sales:.0f}")
            comparison_message = f"The No of Families Shopping Estimate is less than the Average Monthly Basket Estimate which is Rs{No_of_Families_Shopping_Estimate}"
        else:
            # salt_msg = f"No. of Families shopping is {No_of_Salt_Packets:.0f}.\n Average Monthly Basket Estimate is Rs{No_of_Families_Shopping_Estimate:.0f}.\n Total Sales is Rs{Total_Sales:.0f}."
            salt_msg = (f"No. of Families shopping is {No_of_Salt_Packets:.0f}<br />"
                f"Average Monthly Basket Estimate is Rs {No_of_Families_Shopping_Estimate:.0f}<br />"
                f"Total Sales is Rs {Total_Sales:.0f}")

        print(comparison_message)


        if mandatory_filename:
            total_sku = len(combined_df)
            total_sales = combined_df['Price First Excel'].sum()
            df_sorted = combined_df.sort_values(by='Price First Excel', ascending=False)
            df_sorted['cumulative_sales'] = df_sorted['Price First Excel'].cumsum()
            df_sorted['cumulative_percentage'] = df_sorted['cumulative_sales'] / total_sales * 100
            cutoff_index = df_sorted[df_sorted['cumulative_percentage'] >= 80].index[0]
            top_80_percent_items = df_sorted.loc[:cutoff_index]
            num_top_80_percent_items = len(top_80_percent_items)
            percentageofsku = (num_top_80_percent_items / total_sku) * 100
            # SKU_msg = f"Total SKU is {total_sku}. \nTotal Sales is Rs{total_sales:.2f}. \n{num_top_80_percent_items} SKU's contribute to 80% of sales which is {percentageofsku:.2f}% of total SKU's."
            SKU_msg = (f"Total SKU is {total_sku}.<br />"
            f"Total Sales is Rs{total_sales:.2f}.<br />"
            f"{num_top_80_percent_items} SKU's contribute to 80% of sales which is {percentageofsku:.2f}% of total SKU's.")

            # output_path = 'Top_80_Percent_High_Selling_Sku.xlsx'
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f'SKU_Contributing_80%_sales_{timestamp}.xlsx'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_path)
            top_80_percent_items.to_excel(output_path, index=False)

            top_20_items = top_80_percent_items.head(20)[['Item Name', 'Price First Excel']]

            top_20_items = top_20_items.rename(columns={
                'Price First Excel': 'Sales Last Month',
            })
            top_20_items['Sales Last Month'] = top_20_items['Sales Last Month'].astype(int)
            data103 = [top_20_items.columns.tolist()] + top_20_items.values.tolist()

        # Category Growth
        # def safe_divide(numerator, denominator):
        #     """ Prevent division by zero and handle with NaN """
        #     if denominator == 0:
        #         return 0  # Could also choose to return 0 or another placeholder value
        #     else:
        #         return (numerator - denominator) / denominator * 100

        if mandatory_filename and optional_filename:
            aggregated_data108 = combined_df.groupby('User Category').agg({
                'Quantity Sold First Excel': 'sum',
                'Quantity Sold Second Excel': 'sum',  
                "Price First Excel": "sum",
                "Price Second Excel": "sum"
            }).reset_index()

            # Select only the relevant columns for the table
            columns = ['User Category', 'Value_Growth', 'Quantity_growth']

            # Calculate value and volume growth without adding '%' yet
            aggregated_data108['Value_Growth'] = aggregated_data108.apply(
                lambda x: safe_divide(x['Price First Excel'] - x['Price Second Excel'], x['Price Second Excel']), axis=1)
            aggregated_data108['Quantity_growth'] = aggregated_data108.apply(
                lambda x: safe_divide(x['Quantity Sold First Excel'] - x['Quantity Sold Second Excel'], x['Quantity Sold Second Excel']), axis=1)

            # Get top 20 and bottom 20 categories for Value Growth
            top20Cat = aggregated_data108.nlargest(20, 'Value_Growth')   
            bottom20Cat = aggregated_data108.nsmallest(20, 'Value_Growth')

            # Format growth fields as percentage strings rounded to two decimal places
            aggregated_data108['Value_Growth'] = aggregated_data108['Value_Growth'].round(2).astype('str') + '%'
            aggregated_data108['Quantity_growth'] = aggregated_data108['Quantity_growth'].round(2).astype('str') + '%'

            top20Cat = top20Cat[columns]
            bottom20Cat = bottom20Cat[columns]

            # Format growth fields as percentage strings rounded to two decimal places for both top and bottom categories
            for df in [top20Cat, bottom20Cat]:
                df['Value_Growth'] = df['Value_Growth'].apply(lambda x: '{:.2f}%'.format(x) if pd.notnull(x) else 'nan')
                df['Quantity_growth'] = df['Quantity_growth'].apply(lambda x: '{:.2f}%'.format(x) if pd.notnull(x) else 'nan')

            # Convert to list format for ReportLab processing
            top20Cat_data = dataframe_to_list(top20Cat)
            bottom20Cat_data = dataframe_to_list(bottom20Cat)
            

            # # now we need to calculate value growth using price first excel and price second excel
            # aggregated_data108['Value_Growth'] = ((int(aggregated_data108['Price First Excel']) - int(aggregated_data108['Price Second Excel']) / int(aggregated_data108['Price Second Excel'])) * 100).astype(int).astype(str) + '%'
            # aggregated_data108['Quantity_growth'] = ((int(aggregated_data108['Quantity Sold First Excel']) - int(aggregated_data108['Quantity Sold Second Excel']) / int(aggregated_data108['Quantity Sold Second Excel'])) * 100).astype(int).astype(str) + '%'

            # # Format the growth columns for percentage display
            # aggregated_data108['Value_Growth'] = aggregated_data108['Value_Growth'].round(0).astype('Int64').astype(str) + '%'
            # aggregated_data108['Quantity_growth'] = aggregated_data108['Quantity_growth'].round(0).astype('Int64').astype(str) + '%'

            # # Get top 20 and bottom 20 categories for the selected columns
            # top20Cat_data = df_to_table_data(aggregated_data108.nlargest(20, 'Value_Growth'), columns)
            # bottom20Cat_data = df_to_table_data(aggregated_data108.nsmallest(20, 'Value_Growth'), columns)

            # # Apply nlargest and nsmallest on the numeric column
            # # top20Cat = aggregated_data108.nlargest(20, 'Value_Growth')
            # # bottom20Cat = aggregated_data108.nsmallest(20, 'Value_Growth')

            # # top20Cat_data = df_to_table_data(top20Cat)
            # # bottom20Cat_data = df_to_table_data(bottom20Cat)

            # # print()
            # # print()
            # # print(top20Cat)
            # # print()
            # # print()

            # # Convert growth fields to string and append '%' for display
            # # aggregated_data108['Value_Growth'] = aggregated_data108['Value_Growth'].round(0).astype('Int64').astype(str) + '%'
            # # aggregated_data108['Quantity_growth'] = aggregated_data108['Quantity_growth'].round(0).astype('Int64').astype(str) + '%'


        # Product Contributing 80% of Sales
        if mandatory_filename and optional_filename:
            # Assume calculate_volume_growthNew function returns percentage as string with '%'
            aggregated_data1 = combined_df.groupby('Product Predicted').agg({
                'Quantity Sold First Excel': 'sum',
                'Quantity Sold Second Excel': 'sum',  
                "Price First Excel": "sum",
                "Price Second Excel": "sum"
            }).reset_index()
            total_product = len(aggregated_data1)
            total_sales = combined_df['Price First Excel'].sum()
            df_sorted = aggregated_data1.sort_values(by='Price First Excel', ascending=False)
            df_sorted['cumulative_sales'] = df_sorted['Price First Excel'].cumsum()
            df_sorted['cumulative_percentage'] = df_sorted['cumulative_sales'] / total_sales * 100
            cutoff_index1 = df_sorted[df_sorted['cumulative_percentage'] >= 80].index[0]
            top_80_percent_items1 = df_sorted.loc[:cutoff_index1]
            num_top_80_percent_items1 = len(top_80_percent_items1)
            percentageofsku1 = (num_top_80_percent_items1 / total_product) * 100
            Product_msg = (f"Total Product is {total_product}.<br />"
            f"Total Sales is Rs{total_sales:.2f}.<br />"
            f"{num_top_80_percent_items1} Products contribute to 80% of sales which is {percentageofsku1:.2f}% of total Products.")
            top_20_items1 = top_80_percent_items1.head(20)[['Product Predicted', 'Price First Excel']]
            top_20_items1 = top_20_items1.rename(columns={
                'Product Predicted': 'Product',
                'Price First Excel': 'Sales Last Month',
            })
            top_20_items1['Sales Last Month'] = top_20_items1['Sales Last Month'].astype(int)
            data104 = [top_20_items1.columns.tolist()] + top_20_items1.values.tolist()

        # ASP product
        if mandatory_filename and optional_filename:
            aggregated_data10 = combined_df.groupby('Product Predicted').agg({
                "Price ASP First": "sum",
                "Price ASP Second": "sum"
            }).reset_index()

            # Exclude rows where 'Price ASP First' or 'Price ASP Second' is zero before calculating growth
            filtered_data10 = aggregated_data10[(aggregated_data10['Price ASP First'] != 0) & (aggregated_data10['Price ASP Second'] != 0)]

            # Calculate ASP growth only for non-zero entries and format immediately
            filtered_data10['ASP First Growth'] = ((filtered_data10['Price ASP First'] - filtered_data10['Price ASP Second']) / filtered_data10['Price ASP Second'] * 100).astype(int).astype(str) + '%'

            # Format 'Price ASP First' and 'Price ASP Second' for two decimal places
            filtered_data10['Price ASP First'] = filtered_data10['Price ASP First'].round(2)
            filtered_data10['Price ASP Second'] = filtered_data10['Price ASP Second'].round(2)
            
            # Get the top 25 ASP growth
            top_25_asp = filtered_data10.sort_values(by='ASP First Growth', ascending=False).head(20)
            top_25_asp = top_25_asp.rename(columns={
                'Product Predicted': 'Product',
                'Price ASP First': 'ASP Last Month',
                'Price ASP Second': 'ASP Previous Month',
                'ASP First Growth': 'ASP Growth (%)'
            })
            
            # Get the bottom 25 ASP growth
            bottom_25_asp = filtered_data10.sort_values(by='ASP First Growth', ascending=True).head(20)
            bottom_25_asp = bottom_25_asp.rename(columns={
                'Product Predicted': 'Product',
                'Price ASP First': 'ASP Last Month',
                'Price ASP Second': 'ASP Previous Month',
                'ASP First Growth': 'ASP Growth (%)'
            })





        if mandatory_filename:
            top20product = combined_df.groupby('Product Predicted').agg({
                'Quantity Sold First Excel': 'sum',
                'Price First Excel': 'sum',
            }).reset_index()

            # Filter out rows where both 'Price First Excel' and 'Quantity Sold First Excel' are zero
            top20product_filtered = top20product[(top20product['Price First Excel'] > 0) | (top20product['Quantity Sold First Excel'] > 0)]

            # Get the top 20 products by 'Price First Excel' (highest sales)
            top20product_top = top20product_filtered.sort_values(by='Price First Excel', ascending=False).head(20).rename(columns={
                "Product Predicted": "Product",
                'Price First Excel': 'Sales Value',
                'Quantity Sold First Excel': 'Sold Quantity'
            })

            # Convert 'Sales Value' and 'Sold Quantity' to integer
            top20product_top['Sales Value'] = top20product_top['Sales Value'].astype(int)
            top20product_top['Sold Quantity'] = top20product_top['Sold Quantity'].astype(int)

            # Get the bottom 20 products by 'Price First Excel' (lowest sales but not zero)
            top20product_bottom = top20product_filtered.sort_values(by='Price First Excel', ascending=True).head(20).rename(columns={
                "Product Predicted": "Product",
                'Price First Excel': 'Sales Value',
                'Quantity Sold First Excel': 'Sold Quantity'
            })

            # Convert 'Sales Value' and 'Sold Quantity' to integer
            top20product_bottom['Sales Value'] = top20product_bottom['Sales Value'].astype(int)
            top20product_bottom['Sold Quantity'] = top20product_bottom['Sold Quantity'].astype(int)

        

        if mandatory_filename:
            aggregated_data = combined_df.groupby('Product Predicted').agg({
                'Quantity Sold First Excel': 'sum',
                'Price First Excel': 'sum',
            }).reset_index()
            total_sales = aggregated_data['Price First Excel'].sum()
            df_sorted = aggregated_data.sort_values(by='Price First Excel', ascending=False)
            df_sorted['cumulative_sales'] = df_sorted['Price First Excel'].cumsum()
            df_sorted['cumulative_percentage'] = df_sorted['cumulative_sales'] / total_sales * 100
            cutoff_index = df_sorted[df_sorted['cumulative_percentage'] >= 80].index[0]
            top_80_percent_items = df_sorted.loc[:cutoff_index]
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f'Product_Contributing_80%_sales_{timestamp}.xlsx'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_path)
            top_80_percent_items.to_excel(output_path, index=False)


        if mandatory_filename and optional_filename:
            # Assume calculate_volume_growthNew function returns percentage as string with '%'
            aggregated_data1 = combined_df.groupby('Product Predicted').agg({
                'Quantity Sold First Excel': 'sum',
                'Quantity Sold Second Excel': 'sum',   
            }).reset_index()

            # Store 'Volume Growth' as numeric for sorting and as string for display
            aggregated_data1['Volume Growth'] = aggregated_data1.apply(calculate_volume_growthNew, axis=1)
            aggregated_data1['Volume Growth Numeric'] = aggregated_data1['Volume Growth'].str.rstrip('%').astype(float)

            # Sort by 'Volume Growth Numeric' to accurately determine top and bottom 50
            aggregated_data_Top_50 = aggregated_data1.sort_values(by='Volume Growth Numeric', ascending=False).head(50)
            aggregated_data_Bottom_50 = aggregated_data1.sort_values(by='Volume Growth Numeric', ascending=True).head(50)

            # Export paths for Excel files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            top_50_SKU = f"Top_50_Product_volume_growth_{timestamp}.xlsx"
            top_50_file_path = os.path.join(app.config['UPLOAD_FOLDER'], top_50_SKU)
            bottom_50_SKU = f"Bottom_50_Product_volume_De-Growth_{timestamp}.xlsx"
            bottom_50_file_path = os.path.join(app.config['UPLOAD_FOLDER'], bottom_50_SKU)

            # Export to Excel with percentage format in 'Volume Growth'
            aggregated_data_Top_50.to_excel(top_50_file_path, index=False, columns=['Product Predicted', 'Quantity Sold First Excel', 'Quantity Sold Second Excel', 'Volume Growth'])
            aggregated_data_Bottom_50.to_excel(bottom_50_file_path, index=False, columns=['Product Predicted', 'Quantity Sold First Excel', 'Quantity Sold Second Excel', 'Volume Growth'])

            # Prepare data for display (Top and Bottom 20)
            top_20_growth_data_Product = aggregated_data_Top_50.head(20)
            bottom_20_de_growth_data_Product = aggregated_data_Bottom_50.head(20)

            # Rename columns for display and convert 'Volume Growth' to display without '%'
            top_20_growth_data_Product_display = top_20_growth_data_Product.rename(columns={
                'Product Predicted': "Product",
                'Quantity Sold First Excel': 'Quantity Sold in Last Month',
                'Quantity Sold Second Excel': 'Quantity Sold in Previous Month',
                'Volume Growth': 'Volume Growth (%)'
            })
            bottom_20_de_growth_data_Product_display = bottom_20_de_growth_data_Product.rename(columns={
                'Product Predicted': "Product",
                'Quantity Sold First Excel': 'Quantity Sold in Last Month',
                'Quantity Sold Second Excel': 'Quantity Sold in Previous Month',
                'Volume Growth': 'Volume Growth (%)'
            })

            # Remove '%' for internal display
            top_20_growth_data_Product_display['Volume Growth (%)'] = top_20_growth_data_Product_display['Volume Growth Numeric']
            bottom_20_de_growth_data_Product_display['Volume Growth (%)'] = bottom_20_de_growth_data_Product_display['Volume Growth Numeric']

            output_path = f'Top_50_Product_volume_growth_{timestamp}.xlsx'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_path)
            aggregated_data_Top_50.to_excel(output_path, index=False)

            aggregated_data_Bottom_50 = aggregated_data1.sort_values(by='Volume Growth', ascending=True).head(20)
            output_path = f'Top_50_Product_volume_De-Growth_{timestamp}.xlsx'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_path)
            aggregated_data_Bottom_50.to_excel(output_path, index=False)

            top_20_growth_data_Product_display = aggregated_data_Top_50.rename(columns={
                'Product Predicted': "Product",
                'Quantity Sold First Excel': 'Quantity Sold in Last Month',
                'Quantity Sold Second Excel': 'Quantity Sold in Previous Month'
            })

            bottom_20_de_growth_data_Product_display = aggregated_data_Bottom_50.rename(columns={
                'Product Predicted': "Product",
                'Quantity Sold First Excel': 'Quantity Sold in Last Month',
                'Quantity Sold Second Excel': 'Quantity Sold in Previous Month'
            })


        # if mandatory_filename and optional_filename:
        #     aggregated_data1 = combined_df.groupby('Product Predicted').agg({
        #         'Quantity Sold First Excel': 'sum',
        #         'Quantity Sold Second Excel': 'sum',   
        #     }).reset_index()
        #     # aggregated_data1['Volume Growth'] = (aggregated_data1['Quantity Sold First Excel'] / aggregated_data1['Quantity Sold Second Excel']) * 100
        #     aggregated_data1['Volume Growth'] = aggregated_data1.apply(calculate_volume_growthNew, axis=1)
        #     aggregated_data_Top_50 = aggregated_data1.sort_values(by='Volume Growth', ascending=False).head(20)
        #     # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     output_path = f'Top_50_Product_volume_growth_{timestamp}.xlsx'
        #     output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_path)
        #     aggregated_data_Top_50.to_excel(output_path, index=False)
        #     aggregated_data_Bottom_50 = aggregated_data1.sort_values(by='Volume Growth', ascending=True).head(20)
        #     # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     output_path = f'Top_50_Product_volume_De-Growth_{timestamp}.xlsx'
        #     output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_path)
        #     aggregated_data_Bottom_50.to_excel(output_path, index=False)
        #     # Prepare data for PDF (Top 20 only from each)
        #     top_20_growth_data_Product = aggregated_data_Top_50.head(20)
        #     bottom_20_de_growth_data_Product = aggregated_data_Bottom_50.head(20)

        #     top_20_growth_data_Product_display = aggregated_data_Top_50.rename(columns={
        #         'Product Predicted': "Product",
        #         'Quantity Sold First Excel': 'Quantity Sold in Last Month',
        #         'Quantity Sold Second Excel': 'Quantity Sold in Previous Month'
        #     })

        #     bottom_20_de_growth_data_Product_display = aggregated_data_Bottom_50.rename(columns={
        #         'Product Predicted': "Product",
        #         'Quantity Sold First Excel': 'Quantity Sold in Last Month',
        #         'Quantity Sold Second Excel': 'Quantity Sold in Previous Month'
        #     })

            # Format Volume Growth post-calculation if not already done
            top_20_growth_data_Product_display['Volume Growth'] = top_20_growth_data_Product_display['Volume Growth'].apply(lambda x: f"{int(float(x.strip('%')))}%")
            bottom_20_de_growth_data_Product_display['Volume Growth'] = bottom_20_de_growth_data_Product_display['Volume Growth'].apply(lambda x: f"{int(float(x.strip('%')))}%")

            # Ensure the 'Volume Growth Numeric' column is not included in the DataFrame used for display or export:
            top_20_growth_data_Product_display = top_20_growth_data_Product_display[['Product', 'Quantity Sold in Last Month', 'Quantity Sold in Previous Month', 'Volume Growth']]
            bottom_20_de_growth_data_Product_display = bottom_20_de_growth_data_Product_display[['Product', 'Quantity Sold in Last Month', 'Quantity Sold in Previous Month', 'Volume Growth']]


        if mandatory_filename and optional_filename:
            # Assuming combined_df is already defined and filled with the necessary data
            # Step 1: Aggregate the data
            aggregated_data = combined_df.groupby('Product Predicted').agg({
                'Quantity Sold First Excel': 'sum',
                'Price First Excel': 'sum',
                'Quantity Sold Second Excel': 'sum',
                'Price Second Excel': 'sum'
            }).reset_index()

            # Step 2: Calculate total sales
            aggregated_data['Total Sales First Excel'] = aggregated_data['Price First Excel']
            aggregated_data['Total Sales Second Excel'] = aggregated_data['Price Second Excel']

            # Calculate the total sales from both excels
            total_sales_first = aggregated_data['Total Sales First Excel'].sum()
            total_sales_second = aggregated_data['Total Sales Second Excel'].sum()

            # Step 3: Sort data by total sales in descending order and calculate cumulative percentage
            aggregated_data_sorted_first = aggregated_data.sort_values(by='Total Sales First Excel', ascending=False)
            aggregated_data_sorted_second = aggregated_data.sort_values(by='Total Sales Second Excel', ascending=False)

            # Calculate cumulative sums
            aggregated_data_sorted_first['Cumulative Sales First'] = aggregated_data_sorted_first['Total Sales First Excel'].cumsum()
            aggregated_data_sorted_second['Cumulative Sales Second'] = aggregated_data_sorted_second['Total Sales Second Excel'].cumsum()

            # Calculate cumulative percentage
            aggregated_data_sorted_first['Cumulative Percentage First'] = 100 * aggregated_data_sorted_first['Cumulative Sales First'] / total_sales_first
            aggregated_data_sorted_second['Cumulative Percentage Second'] = 100 * aggregated_data_sorted_second['Cumulative Sales Second'] / total_sales_second

            # Find the number of products contributing to 80% of sales
            number_of_products_first = (aggregated_data_sorted_first['Cumulative Percentage First'] <= 80).sum()
            number_of_products_second = (aggregated_data_sorted_second['Cumulative Percentage Second'] <= 80).sum()

            # Filter the DataFrame for products contributing to top 80% of sales
            top_80_first = aggregated_data_sorted_first[aggregated_data_sorted_first['Cumulative Percentage First'] <= 80]
            top_80_second = aggregated_data_sorted_second[aggregated_data_sorted_second['Cumulative Percentage Second'] <= 80]

            # Combine these DataFrames to have a complete view of top contributing products
            # Optionally, you can merge them on 'Product Predicted' if they share similar products and you want to compare
            # top_80_combined = pd.merge(top_80_first, top_80_second, on='Product Predicted', suffixes=('_First', '_Second'), how='outer')

            # Save to Excel
            with pd.ExcelWriter('Top_80_Percent_Contributors.xlsx') as writer:
                top_80_first.to_excel(writer, sheet_name='Top 80% First Excel', index=False)
                top_80_second.to_excel(writer, sheet_name='Top 80% Second Excel', index=False)
                # top_80_combined.to_excel(writer, sheet_name='Combined Top 80%', index=False)

            # Print results
            print("Number of products contributing to 80% of sales in First Excel:", number_of_products_first)
            print("Number of products contributing to 80% of sales in Second Excel:", number_of_products_second)



        #SKU Sales Performance Top 50 SKU volume growth & SKU Sales Performance Bottom 50 SKU volume degrowth
        if mandatory_filename and optional_filename:
            combined_df['Quantity Growth'] = pd.to_numeric(combined_df['Quantity Growth'].str.rstrip('%'), errors='coerce')
            sorted_combined_df = combined_df.sort_values(by='Quantity Growth', ascending=False).head(50)
            top_50_SKU = f"Top50SKU_Volume_Growth_{uuid4()}.xlsx"
            top_50_file_path = os.path.join(app.config['UPLOAD_FOLDER'], top_50_SKU)   
            sorted_combined_df.to_excel(top_50_file_path, index=False)

            sorted_combined_df_degrowth = combined_df.sort_values(by='Quantity Growth', ascending=True).head(50)
            bottom_50_products_degrowth = sorted_combined_df_degrowth.drop_duplicates(subset=['Product Predicted'], keep='first').head(50)
            bottom_50_SKU = f"Bottom50SKU_Volume_De-Growth_{uuid4()}.xlsx"
            bottom_50_file_path = os.path.join(app.config['UPLOAD_FOLDER'], bottom_50_SKU)
            bottom_50_products_degrowth.to_excel(bottom_50_file_path, index=False)

            # Prepare top 20 growth and de-growth data for PDF
            top_20_growth_data = sorted_combined_df.head(20)
            columns_to_keep = ['Item Name', 'Quantity Sold First Excel', 'Quantity Sold Second Excel','Quantity Growth']
            top_20_growth_data = top_20_growth_data.loc[:, columns_to_keep].rename(columns={
                'Quantity Sold First Excel': 'Quantity Sold Last Month',
                'Quantity Sold Second Excel': 'Quantity Sold Previous Month',
                'Quantity Growth': 'Volume Growth'
            })
            bottom_20_degrowth_data = bottom_50_products_degrowth.head(20)
            bottom_20_degrowth_data = bottom_20_degrowth_data.loc[:, columns_to_keep].rename(columns={
                'Quantity Sold First Excel': 'Quantity Sold Last Month',
                'Quantity Sold Second Excel': 'Quantity Sold Previous Month',
                'Quantity Growth': 'Volume Growth'
            })
            top_20_growth_data['Volume Growth'] = top_20_growth_data['Volume Growth'].apply(lambda x: f"{x}%")
            bottom_20_degrowth_data['Volume Growth'] = bottom_20_degrowth_data['Volume Growth'].apply(lambda x: f"{x}%")

            # SKUs with less than 5 units sale IN BOTH MONTHS
            less_then_5_sku =  combined_df[(combined_df['Quantity Sold First Excel'] < 5) & (combined_df['Quantity Sold Second Excel'] < 5)].head(50)
            less_then_5_sku_filtered = less_then_5_sku[['Item Name', 'Quantity Sold First Excel', 'Quantity Sold Second Excel']].rename(
            columns={
                'Item Name': 'SKU Name',
                'Quantity Sold First Excel': 'Quantity Sold Last Month',
                'Quantity Sold Second Excel': 'Quantity Sold Previous Month'
            }
            )
            unique_filename = f'less_then_5_sku_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            less_then_5_sku.to_excel(output_path, index=False)

            # # Filter and collect details for SKUs sold less than 5 times in the mandatory file
            # mandatory_filter = combined_df[combined_df['Quantity Sold First Excel'] < 5]
            # # Similar for optional, but make sure to adjust column names as needed
            # optional_filter = combined_df[combined_df['Quantity Sold Second Excel'] < 5]
            # # Initialize combined_df with specified columns
            # combined_df5 = pd.DataFrame(columns=[
            #     'Item Name', 'User Category', 'Category Predicted', 'Product Predicted',
            #     'Brand Predicted', 'Category Type', 'Quantity Sold First Excel',
            #     'Price First Excel', 'Quantity Sold Second Excel',
            #     'Price Second Excel', 'Quantity Growth', 'Value Growth'
            # ])

            # # Process mandatory_filter DataFrame
            # new_rows = []
            # for _, row in mandatory_filter.iterrows():
            #     new_row =  {
            #         'Item Name': row['Item Name'],
            #         # Add other details similarly
            #         'Quantity Sold First Excel': row['Quantity Sold First Excel'],
            #         'Price First Excel': row['Price First Excel'],
            #         # Set optional fields to None or appropriate default
            #         'Quantity Sold Second Excel': None,
            #         'Price Second Excel': None
            #     }
            #     new_rows.append(new_row)
            # new_rows_df = pd.DataFrame(new_rows)
            # combined_df5 = pd.concat([combined_df5, new_rows_df], ignore_index=True)    
            # # Process optional_filter DataFrame
            # new_rows = []
            # for _, row in optional_filter.iterrows():
            #     if row['Item Name'] in combined_df5['Item Name'].values:
            #         # Update the existing row if SKU exists
            #         idx = combined_df5.index[combined_df5['Item Name'] == row['Item Name']].tolist()[0]
            #         combined_df5.at[idx, 'Quantity Sold Second Excel'] = row['Quantity Sold Second Excel']
            #         combined_df5.at[idx, 'Price Second Excel'] = row['Price Second Excel']
            #     else:
            #         # Append a new row if SKU doesn't exist
            #         new_row =  {
            #             'Item Name': row['Item Name'],
            #             'Quantity Sold First Excel': None,
            #             'Price First Excel': None,
            #             'Quantity Sold Second Excel': row['Quantity Sold Second Excel'],
            #             'Price Second Excel': row['Price Second Excel']
            #         }
            #         new_rows.append(new_row)
            # new_rows_df = pd.DataFrame(new_rows)
            # combined_df5 = pd.concat([combined_df5, new_rows_df], ignore_index=True)
            # output_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'SKUs_Sold_Less_Than_5_Combined.xlsx')
            # combined_df5.to_excel(output_filename, index=False)



            # Step 1: Convert 'Quantity Growth' from percentage string to numeric format
            # Note: Assuming 'Quantity Growth' is calculated and formatted as a string with a percent sign
            combined_df['Quantity Growth'] = combined_df['Quantity Growth'].astype(str)
            combined_df['Quantity Growth Numeric'] = pd.to_numeric(combined_df['Quantity Growth'].str.rstrip('%'), errors='coerce') / 100

            # Step 2: Sort the DataFrame by 'Quantity Growth Numeric' in descending order
            sorted_combined_df = combined_df.sort_values(by='Quantity Growth Numeric', ascending=False)
            # Step 3: Select the top 50 SKUs based on Quantity Growth
            top_50_skus_by_growth = sorted_combined_df.head(50)
            # Optional: If you want to drop the 'Quantity Growth Numeric' column and keep the original 'Quantity Growth' as string
            top_50_skus_by_growth = top_50_skus_by_growth.drop(columns=['Quantity Growth Numeric'])
            # Save the top 50 SKUs to a new Excel file
            top_50_growth_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Top_50_SKUs_by_Quantity_Growth.xlsx')
            top_50_skus_by_growth.to_excel(top_50_growth_filename, index=False)

            # Step 2: Sort the DataFrame by 'Quantity Growth Numeric' in ascending order to find the largest de-growth
            sorted_combined_df_for_degrowth = combined_df.sort_values(by='Quantity Growth Numeric', ascending=True)
            # Step 3: Select the top 50 SKUs based on Quantity De-growth
            top_50_skus_by_degrowth = sorted_combined_df_for_degrowth.head(50)
            # Save the top 50 SKUs with highest de-growth to a new Excel file
            top_50_degrowth_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Top_50_SKUs_by_Quantity_DeGrowth.xlsx')
            top_50_skus_by_degrowth.to_excel(top_50_degrowth_filename, index=False)

        if mandatory_filename and optional_filename:
            save_pie_chart(category_percentage_first, 'last_month_pie_chart.png', 'Last Month Category Percentages')
            save_pie_chart(category_percentage_second, 'previous_month_pie_chart.png', 'Previous Month Category Percentages')
        else:
            save_pie_chart(category_percentage_first, 'last_month_pie_chart.png', 'Last Month Category Percentages')


        # Pdf Generation
        pdf_filename = f'Sales_Analysis_Report_{timestamp}.pdf'
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
        pdf = SimpleDocTemplate(pdf_path, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        elements = []
        elements1 = []

        if mandatory_filename and optional_filename:
            # Section : Purpose
            styles.add(ParagraphStyle(name='CenteredBoldHeading2', parent=styles['Heading2'], alignment=TA_CENTER, fontName='Helvetica-Bold'))
            styles.add(ParagraphStyle(name='CenteredHeading1', parent=styles['Heading1'], alignment=TA_CENTER))

            # Header Section
            story.append(Spacer(1, 12))
            story.append(Paragraph("Sales Analysis Insight & Action Report", styles['CenteredBoldHeading2']))
            story.append(Paragraph("Month: May 2024", styles['CenteredBoldHeading2']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("This Sales Analysis Tool provides you useful information about your business at various levels like Business Groups, Categories Products and SKUs. It makes important observations that is essential for you to know and provides you recommended actions for you to consider to improve your sales performance and profits.", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("Disclaimer – The analysis has been done using computers and may have certain inaccuracies. While we are continually improving the tool, we recommend that before you take any corrective action, please verify the information and evaluate the recommendations. You should only take action when you believe it is the right thing to do for you.", styles['BodyText']))
            story.append(PageBreak())

            # Section 1: 
            story.append(Paragraph("Part 1: Key Performance Indicators", styles['Heading1']))
            story.append(Paragraph("These Parameters are indicators of your sales performance", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("Average Basket Value (ABV): Average Value of each bill", styles['BodyText']))
            story.append(Paragraph(f"(a) Average Basket Value (ABV) = Rs {ABV:.2f}", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("Average Basket Size (ABS): Average number of items in each bill", styles['BodyText']))
            story.append(Paragraph(f"(b) Average Basket Size (ABS) = {ABS:.2f} units", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("Average Selling price (ASP): Average price of each item sold", styles['BodyText']))
            story.append(Paragraph(f"(c) Average Selling price (ASP) = Rs {AIV:.2f}", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"(d) Sales Per Sq.ft = Rs {sqft:.0f}", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(Spacer(1, 12))

            story.append(Paragraph("Part 2:Family Basket", styles['Heading1']))
            story.append(Paragraph("(Based on Salt Circle)", styles['BodyText']))
            story.append(Paragraph("This part of the report shows you the estimates of numbers of families who have shopped with you in the given month and their shopping basket assessment.", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("Insight:", styles['BodyText']))
            story.append(Paragraph(f"{salt_msg}", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("Recommendation: ", styles['BodyText']))
            story.append(Paragraph("1. Keep track of trends", styles['BodyText']))
            story.append(Paragraph("2. Target more families", styles['BodyText']))
            story.append(Paragraph("3. Target improving ABS and ASP", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(PageBreak())


            story.append(Paragraph("Part 3: Business Group Performance", styles['Heading1']))
            story.append(Paragraph("This part of the analysis shows you the performance of your various business groups that consist of similar categories in the way customers see them. It considers the benchmark participation of these business groups and provides you insights and actions required,", styles['BodyText']))
            last_month_image = Image('last_month_pie_chart.png')
            last_month_image._restrictSize(400, 400)
            story.append(last_month_image)
            story.append(Spacer(1, 12))
            story.append(Paragraph("Unique User Categories by Category Type", styles['Heading2']))
            table_data = list(zip(*sorted_result_categories))
            table = Table(table_data)
            table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), '#d0d0d0'),
            ('TEXTCOLOR', (0, 0), (-1, 0), '#000000'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), '#f0f0f0'),
            ('GRID', (0, 0), (-1, -1), 1, '#000000')
            ]))
            story.append(table)
            if destination_food_message.strip():
                # story.append(Paragraph("Destination Category", styles['Heading2']))
                story.append(Paragraph("Insight:", styles['BodyText']))
                destination_insights1 = Paragraph(destination_insights, styles['BodyText'])
                story.append(destination_insights1)
                story.append(Spacer(1, 12))
                story.append(Paragraph("Recommendation:", styles['BodyText']))
                destination_food_paragraph = Paragraph(destination_food_message, styles['BodyText'])
                story.append(destination_food_paragraph)
                columns_to_display = [
                    "Item Name",
                    "Quantity Sold First Excel",
                    "Price First Excel",
                ]
            if food_categories_message.strip():
                # story.append(Paragraph("All Food Category", styles['Heading2']))
                story.append(Paragraph("Insight:", styles['BodyText']))
                line12 = Paragraph(food_categories_insights, styles['BodyText'])
                story.append(line12)
                story.append(Spacer(1, 12))
                story.append(Paragraph("Recommendation:", styles['BodyText']))
                line = Paragraph(food_categories_message, styles['BodyText'])
                story.append(line)
                story.append(Spacer(1, 12))
            if routine_non_food_message.strip():
                # story.append(Paragraph("Non Food Category", styles['Heading2']))
                story.append(Paragraph("Insight:", styles['BodyText']))
                line13 = Paragraph(routine_non_food_insights, styles['BodyText'])
                story.append(line13)
                story.append(Spacer(1, 12))
                story.append(Paragraph("Recommendation:", styles['BodyText']))
                line1 = Paragraph(routine_non_food_message, styles['BodyText'])
                story.append(line1)
                story.append(Spacer(1, 12))
            story.append(Spacer(1, 12))
            story.append(PageBreak())

            story.append(Paragraph("Part 4: Business Group Growth", styles['Heading1']))
            story.append(Paragraph("This part of the report analyses sales growth of different business groups. This helps you understand if the business groups participation is moving in the right direction against the benchmarks.", styles['BodyText']))
            story.append(Paragraph("(a) Overall Growth", styles['Heading3']))
            story.append(Paragraph(f"Volume Growth: {quantity_percentage100_overall:.2f}%", styles['BodyText']))
            story.append(Paragraph(f"   Note: {overallvolume}", styles['BodyText']))
            story.append(Paragraph(f"Value Growth: {price_percentage100_overall:.2f}%", styles['BodyText']))
            story.append(Paragraph(f"   Note: {overallprice}", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("(b) Destination Categories", styles['Heading3']))
            story.append(Paragraph(f"Volume Growth: {quantity_percentage100:.2f}%", styles['BodyText']))
            story.append(Paragraph(f"   Note: {destinationvolume}", styles['BodyText']))
            story.append(Paragraph(f"Value Growth: {price_percentage100:.2f}%", styles['BodyText']))
            story.append(Paragraph(f"   Note: {destinationprice}", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("(c) Food Categories", styles['Heading3']))
            story.append(Paragraph(f"Volume Growth: {quantity_sold_percentage_change:.2f}%", styles['BodyText']))
            story.append(Paragraph(f"   Note: {foddvolume}", styles['BodyText']))
            story.append(Paragraph(f"Value Growth: {price_percentage_change:.2f}%", styles['BodyText']))
            story.append(Paragraph(f"   Note: {foodprice}", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("(d) Non Food Categories Growth", styles['Heading3']))
            story.append(Paragraph(f"Volume Growth: {quantity_percentage102:.2f}%", styles['BodyText']))
            story.append(Paragraph(f"   Note: {routinevolume}", styles['BodyText']))
            story.append(Paragraph(f"Value Growth: {price_percentage102:.2f}%", styles['BodyText']))
            story.append(Paragraph(f"   Note: {routineprice}", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(PageBreak())


            story.append(Paragraph("Part 5: Category Growth", styles['Heading1']))
            story.append(Paragraph("This part of the report shows you how your categories have performed as compared to the previous month.", styles['BodyText']))
            story.append(Paragraph(f"Top Growth Categories", styles['Heading3']))
            story.append(Table(top20Cat_data, style=[
            ('BACKGROUND', (0,0), (-1,0), colors.gray),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ]))
            story.append(Paragraph("Bottom Growth Categories", styles['Heading3']))
            story.append(Table(bottom20Cat_data, style=[
            ('BACKGROUND', (0,0), (-1,0), colors.gray),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ]))
            story.append(PageBreak())


            story.append(Paragraph("Part 6: Product Sales Performance", styles['Heading1']))
            story.append(Paragraph("This part of the report shows you business growth at product level", styles['BodyText']))
            story.append(Paragraph("A. Highest Sales Value Products", styles['Heading2']))
            add_table_to_story(top20product_top,"Top 20 List", story, styles)
            story.append(Spacer(1, 12))
            add_table_to_story(top20product_bottom, "Bottom 20 List", story, styles)
            story.append(Spacer(1, 12))
            story.append(Paragraph("B. Highest Volume Growth Products", styles['Heading2']))
            # add_table_to_story(less_then_5_sku_filtered, "SKUs with less than 10 unit sales in the month", story, styles)
            # story.append(Paragraph("Top 20 List", styles['Heading2']))
            # story.append(Spacer(1, 12))
            
            story.append(Spacer(1, 12))
            add_table_to_story(top_20_growth_data_Product_display,"Top 20 List", story, styles)
            story.append(Paragraph("Recommendation:", styles['BodyText']))
            story.append(Paragraph("1. Review these products", styles['BodyText']))
            story.append(Paragraph("2. Analyse the causes", styles['BodyText']))
            story.append(Paragraph("3. Continue good practices", styles['BodyText']))
            story.append(Spacer(1, 12))
            # story.append(Paragraph("Bottom 20 List", styles['Heading2']))
            
            story.append(Spacer(1, 12))
            add_table_to_story(bottom_20_de_growth_data_Product_display,"Bottom 20 List", story, styles)
            story.append(Paragraph("Recommendation:", styles['BodyText']))
            story.append(Paragraph("1. Review these Products", styles['BodyText']))
            story.append(Paragraph("2. Is this a Market trend? If so, consider the change in your range.", styles['BodyText']))
            story.append(Paragraph("3. If not, Ensure these products are available in sufficient quantity and displayed well.", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("C. Product contributing to 80% Sales", styles['Heading2']))
            story.append(Paragraph("This part of the report shows you products that constitute large part (80%) of your business.", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"{Product_msg}", styles['BodyText']))
            table = Table(data104)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
                ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ]))
            story.append(table)
            story.append(Paragraph("Recommendation:", styles['BodyText']))
            story.append(Paragraph("Take extra care of these Products", styles['BodyText']))
            story.append(Paragraph("Ensure that they are:", styles['BodyText']))
            story.append(Paragraph("-Always Available", styles['BodyText']))
            story.append(Paragraph("-Displayed Well", styles['BodyText']))
            story.append(Spacer(1, 12))
            # story.append(Paragraph("These products have shown increase in Average Selling Price", styles['BodyText']))
            add_table_to_story(top_25_asp,"Top 20 ASP <br /> These products have shown increase in Average Selling Price", story, styles)
            story.append(Paragraph("Recommendation:", styles['BodyText']))
            story.append(Paragraph("Analyse the causes", styles['BodyText']))
            story.append(Paragraph("Is it because of the change in product mix", styles['BodyText']))
            story.append(Paragraph("Is it because of the price increase by manufacturers", styles['BodyText']))
            story.append(Paragraph("If conscious premiumisation is working for you, carefully keep improving the mix.", styles['BodyText']))
            story.append(Spacer(1, 12))

            # story.append(Paragraph("These products have shown decrease in Average Selling Price", styles['BodyText']))
            add_table_to_story(bottom_25_asp,"Bottom 20 ASP <br /> These products have shown decrease in Average Selling Price", story, styles)
            story.append(Paragraph("Recommendation:", styles['BodyText']))
            story.append(Paragraph("Analyse the causes", styles['BodyText']))
            story.append(Paragraph("Is it because of the change in product mix", styles['BodyText']))
            story.append(Paragraph("Is it because of the price reduction by manufacturers", styles['BodyText']))
            story.append(Paragraph("You need to reverse the trend by improving product assortment", styles['BodyText']))
            story.append(PageBreak())


            story.append(Paragraph("Part 7: SKU Sales Growth", styles['Heading1']))
            story.append(Paragraph("This part of the report shows you business growth at SKU level", styles['BodyText']))
            add_table_to_story(top_20_growth_data,"Top 20 List", story, styles)
            story.append(Spacer(1, 12))
            story.append(Paragraph("Recommendation:", styles['BodyText']))
            story.append(Paragraph("Find out the reasons:", styles['BodyText']))
            story.append(Paragraph("1. Is it a market trend?", styles['BodyText']))
            story.append(Paragraph("2. Greater Stock Availability", styles['BodyText']))
            story.append(Paragraph("3. Great Display", styles['BodyText']))
            story.append(Paragraph("4. Any other", styles['BodyText']))
            story.append(Paragraph("And take decision", styles['BodyText']))
            story.append(Spacer(1, 12))
            add_table_to_story(bottom_20_degrowth_data, "Bottom 20 List", story, styles)
            story.append(Spacer(1, 12))
            story.append(Paragraph("Recommendation:", styles['BodyText']))
            story.append(Paragraph("1. Is it a market trend?", styles['BodyText']))
            story.append(Paragraph("2. Poor Stock Availability", styles['BodyText']))
            story.append(Paragraph("3. Display Problem", styles['BodyText']))
            story.append(Paragraph("4. Any other", styles['BodyText']))
            story.append(Paragraph("And take decision", styles['BodyText']))
            story.append(Spacer(1, 12))
            add_table_to_story(less_then_5_sku_filtered, "SKUs with less than 10 unit sales in the month", story, styles)
            story.append(Paragraph("Insight: Slow selling SKUs", styles['BodyText']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("Recommendation:", styles['BodyText']))
            story.append(Paragraph("1. Consider eliminating them from the range", styles['BodyText']))
            story.append(Paragraph("2. Introduce new products in its place", styles['BodyText']))

            story.append(Spacer(1, 12))
            story.append(Paragraph("SKU contributing to 80% sales", styles['Heading1']))
            story.append(Paragraph("This part of the report shows you SKUs that constitute large part (80%) of your business.", styles['BodyText']))
            story.append(Paragraph(f"{SKU_msg}", styles['BodyText']))
            story.append(Spacer(1, 12))
            table = Table(data103)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
                ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ]))
            story.append(table)
            story.append(Spacer(1, 12))
            story.append(Paragraph("Recommendation:", styles['BodyText']))
            story.append(Paragraph("Take extra care of these SKUs", styles['BodyText']))    
            story.append(Paragraph("Ensure that they are:", styles['BodyText']))
            story.append(Paragraph("-Always Available", styles['BodyText']))
            story.append(Paragraph("-Displayed Well", styles['BodyText']))


            story.append(Spacer(1, 12))
            story.append(PageBreak())
            pdf.build(story)

            # Upload the PDF to S3
            try:
                s3.upload_file(pdf_path, S3_BUCKET, pdf_filename)
                s3_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{pdf_filename}"
                print(f"PDF uploaded to S3: {s3_url}")
            except Exception as e:
                print(f"Failed to upload PDF to S3: {e}")
                return jsonify({"error": "Failed to upload PDF to S3"}), 500

            # Save filename in session
            session['pdf_filename'] = pdf_filename

            presigned_url = s3.generate_presigned_url('get_object',
                                                    Params={'Bucket': S3_BUCKET, 'Key': pdf_filename},
                                                    ExpiresIn=3600)  # URL expires in 1 hour
            
            language_code = 'en'
            print()
            print()
            print("Mobile Number: ",mobile_number)
            print("presigned_url: ",presigned_url)
            print("language_code: ",language_code)
            print()
            print()
            response = send_interakt_message(mobile_number, presigned_url, language_code)
            print()
            print('Response: ',response)
            print()
            print('Processing complete. Structured data saved and uploaded to S3.')
            print(f"PDF generated: {pdf_filename}")
            return jsonify({
                "success": True,
                "data": {
                    "presigned_url": presigned_url
                }
            })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
